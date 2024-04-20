import subprocess
import io
import json
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from hashlib import sha256
from math import prod
from pathlib import Path
import pickle
from typing import Any, Optional

import cmdstanpy
import numpy as np

from .ifaces import StanOutputScope


def find_model_in_cache(model_cache: Path, model_name: str, model_hash: str) -> Path:
    best_model_filename = None
    for hash_char_count in range(0, len(model_hash)):
        if hash_char_count == 0:
            model_filename = model_cache / f"{model_name}.stan"
        else:
            model_filename = model_cache / f"{model_name} {model_hash[:hash_char_count]}.stan"
        if model_filename.exists():
            # load the model and check its hash if it matches
            with model_filename.open("r") as f:
                model_code = f.read()
            if sha256(model_code.encode()).hexdigest() == model_hash:
                if hash_char_count == len(model_hash) - 1:
                    raise RuntimeError(
                        f"Hash collision for model {model_name} with hash {model_hash} in cache file {model_filename}!")
                return model_filename
        else:
            if best_model_filename is None:
                best_model_filename = model_filename

        hash_char_count += 1

    assert best_model_filename is not None
    return best_model_filename


def infer_param_shapes(one_dim_names: list[str]) -> tuple[dict[str, tuple[int, ...]], dict[str, list[str]]]:
    ans_dims: dict[str, tuple[int, ...]] = {}
    ans_dict: dict[str, list[str]] = {}
    shape = None

    def str2shape(s: str) -> tuple[int, ...]:
        return tuple(map(int, s.split(",")))

    def bigger_shape(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(max(s1[i], s2[i]) for i in range(len(s1)))

    old_base_name = ""
    var_count = 0
    name_list = []
    shape = None
    for one_dim_name in one_dim_names:
        new_base_name = one_dim_name.split("[")[0]

        if new_base_name != old_base_name:
            if shape is not None:
                # Commits variable
                assert var_count == prod(shape)
                assert old_base_name not in ans_dims
                assert old_base_name not in ans_dict
                ans_dims[old_base_name] = shape
                ans_dict[old_base_name] = name_list
                shape = None

            # Iteration restart
            old_base_name = new_base_name
            var_count = 1
            name_list = [one_dim_name]

        else:
            var_count += 1
            name_list.append(one_dim_name)

        if "[" not in one_dim_name:
            shape = (1,)
        else:
            name_coords = one_dim_name.split("[")[1][:-1]
            coords = str2shape(name_coords)

            if shape is None:
                shape = coords
            else:
                shape = bigger_shape(shape, coords)

    if shape is not None:
        # Commits variable
        assert var_count == prod(shape)
        assert old_base_name not in ans_dims
        assert old_base_name not in ans_dict
        ans_dims[old_base_name] = shape
        ans_dict[old_base_name] = name_list

    return ans_dims, ans_dict


def make_dict_serializable(d: dict) -> dict:
    """Turns all numpy arrays in the dictionary into lists"""
    for key in d:
        if isinstance(d[key], dict):
            d[key] = make_dict_serializable(d[key])
        elif isinstance(d[key], np.ndarray):
            d[key] = d[key].tolist()
    return d


def denumpy_dict(d: dict[str, Any]) -> dict[str, Any]:
    ans = {}
    for key in d:
        if isinstance(d[key], np.ndarray):
            ans[key] = d[key].tolist()
        elif isinstance(d[key], dict):
            ans[key] = denumpy_dict(d[key])
        else:
            ans[key] = d[key]
    return ans


def model2json(stan_code: str, model_name: str, data: dict, output_type: StanOutputScope,
               **kwargs) -> str:
    assert isinstance(stan_code, str)
    assert isinstance(model_name, str)
    assert isinstance(data, dict)
    # assert isinstance(engine, StanResultEngine)
    assert isinstance(output_type, StanOutputScope)

    out = {}
    out["model_code"] = stan_code
    out["model_name"] = model_name
    out["data"] = denumpy_dict(data)
    out["output_type"] = output_type.txt_value()
    out.update(kwargs)

    # Convert out to json
    return str(json.dumps(out))


def normalize_stan_model_by_str(stan_code: str) -> tuple[Optional[str], dict[str, str]]:
    # Write the model to a disposable temporary location
    with tempfile.NamedTemporaryFile("w", delete=True) as f:
        f.write(stan_code)
        model_filename = Path(f.name)

        file_out, msg = normalize_stan_model_by_file(model_filename)
        if file_out is None:
            return None, msg

        return file_out.read_text(), msg


def normalize_stan_model_by_file(stan_file: str | Path) -> tuple[Optional[Path], dict[str, str]]:
    if isinstance(stan_file, str):
        stan_file = Path(stan_file)
    assert isinstance(stan_file, Path)
    assert stan_file.exists()
    assert stan_file.is_file()

    stdout = io.StringIO()
    stderr = io.StringIO()

    # Copy stan_file to temporary location
    tmpfile = tempfile.NamedTemporaryFile("w", delete=False)
    tmpfile.write(stan_file.read_bytes().decode())
    tmpfile.close()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            cmdstanpy.format_stan_file(tmpfile.name, overwrite_file=True,
                                       canonicalize=["deprecations", "parentheses", "braces", "includes",
                                                     "strip-comments"])
        except subprocess.CalledProcessError as e:
            msg = {"stanc_output": stdout.getvalue() + e.stdout,
                   "stanc_warning": stderr.getvalue(),
                   "stanc_error": e.stderr}
            return None, msg

    msg = {"stanc_output": stdout.getvalue(),
           "stanc_warning": stderr.getvalue()}

    return Path(tmpfile.name), msg

def serialize_to_bytes(obj: Any, format:str) -> bytes:
    if format == "pickle":
        if isinstance(obj, dict):
            obj = make_dict_serializable(obj)
        return pickle.dumps(obj)
    else:
        raise Exception(f"Unknown format {format}")


def human_readable_size(size, decimal_places=2):
    # Credit to: https://stackoverflow.com/a/43690506
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"
