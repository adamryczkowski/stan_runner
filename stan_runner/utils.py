from hashlib import sha256
from pathlib import Path
from math import prod

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
