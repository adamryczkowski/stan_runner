from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from nats_foundation import IMetaObject, ISerializableObject
from nats_foundation import calc_hash
from overrides import overrides

from .ifaces2 import IStanModelMeta, IStanModel, IMetaObjectBase, MetaObjectBase
from .utils import normalize_stan_model_by_str, find_model_in_cache
from .pystan_run import compile_model
import cmdstanpy


class StanModelMeta(IStanModelMeta, MetaObjectBase):
    _model_name: str
    _is_canonical: bool
    _model_code_hash: str
    _compilation_time: float | None
    _executable_size: int | None

    def __init__(self, model_hash: int, model_name: str, is_canonical: bool, model_code_hash: str,
                 compilation_time: float = -1.0, executable_size: int = -1):
        super().__init__(object_hash=model_hash)
        self._model_name = model_name
        self._is_canonical = is_canonical
        self._model_code_hash = model_code_hash
        self._compilation_time = compilation_time
        self._executable_size = executable_size

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["model_name"] = self._model_name
        d["is_canonical"] = self._is_canonical
        d["model_code_hash"] = self._model_code_hash
        return d

    @property
    @overrides
    def model_name(self) -> str:
        return self._model_name

    @property
    @overrides
    def is_canonical(self) -> bool:
        return self._is_canonical

    @property
    @overrides
    def model_code_hash(self) -> str:
        return self._model_code_hash

    @property
    @overrides
    def compilation_time(self) -> float | None:
        if self.is_compiled:
            return self._compilation_time
        return None

    @property
    @overrides
    def compilation_size(self) -> int | None:
        if self.is_compiled:
            return self._executable_size
        return None

    @property
    @overrides
    def is_compiled(self) -> bool:
        return self._compilation_time >= 0.0


class StanModel(IStanModel, IMetaObjectBase):
    _model_name: str
    _model_file: Path
    _model_code_hash: str
    _compiled_model: cmdstanpy.CmdStanModel | None
    _stanc_opts: dict[str, Any]
    _cpp_opts: dict[str, Any]
    _compilation_time: float

    def __init__(self, model_folder: Path, model_name: str, model_code: str, stanc_opts: dict[str, Any] = None,
                 cpp_opts: dict[str, Any] = None):
        assert isinstance(model_name, str)
        assert isinstance(model_code, str)
        assert isinstance(model_folder, Path)
        if not model_folder.exists():
            model_folder.mkdir(parents=True)

        normalized_model, msg = normalize_stan_model_by_str(model_code)
        if normalized_model is None:
            raise ValueError(f"Error in normalizing the Stan model: {msg['stanc_error']}")

        model_hash = base64.b64encode(abs(calc_hash(normalized_model)).to_bytes(32, byteorder="big")).decode("utf-8")

        if stanc_opts is None:
            stanc_opts = {}
        assert isinstance(stanc_opts, dict)

        if cpp_opts is None:
            cpp_opts = {}
        assert isinstance(cpp_opts, dict)

        super().__init__()
        self._model_file = find_model_in_cache(model_folder, model_name, model_hash)
        if not self._model_file.exists():
            self._model_file.write_bytes(normalized_model.encode())

        if self.compiled_model_file.with_suffix(".time").exists():
            with open(self.compiled_model_file.with_suffix(".time"), "r") as f:
                self._compilation_time = float(f.read())

        self._model_name = model_name
        self._model_code_hash = model_hash
        self._stanc_opts = stanc_opts
        self._cpp_opts = cpp_opts
        self._compilation_time = -1.0

    @property
    @overrides
    def model_name(self) -> str:
        return self._model_name

    @property
    @overrides
    def model_code_hash(self) -> str:
        return self._model_code_hash

    @property
    @overrides
    def is_canonical(self) -> bool:
        return True  # We do not support any other models than in canonical form

    @overrides
    def __getstate__(self) -> dict:
        d = {}
        d["model_code"] = self.model_code
        d["model_name"] = self._model_name
        d["stanc_opts"] = self._stanc_opts
        d["cpp_opts"] = self._cpp_opts

        return d

    @overrides
    def get_object(self) -> ISerializableObject:
        return self

    @property
    @overrides
    def is_object_available(self) -> bool:
        return True

    @property
    def object_hash(self) -> int:
        return calc_hash(self.__getstate__())

    @overrides(check_signature=False)
    def get_metaobject(self) -> IStanModelMeta:
        return StanModelMeta(self.object_hash, self._model_name, self.is_canonical, self._model_code_hash)

    @property
    @overrides
    def model_code(self) -> str:
        return self._model_file.read_text()

    @overrides
    def pretty_print(self) -> str:
        ans = super().pretty_print()
        ans += "\nCode:\n"
        ans += self.model_code
        return ans

    @property
    @overrides
    def is_compiled(self) -> bool:
        return self._compilation_time >= 0.0

    @overrides
    def make_sure_is_compiled(self) -> None:
        if self._compiled_model is None:
            if not self.compiled_model_file.exists():
                time_taken, exe_file, err, model = compile_model(self)
                if exe_file is None:
                    raise ValueError(f"Error in compiling the model: {err}")
                self._compilation_time = time_taken
                self._compiled_model = model
            else:
                # TODO: Check if the following line works
                self._compiled_model = cmdstanpy.CmdStanModel(exe_file=str(self.compiled_model_file))

        return self._compiled_model

    @property
    @overrides
    def compilation_time(self) -> float | None:
        if self._compilation_time < 0.0:
            return None
        return self._compilation_time

    @property
    def compiled_model_file(self) -> Path:
        return self._model_file.with_suffix("")

    @property
    @overrides
    def compilation_size(self) -> int | None:
        if self.compiled_model_file.exists():
            return self.compiled_model_file.stat().st_size
        return None

    @property
    @overrides
    def compilation_options(self) -> dict[str, Any]:
        return self._cpp_opts

    @property
    @overrides
    def stanc_options(self) -> dict[str, Any]:
        return self._stanc_opts

    @property
    @overrides
    def model_filename(self) -> Path | None:
        return self._model_file
