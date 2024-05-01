from __future__ import annotations

import tempfile
from pathlib import Path

from nats_foundation import SerializableObject, IMetaObject, ISerializableObject, IObjectWithMeta
from nats_foundation import calc_hash
from overrides import overrides

from .ifaces2 import IStanModelMeta, IStanModel, IMetaObjectBase, MetaObjectBase
from .utils import normalize_stan_model_by_str, find_model_in_cache
import base64
from typing import Any

class StanModelMeta(IStanModelMeta, MetaObjectBase):
    _model_name: str
    _is_canonical: bool
    _model_code_hash: str

    def __init__(self, model_hash: int, model_name: str, is_canonical: bool, model_code_hash: str):
        super().__init__(object_hash=model_hash)
        self._model_name = model_name
        self._is_canonical = is_canonical
        self._model_code_hash = model_code_hash

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







class StanModel(IStanModel, IMetaObjectBase):
    _model_name: str
    _model_file: Path
    _model_code_hash: str
    _model_opts: dict[str, Any]

    def __init__(self, model_folder: Path, model_name: str, model_code: str, model_opts: dict[str, Any]=None):
        assert isinstance(model_name, str)
        assert isinstance(model_code, str)
        assert isinstance(model_folder, Path)
        if not model_folder.exists():
            model_folder.mkdir(parents=True)

        normalized_model, msg = normalize_stan_model_by_str(model_code)
        if normalized_model is None:
            raise ValueError(f"Error in normalizing the Stan model: {msg['stanc_error']}")

        model_hash = base64.b64encode(abs(calc_hash(normalized_model)).to_bytes(32, byteorder="big")).decode("utf-8")

        if model_opts is None:
            model_opts = {}
        assert isinstance(model_opts, dict)

        super().__init__()
        self._model_file = find_model_in_cache(model_folder, model_name, model_hash)
        if not self._model_file.exists():
            self._model_file.write_bytes(normalized_model.encode())

        self._model_name = model_name
        self._model_code_hash = model_hash
        self._model_opts = model_opts



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
        return True # We do not support any other models than in canonical form

    @overrides
    def __getstate__(self) -> dict:
        d={}
        d["model_code"] = self.model_code
        d["model_name"] = self._model_name
        d["model_opts"] = self._model_opts
        return d

    @property
    @overrides
    def model_opts(self) -> dict[str, Any]:
        return self._model_opts

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

    @overrides
    def get_metaobject(self) -> IMetaObject:
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
