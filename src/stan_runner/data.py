from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from nats_foundation import ISerializableObject, \
    create_dict_serializable_copy, calc_hash, IMetaObject
from overrides import overrides

from .ifaces2 import StanDataType, StanInputVariable, IStanDataMeta, MetaObjectBase, IMetaObjectBase, IStanData


class StanDataMeta(IStanDataMeta, MetaObjectBase):
    _data_hash: int
    _variables: dict[str, StanInputVariable]

    def __init__(self, data_hash: int, variables: dict[str, StanInputVariable]):
        super().__init__(data_hash)
        assert isinstance(variables, dict)
        for k, v in variables.items():
            assert isinstance(k, str)
            assert isinstance(v, StanInputVariable)
        self._variables = variables

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["data_hash"] = self._data_hash
        d["variables"] = {k: v.__getstate__() for k, v in self._variables.items()}
        return d

    @property
    @overrides
    def is_object_available(self) -> bool:
        return False

    @overrides
    def get_object(self) -> ISerializableObject:
        raise NotImplementedError

    @property
    @overrides
    def variable_names(self) -> list[str]:
        return list(self._variables.keys())

    @overrides
    def get_variable_meta(self, name: str) -> StanInputVariable:
        return self._variables[name]



class StanData(IStanData, IMetaObjectBase):
    _data_file: Path
    _data: dict
    _data_opts: dict[str, str]  # Options specifying to the data

    # noinspection PyMethodOverriding

    def __init__(self, run_folder: Path, data: dict[str, float | int | np.ndarray], data_opts: dict[str, str] = None):
        if data_opts is None:
            data_opts = {}
        assert isinstance(data_opts, dict)
        assert isinstance(data, dict)
        assert isinstance(run_folder, Path)
        new_data = self._normalize_and_verify_data(data)
        if not run_folder.exists():
            run_folder.mkdir(parents=True)
        super().__init__()
        self._data_file = run_folder / "data.json"
        if self._data_file.exists():
            self._data_file.unlink()
        self._data_file.write_text(json.dumps(create_dict_serializable_copy(new_data)))
        self._data_opts = data_opts
        self._data = new_data

    @staticmethod
    def _normalize_and_verify_data \
                    (data: dict[str, list | float | int | np.ndarray]) -> dict[str, list | float | int | np.ndarray]:
        new_data = {}
        for key, data_item in data.items():
            if isinstance(data_item, list):
                data_item = np.asarray(data_item)
                new_data[key] = data_item
            if not isinstance(key, str):
                raise ValueError(f"Data key {key} is not a string")
            if isinstance(data_item, np.ndarray):
                if data_item.dtype not in [np.float32, np.float64, np.int32, np.int64]:
                    raise ValueError(f"Data {key} type {data_item.dtype} not supported")
                new_data[key] = data_item
            elif isinstance(data_item, (float, int)):
                new_data[key] = data_item
            else:
                raise ValueError(f"Data type {type(data_item)} not supported")
        # Serialize data to json and store it into self._data_file
        return new_data

    @overrides
    def __getstate__(self) -> dict:
        d = {}
        d["_data"] = create_dict_serializable_copy(self._data)
        d["_data_opts"] = self._data_opts
        return d

    @property
    @overrides
    def variable_names(self) -> list[str]:
        return list(self._data.keys())

    @overrides
    def get_variable_meta(self, name: str) -> StanInputVariable:
        var = self._data[name]
        if isinstance(var, np.ndarray):
            return StanInputVariable(name, list(var.shape), StanDataType.from_numpy_dtype(var.dtype))
        else:
            return StanInputVariable(name, [1], StanDataType.from_numpy_dtype(type(var)))

    @overrides
    def get_object(self) -> ISerializableObject:
        return self

    @overrides(check_signature=False)
    def get_metaobject(self) -> IStanDataMeta:
        return StanDataMeta(self.object_hash, {k: self.get_variable_meta(k) for k in self.variable_names})

    @property
    @overrides
    def is_object_available(self) -> bool:
        return True

    @property
    @overrides
    def object_hash(self) -> int:
        return calc_hash(self.__getstate__())

    @property
    def data_json_file(self) -> Path:
        return self._data_file

    @property
    def data_dict(self) -> dict[str, np.ndarray | int | float]:
        return self._data




