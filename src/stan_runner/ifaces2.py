from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Type, Any

import numpy as np
import prettytable
from nats_foundation import ISerializableObject, IObjectWithMeta, IPrettyPrintable, IMetaObject, ObjectWithID
from nats_foundation import SerializableObject, HashContainingObject, IObjectWithID
from overrides import overrides


class IMetaObjectBase(SerializableObject, ObjectWithID, IMetaObject):
    pass


class MetaObjectBase(HashContainingObject, IMetaObjectBase):
    """Base object for all meta objects created by pyStan. These objects are to be serialized only - no deserialization."""

    def __init__(self, object_hash: int):
        super().__init__(object_hash)

    @abstractmethod
    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["_object_hash"] = self._object_hash
        return d

    @overrides
    def get_object(self) -> ISerializableObject:
        assert False

    @property
    @overrides
    def is_object_available(self) -> bool:
        return False


class StanDataType(Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"

    @staticmethod
    def from_numpy_dtype(dtype: np.dtype | Type) -> StanDataType:
        if dtype in [np.int32, np.int64, int]:
            return StanDataType.INT
        elif dtype in [np.float32, np.float64, float]:
            return StanDataType.FLOAT
        elif dtype == bool:
            return StanDataType.BOOL
        else:
            assert False


@dataclass
class StanInputVariable:
    name: str
    dims: list[int]
    type: StanDataType

    def __getstate__(self) -> dict:
        return {
            "name": self.name,
            "dims": self.dims,
            "data": self.data
        }

    def __setstate__(self, state: dict):
        self.name = state["name"]
        self.dims = state["dims"]
        self.data = state["data"]

    def pretty_print(self) -> str:
        return f"{self.name} {self.dims} {self.type.name}"


class IStanDataMeta(IPrettyPrintable, IObjectWithID):
    @property
    @abstractmethod
    def variable_names(self) -> list[str]:
        ...

    @abstractmethod
    def get_variable_meta(self, name: str) -> StanInputVariable:
        ...

    @overrides
    def pretty_print(self) -> str:
        ans = f"{str(type(self).__name__)} {self.object_id[0:5]}:\n"

        # Returns a pretty table of variable names, types and shapes
        table = prettytable.PrettyTable()
        table.field_names = ["Input", "type", "shape"]
        table.align["Input"] = "l"
        table.align["type"] = "c"
        table.align["shape"] = "r"
        for name in self.variable_names:
            var = self.get_variable_meta(name)
            if var.type == StanDataType.INT:
                var_type = "int"
            elif var.type == StanDataType.FLOAT:
                var_type = "float"
            elif var.type == StanDataType.BOOL:
                var_type = "int (bool)"
            else:
                assert False

            if len(var.dims) == 0 or (len(var.dims) == 1 and var.dims[0] == 1):
                var_shape = "scalar"
            else:
                var_shape = "×".join([str(x) for x in var.dims])
            table.add_row([name, var_type, var_shape])
        return ans + table.get_string()


class IStanData(IObjectWithMeta, IStanDataMeta):
    pass


class IStanModelMeta(IPrettyPrintable, IObjectWithID):
    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_code_hash(self) -> str:
        ...

    @property
    @abstractmethod
    def is_canonical(self) -> bool:
        ...

    @overrides
    def pretty_print(self) -> str:
        ans = f"{str(type(self).__name__)} {self.object_id[0:5]} with name {self.model_name}:\n"
        ans += f"Hash: {self.object_id}, Code hash: {self.model_code_hash}\n"
        return ans


class IStanModel(IObjectWithMeta, IStanModelMeta):
    @property
    @abstractmethod
    def model_opts(self) -> dict[str, Any]:
        ...

    @property
    @abstractmethod
    def model_code(self) -> str:
        ...
