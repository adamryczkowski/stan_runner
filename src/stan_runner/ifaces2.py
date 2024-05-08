from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Type, Any

import numpy as np
import prettytable
from nats_foundation import ISerializableObject, IObjectWithMeta, IPrettyPrintable, IMetaObject, ObjectWithID
from nats_foundation import SerializableObject, HashContainingObject, IObjectWithID
from overrides import overrides
from prettytable import PrettyTable
import humanize
import cmdstanpy
from datetime import timedelta
from ValueWithError import IValueWithError

from .ifaces import StanResultEngine, StanOutputScope


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
    @property
    @abstractmethod
    def data_json_file(self) -> Path:
        ...

    @property
    @abstractmethod
    def data_dict(self) -> dict[str, np.ndarray | int | float]:
        ...


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

    @property
    @abstractmethod
    def compilation_time(self) -> float | None:
        ...

    @property
    @abstractmethod
    def compilation_size(self) -> int | None:
        ...

    @property
    @abstractmethod
    def is_compiled(self) -> bool:
        ...

    @overrides
    def pretty_print(self) -> str:
        ans = f"{str(type(self).__name__)} {self.object_id[0:5]} with name {self.model_name}:\n"
        ans += f"Hash: {self.object_id}, Code hash: {self.model_code_hash}\n"
        if self.is_compiled:
            ans += f"Compiled for {humanize.naturaldelta(self.compilation_time)} for an executable with the size of {humanize.naturalsize(self.compilation_size)}\n"
        else:
            ans += "Model has not been compiled\n"
        return ans

    @property
    @abstractmethod
    def exe_metadata(self)->dict[str, Any]|None:
        ...


class IStanModel(IObjectWithMeta, IStanModelMeta):

    @property
    @abstractmethod
    def model_code(self) -> str:
        ...

    @abstractmethod
    def make_sure_is_compiled(self) -> None | cmdstanpy.CmdStanModel:
        ...

    @property
    @abstractmethod
    def compilation_options(self) -> dict[str, Any]:
        ...

    @property
    @abstractmethod
    def stanc_options(self) -> dict[str, Any]:
        ...

    @property
    @abstractmethod
    def model_filename(self) -> Path | None:
        ...


class IStanRunMeta(IPrettyPrintable, IObjectWithID):
    @property
    @abstractmethod
    def run_opts(self) -> dict[str, Any]:
        ...

    @property
    @abstractmethod
    def output_scope(self) -> StanOutputScope:
        ...

    @property
    @abstractmethod
    def run_engine(self) -> StanResultEngine:
        ...

    @property
    @abstractmethod
    def sample_count(self) -> int:
        ...

    @abstractmethod
    def get_data_meta(self) -> IStanDataMeta:
        ...

    @abstractmethod
    def get_model_meta(self) -> IStanModelMeta:
        ...

    @overrides
    def pretty_print(self) -> str:
        """Pretty-prints the run. Provide information of the model and data, engine to be used, output scope and a table of all the options. """
        ans = f""""
        Run {self.object_id[0:5]} to be used with engine {self.run_engine.txt_value()}. Requested result scope: {self.output_scope.txt_value()}

* Model:
{self.get_model_meta().pretty_print()}

* Data:
{self.get_data_meta().pretty_print()}

* Options:
"""
        # Pretty table with the options
        table = PrettyTable()
        table.field_names = ["Option", "Value"]
        for k, v in self.run_opts.items():
            table.add_row([k, v])

        ans += str(table)
        ans += f"""
Return scope: {self.output_scope.txt_value()}
"""
        return ans



class IResultPromise(ABC):
    pass


class IStanRun(ISerializableObject, IObjectWithMeta, IStanRunMeta):
    @abstractmethod
    def run(self) -> IStanResultBase:
        ...

    @property
    @abstractmethod
    def run_folder(self) -> str:
        ...


class IStanResultMeta(IMetaObjectBase, IPrettyPrintable, IObjectWithID):
    @property
    @abstractmethod
    def is_error(self) -> bool:
        ...

    @property
    @abstractmethod
    def runtime(self) -> timedelta | None:
        ...

    @property
    @abstractmethod
    def data_hash(self) -> int:
        ...

    @property
    @abstractmethod
    def model_hash(self) -> int:
        ...

    @property
    @abstractmethod
    def run_hash(self) -> int:
        ...

    @abstractmethod
    def get_data_meta(self) -> IStanDataMeta:
        ...

    @abstractmethod
    def get_model_meta(self) -> IStanModelMeta:
        ...

    @abstractmethod
    def get_run_meta(self) -> IStanRunMeta:
        ...

    @property
    @abstractmethod
    def result_type(self) -> StanResultEngine:
        ...

    @property
    @abstractmethod
    def output_scope(self) -> StanOutputScope:
        ...

    @property
    @abstractmethod
    def one_dim_parameters_count(self) -> int:
        ...

    @abstractmethod
    def get_onedim_parameter_names(self, user_parameter_name: str) -> list[str]:
        ...

    @property
    @abstractmethod
    def user_parameters(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def onedim_parameters(self) -> list[str]:
        ...

    @abstractmethod
    def get_parameter_shape(self, user_parameter_name: str) -> tuple[int, ...]:
        ...

    @abstractmethod
    def sample_count(self, onedim_parameter_name: str = None) -> float | int | None:
        ...

    @abstractmethod
    def formatted_runtime(self) -> str:
        ...

    @abstractmethod
    def repr_with_sampling_errors(self):
        ...

    @abstractmethod
    def repr_without_sampling_errors(self):
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        ...


    @property
    @abstractmethod
    def requested_algorithm_variation(self)->str:
        ...


class ImplementationOfUser2OneDim(ABC):
    @abstractmethod
    def _get_user2onedim(self) -> dict[str, list[str]]:
        pass

    @abstractmethod
    def _get_user2dim(self) -> dict[str, tuple[int, ...]]:
        pass

    @property
    # @overrides
    def one_dim_parameters_count(self) -> int:
        return len(self._get_user2onedim())

    # @overrides
    def get_onedim_parameter_names(self, user_parameter_name: str) -> list[str]:
        return self._get_user2onedim()[user_parameter_name]

    @property
    # @overrides
    def user_parameters(self) -> list[str]:
        return list(self._get_user2onedim().keys())

    @property
    # @overrides
    def onedim_parameters(self) -> list[str]:
        return [item for sublist in self._get_user2onedim().values() for item in sublist]

    # @overrides
    def get_parameter_shape(self, user_parameter_name: str) -> tuple[int, ...]:
        return self._get_user2dim()[user_parameter_name]


class IStanResultBase(IObjectWithMeta, IStanResultMeta):
    @abstractmethod
    def get_parameter_estimates(self, user_parameter_name: str,
                                store_values: bool = False) -> Any:  # Sufficiently nested list of IValueWithError
        ...

    @abstractmethod
    def get_onedim_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> IValueWithError:
        ...

    @abstractmethod
    def get_parameter_mu(self, user_parameter_name: str) -> np.ndarray:
        ...

    @abstractmethod
    def get_parameter_sigma(self, user_parameter_name: str) -> np.ndarray:
        ...

    @abstractmethod
    def all_main_effects_onedim_par(self) -> dict[str, IValueWithError]:
        ...

    @property
    @abstractmethod
    def messages(self) -> str:
        ...

    @property
    @abstractmethod
    def warnings(self) -> str:
        ...

    @property
    @abstractmethod
    def errors(self) -> str:
        ...


class IStanResultCovariances(IStanResultBase):
    @abstractmethod
    def get_cov_onedim_par(self, one_dim_par1: str, one_dim_par2: str) -> float | np.ndarray:
        ...

    @abstractmethod
    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        ...

    @abstractmethod
    def pretty_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> str:
        ...

    @abstractmethod
    def downcast_to_main_effects(self) -> IStanResultBase:
        ...


class IStanResultFullSamples(IStanResultCovariances):
    @abstractmethod
    def draws(self, onedim_varname:str) -> np.ndarray:
        ...

    @abstractmethod
    def downcast_to_covariances(self) -> IStanResultCovariances:
        ...


class IStanResultRawResult(IStanResultFullSamples):
    @property
    @abstractmethod
    def get_raw_result(self) -> Any:  # e.g. CmdStanVB
        ...

    @abstractmethod
    def draws_of_all_variables(self, bInclSpecial:bool=False) -> tuple[np.ndarray, tuple[str, ...]]:
        ...

    @abstractmethod
    def serialize_to_file(self) -> Path:
        ...

    @abstractmethod
    def downcast_to_full_samples(self) -> IStanResultFullSamples:
        ...