import math
from abc import abstractmethod
from datetime import timedelta
from typing import Any

import humanize
import numpy as np
import prettytable
from nats_foundation import ISerializableObject, IMetaObject
from overrides import overrides

from ValueWithError import ValueWithError, IValueWithError
from . import StanResultEngine, StanOutputScope
from .ifaces2 import IStanResultBase, IMetaObjectBase, IStanDataMeta, IStanModelMeta, \
    IStanRunMeta, IStanRun, IStanResultMeta, IStanResultCovariances
from .utils import infer_param_shapes
import datetime


class ImplStanResultMetaWithUser2Onedim(IStanResultMeta):

    @abstractmethod
    def _get_user2onedim(self) -> dict[str, list[str]]:
        pass

    @abstractmethod
    def _get_user2dim(self) -> dict[str, tuple[int, ...]]:
        pass

    @property
    @overrides
    def one_dim_parameters_count(self) -> int:
        return len(self._get_user2onedim())

    @overrides
    def get_onedim_parameter_names(self, user_parameter_name: str) -> list[str]:
        return self._get_user2onedim()[user_parameter_name]

    @property
    @overrides
    def user_parameters(self) -> list[str]:
        return list(self._get_user2onedim().keys())

    @property
    @overrides
    def onedim_parameters(self) -> list[str]:
        return [item for sublist in self._get_user2onedim().values() for item in sublist]

    @overrides
    def get_parameter_shape(self, user_parameter_name: str) -> tuple[int, ...]:
        return self._get_user2dim()[user_parameter_name]



class ImplAlgorithmMetadata(IStanResultMeta):
    _algorithm: str

    def __init__(self, algorithm: str, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(algorithm, str)
        self._algorithm = algorithm

    @property
    @overrides
    def requested_algorithm_variation(self) -> str:
        return self._algorithm

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["algorithm"] = self._algorithm
        return d


class StanResultMeta(ImplStanResultMetaWithUser2Onedim, IStanResultMeta):
    _runtime: float
    _errors: str
    _run: IStanRunMeta
    _method_name: str
    _user2onedim: dict[str, list[str]] | None  # Translates user parameter names to one-dim parameter names
    _user2dim: dict[str, tuple[int, ...]] | None  # Translates user parameter names to shapes

    def __init__(self, run: IStanRunMeta, runtime: float, errors: str, method_name: str,
                 onedim_names: list[str] | set[str], sample_count: int, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(run, IStanRunMeta)
        assert isinstance(runtime, float)
        assert isinstance(errors, str)
        assert isinstance(method_name, str)
        assert onedim_names is not None
        if isinstance(onedim_names, list):
            assert len(onedim_names) == len(set(onedim_names))
        assert isinstance(sample_count, int)
        self._user2dim, self._user2onedim = infer_param_shapes(onedim_names)
        for k in self._user2onedim.keys():
            self._user2dim[k] = self._user2dim[k] + (sample_count,)
        self._run = run
        self._runtime = runtime
        self._errors = errors
        self._method_name = method_name

    @overrides
    def _get_user2onedim(self) -> dict[str, list[str]]:
        return self._user2onedim

    @overrides
    def _get_user2dim(self) -> dict[str, tuple[int, ...]]:
        return self._user2dim

    @property
    @overrides
    def is_error(self) -> bool:
        return self._errors == ""

    @property
    @overrides
    def runtime(self) -> timedelta | None:
        if self._runtime is None:
            return None
        return timedelta(seconds=self._runtime)

    @property
    @overrides
    def data_hash(self) -> int:
        return self._run.get_data_meta().object_hash

    @property
    @overrides
    def model_hash(self) -> int:
        return self._run.get_model_meta().object_hash

    @property
    @overrides
    def run_hash(self) -> int:
        return self._run.object_hash

    @property
    @overrides
    def object_hash(self) -> int:
        return self.run_hash

    @overrides
    def get_object(self) -> ISerializableObject:
        raise NotImplementedError

    @property
    @overrides
    def is_object_available(self) -> bool:
        return False

    @overrides
    def get_data_meta(self) -> IStanDataMeta:
        return self._run.get_data_meta()

    @overrides
    def get_model_meta(self) -> IStanModelMeta:
        return self._run.get_model_meta()

    @overrides
    def get_run_meta(self) -> IStanRunMeta:
        return self._run

    @property
    @overrides
    def result_type(self) -> StanResultEngine:
        return self._run.run_engine

    @property
    @overrides
    def output_scope(self) -> StanOutputScope:
        return self._run.output_scope

    @overrides
    def pretty_print(self) -> str:
        ans = f"""Result Meta after a {'successful' if not self.is_error else 'failed'} run:
Data:
{self.get_data_meta().pretty_print()}

Model:
{self.get_model_meta().pretty_print()}

Run:
{self.get_run_meta().pretty_print()}
"""
        return ans

    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["runtime"] = self._runtime
        d["status"] = "error" if self.is_error else "success"
        d["run_hash"] = self.run_hash
        d["model_hash"] = self.model_hash
        d["data_hash"] = self.data_hash
        d["output_scope"] = self.output_scope.txt_value()
        d["algorithm"] = self._algorithm

        return d

    @overrides
    def sample_count(self, onedim_parameter_name: str = None) -> float | int | None:
        """To override in a subclass for the MCMC engine"""
        if onedim_parameter_name is None:
            return self._user2dim[next(iter(self._user2dim))][-1]
        return self._user2dim[onedim_parameter_name][-1]

    @overrides
    def formatted_runtime(self) -> str:
        if self._runtime is None:
            return "Run time: not available"
        else:
            return f"Run taken: {humanize.precisedelta(datetime.timedelta(seconds=self._runtime))}"

    @property
    @overrides
    def method_name(self) -> str:
        return self._method_name


class ImplStanResultBase(IStanResultBase):
    _run: IStanRun
    _output: str
    _warnings: str
    _errors: str
    _runtime: float

    def __init__(self, run: IStanRun, output: str, warnings: str, errors: str, runtime: float, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(run, IStanRun)
        assert isinstance(output, str)
        assert isinstance(warnings, str)
        assert isinstance(errors, str)
        assert isinstance(runtime, float)
        self._run = run
        self._output = output
        self._warnings = warnings
        self._errors = errors
        self._runtime = runtime

    @property
    @overrides
    def is_error(self) -> bool:
        return self._errors != ""

    @property
    @overrides
    def runtime(self) -> timedelta | None:
        return timedelta(seconds=self._runtime)

    @property
    @overrides
    def data_hash(self) -> int:
        return self._run.get_data_meta().object_hash

    @property
    @overrides
    def model_hash(self) -> int:
        return self._run.get_model_meta().object_hash

    @property
    @overrides
    def run_hash(self) -> int:
        return self._run.object_hash

    @overrides
    def get_data_meta(self) -> IStanDataMeta:
        return self._run.get_data_meta()

    @overrides
    def get_model_meta(self) -> IStanModelMeta:
        return self._run.get_model_meta()

    @overrides
    def get_run_meta(self) -> IStanRunMeta:
        return self._run

    @property
    @overrides
    def result_type(self) -> StanResultEngine:
        return self._run.run_engine

    @property
    @overrides
    def output_scope(self) -> StanOutputScope:
        return self._run.output_scope

    @overrides
    def get_object(self) -> ISerializableObject:
        return self

    @overrides
    def get_metaobject(self) -> IStanResultMeta:
        return StanResultMeta(self.get_run_meta(), self._runtime, self._errors, "run")

    @abstractmethod
    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["runtime"] = self._runtime
        d["status"] = "error" if self.is_error else "success"
        d["run_hash"] = self.run_hash
        d["model_hash"] = self.model_hash
        d["data_hash"] = self.data_hash
        d["output_scope"] = self.output_scope.txt_value()
        return d

    @property
    @overrides
    def messages(self) -> str:
        return self._output

    @property
    @overrides
    def warnings(self) -> str:
        return self._warnings

    @property
    @overrides
    def errors(self) -> str:
        return self._errors

    @overrides
    def repr_with_sampling_errors(self):
        # Table example:
        #         mean   se_mean       sd       10%      90%
        # mu  7.751103 0.1113406 5.199004 1.3286256 14.03575
        # tau 6.806410 0.1785522 6.044944 0.9572097 14.48271
        out = self.method_name + "\n"

        out += self.formatted_runtime() + "\n"

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "mu", "sigma", "10%", "90%"]
        for user_par in self.user_parameters:
            dims = self.get_parameter_shape(user_par)
            if len(dims) == 0 or (len(dims) == 1 and dims[0] == 1):
                par_name = user_par
                par_value = self.get_onedim_parameter_estimate(par_name)
                ci = par_value.get_CI(0.8)
                table.add_row([par_name, "", str(par_value.estimateMean()), str(par_value.estimateSE()),
                               str(ci.pretty_lower), str(ci.pretty_upper)])
            else:
                max_idx = math.prod(dims)
                idx = [0 for _ in dims]
                i = 0
                par_txt = user_par
                while i < max_idx:
                    idx_txt = "[" + ",".join([str(i + 1) for i in idx]) + "]"
                    par_name = f"{user_par}{idx_txt}"
                    par_value = self.get_onedim_parameter_estimate(par_name)
                    ci = par_value.get_CI(0.8)
                    table.add_row([par_txt, idx_txt, str(par_value.estimateMean()), str(par_value.estimateSE()),
                                   str(ci.pretty_lower), str(ci.pretty_upper)])
                    par_txt = ""
                    i += 1
                    idx[-1] += 1
                    for j in range(len(dims) - 1, 0, -1):
                        if idx[j] >= dims[j]:
                            idx[j] = 0
                            idx[j - 1] += 1

        return out + str(table)

    @overrides
    def get_parameter_estimates(self, user_parameter_name: str, store_values: bool = False) -> Any:
        values = np.asarray(
            [self.get_onedim_parameter_estimate(name) for name in self.get_onedim_parameter_names(user_parameter_name)])
        values.shape = self.get_parameter_shape(user_parameter_name)
        return values.tolist()

    @overrides
    def repr_without_sampling_errors(self):
        # Table example:
        #        value        10%      90%
        # mu  7.751103  1.3286256 14.03575
        # tau 6.806410  0.9572097 14.48271
        out = self.method_name + "\n"

        out += self.formatted_runtime() + "\n"

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "value", "10%", "90%"]
        for par in self.user_parameters:
            dims = self.get_parameter_shape(par)
            if len(dims) == 0:
                par_name = par
                par_value = self.get_onedim_parameter_estimate(par_name)
                ci = par_value.get_CI(0.8)
                table.add_row([par_name, "", str(par_value),
                               str(ci.pretty_lower), str(ci.pretty_upper)])
            else:
                max_idx = math.prod(dims)
                idx = [0 for _ in dims]
                i = 0
                par_txt = par
                while i < max_idx:
                    idx_txt = "[" + ",".join([str(i + 1) for i in idx]) + "]"
                    par_name = f"{par}{idx_txt}"
                    par_value = self.get_onedim_parameter_estimate(par_name)
                    ci = par_value.get_CI(0.8)
                    table.add_row([par_txt, idx_txt, str(par_value),
                                   str(ci.pretty_lower), str(ci.pretty_upper)])
                    par_txt = ""
                    i += 1
                    idx[-1] += 1
                    for j in range(len(dims) - 1, 0, -1):
                        if idx[j] >= dims[j]:
                            idx[j] = 0
                            idx[j - 1] += 1

        return out + str(table)

    def __repr__(self):
        if self.sample_count is None:
            return self.repr_without_sampling_errors()
        else:
            return self.repr_with_sampling_errors()

    @property
    @overrides
    def method_name(self) -> str:
        if self.result_type == StanResultEngine.NONE:
            return "No result available"
        elif self.result_type == StanResultEngine.LAPLACE:
            return "Laplace approximation around the posterior mode"
        elif self.result_type == StanResultEngine.PATHFINDER:
            return "Pathfinder variational inference algorithm."
        elif self.result_type == StanResultEngine.VB:
            return f"Variational inference algorithm {self.requested_algorithm_variation}"
        elif self.result_type == StanResultEngine.MCMC:
            return f"MCMC algorithm {self.requested_algorithm_variation}, engine {self._run.run_engine.txt_value()}"

    @overrides
    def formatted_runtime(self) -> str:
        if self._runtime is None:
            return "Run time: not available"
        else:
            return f"Run taken: {humanize.precisedelta(datetime.timedelta(seconds=self._runtime))}"

    @property
    @overrides
    def is_object_available(self) -> bool:
        return True

    @property
    @overrides
    def object_hash(self) -> int:
        return self.run_hash

    @overrides
    def pretty_print(self) -> str:
        return self.__repr__()


class ImplCovarianceInterface(IStanResultCovariances):
    @overrides
    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        if isinstance(user_parameter_names, str):
            user_parameter_names = [user_parameter_names]
            one_dim_names = []
            for name in user_parameter_names:
                one_dim_names.extend(self.get_onedim_parameter_names(name))
        else:
            one_dim_names = self.onedim_parameters

        cov_matrix = np.asarray(
            [self.get_cov_onedim_par(name1, name2) for name1 in one_dim_names for name2 in one_dim_names])

        return cov_matrix, one_dim_names

    @overrides
    def pretty_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> str:
        cov_matrix, one_dim_names = self.get_cov_matrix(user_parameter_names)
        out = prettytable.PrettyTable()
        out.field_names = [""] + one_dim_names

        # Calculate the smallest standard error of variances
        factor = np.sqrt(0.5 / (self.sample_count() - 1))
        se = np.sqrt(min(np.diag(cov_matrix))) * factor
        digits = int(np.ceil(-np.log10(se))) + 1

        cov_matrix = np.round(cov_matrix, digits)
        cov_matrix_txt = np.ndarray((len(one_dim_names), len(one_dim_names)), dtype=str).tolist()
        # Remove upper triangle
        for i in range(len(one_dim_names)):
            for j in range(0, i + 1):
                cov_matrix_txt[i][j] = f"{cov_matrix[i, j]:.4f}"

        for i in range(len(one_dim_names)):
            # Suppres scientific notation
            out.add_row([one_dim_names[i]] + [f"{cov_matrix_txt[i][j]}" for j in range(len(one_dim_names))])
        return str(out)

    @overrides
    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        if user_parameter_names is None:
            user_parameter_names = self.user_parameters
        elif isinstance(user_parameter_names, str):
            user_parameter_names = [user_parameter_names]

        onedim_parameter_names = [self.get_onedim_parameter_names(key) for key in user_parameter_names]
        onedim_parameter_names = [name for sublist in onedim_parameter_names for name in sublist]

        cov_matrix = np.zeros((len(onedim_parameter_names), len(onedim_parameter_names)))

        for i, name1 in enumerate(onedim_parameter_names):
            for j, name2 in enumerate(onedim_parameter_names):
                cov_matrix[i, j] = self.get_cov_onedim_par(name1, name2)

        return cov_matrix, onedim_parameter_names


class ImplValueWithError(IStanResultBase):
    @overrides
    def get_parameter_mu(self, user_parameter_name: str) -> np.ndarray:
        one_dim_pars = self.get_onedim_parameter_names(user_parameter_name)
        return np.array([self.get_onedim_parameter_estimate(p).value for p in one_dim_pars])

    @overrides
    def get_parameter_sigma(self, user_parameter_name: str) -> np.ndarray:
        one_dim_pars = self.get_onedim_parameter_names(user_parameter_name)
        return np.array([self.get_onedim_parameter_estimate(p).SE for p in one_dim_pars])

    @overrides
    def all_main_effects_onedim_par(self) -> dict[str, IValueWithError]:
        return {k: self.get_onedim_parameter_estimate(k) for k in self.onedim_parameters}

    @overrides
    def sample_count(self, onedim_parameter_name: str = None) -> float | int | None:
        if onedim_parameter_name is None:
            obj = self.get_run_meta()
            assert isinstance(obj, IStanRunMeta)
            return obj.sample_count
        else:
            return self.get_onedim_parameter_estimate(onedim_parameter_name).N
