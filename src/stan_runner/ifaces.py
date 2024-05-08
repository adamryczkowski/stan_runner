from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import humanize
import numpy as np
import prettytable
from ValueWithError import IValueWithError


class StanErrorType(Enum):
    NO_ERROR = 0
    SYNTAX_ERROR = 1
    COMPILE_ERROR = 2


class StanResultEngine(Enum):
    NONE = 0
    LAPLACE = 1
    VB = 2
    MCMC = 3
    PATHFINDER = 4

    @staticmethod
    def FromStr(value: str) -> StanResultEngine:
        if value == "laplace":
            return StanResultEngine.LAPLACE
        elif value == "vb":
            return StanResultEngine.VB
        elif value == "mcmc":
            return StanResultEngine.MCMC
        elif value == "pathfinder":
            return StanResultEngine.PATHFINDER
        else:
            raise ValueError(f"Unknown StanResultEngine: {value}")

    def txt_value(self):
        if self == StanResultEngine.LAPLACE:
            return "laplace"
        elif self == StanResultEngine.VB:
            return "vb"
        elif self == StanResultEngine.MCMC:
            return "mcmc"
        elif self == StanResultEngine.PATHFINDER:
            return "pathfinder"
        else:
            raise ValueError(f"Unknown StanResultEngine: {self}")


class StanOutputScope(Enum):
    MainEffects = 1
    Covariances = 2
    FullSamples = 3
    RawOutput = 4

    def __str__(self):
        if self == StanOutputScope.MainEffects:
            return "main_effects"
        elif self == StanOutputScope.Covariances:
            return "covariances"
        elif self == StanOutputScope.FullSamples:
            return "draws"
        elif self == StanOutputScope.RawOutput:
            return "raw"
        else:
            raise ValueError(f"Unknown StanOutputScope: {self}")

    @staticmethod
    def FromStr(value: str) -> StanOutputScope:
        if value == "main_effects":
            return StanOutputScope.MainEffects
        elif value == "covariances":
            return StanOutputScope.Covariances
        elif value == "draws":
            return StanOutputScope.FullSamples
        elif value == "raw":
            return StanOutputScope.RawOutput
        else:
            raise ValueError(f"Unknown StanOutputScope: {value}")

    def txt_value(self):
        if self == StanOutputScope.MainEffects:
            return "main_effects"
        elif self == StanOutputScope.Covariances:
            return "covariances"
        elif self == StanOutputScope.FullSamples:
            return "draws"
        elif self == StanOutputScope.RawOutput:
            return "raw"
        else:
            raise ValueError(f"Unknown StanOutputScope: {self}")

    def __gt__(self, other: StanOutputScope):
        return self.value > other.value


class IInferenceResult(ABC):


    @property
    @abstractmethod
    async def one_dim_parameters_count(self) -> int:
        ...

    async def get_onedim_parameter_names(self, user_parameter_name: str) -> list[str]:
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
    async def sample_count(self, onedim_parameter_name: str = None) -> float | int | None:
        ...

    @abstractmethod
    async def draws(self, incl_raw: bool = True) -> np.ndarray|None:
        ...

    @abstractmethod
    async def get_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> IValueWithError:
        ...

    @abstractmethod
    async def get_parameter_mu(self, user_parameter_name: str) -> np.ndarray:
        ...

    @abstractmethod
    async def get_parameter_sigma(self, user_parameter_name: str) -> np.ndarray:
        ...

    @abstractmethod
    async def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        ...

    @abstractmethod
    async def all_main_effects(self) -> dict[str, IValueWithError]:
        ...

    async def pretty_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> str:
        cov_matrix, one_dim_names = await self.get_cov_matrix(user_parameter_names)
        out = prettytable.PrettyTable()
        out.field_names = [""] + one_dim_names

        # Calculate the smallest standard error of variances
        factor = np.sqrt(0.5 / (self.sample_count() - 1))
        se = np.sqrt(min(np.diag(cov_matrix))) * factor
        digits = int(np.ceil(-np.log10(se))) + 1

        cov_matrix_txt = np.round(cov_matrix, digits)

        for i in range(len(one_dim_names)):
            # Suppres scientific notation
            out.add_row([one_dim_names[i]] + [f"{cov_matrix_txt[i, j]:.4f}" for j in range(len(one_dim_names))])
        return str(out)

    async def repr_with_sampling_errors(self):
        # Table example:
        #         mean   se_mean       sd       10%      90%
        # mu  7.751103 0.1113406 5.199004 1.3286256 14.03575
        # tau 6.806410 0.1785522 6.044944 0.9572097 14.48271
        out = self.method_name + "\n"

        out += self.formatted_runtime() + "\n"

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "mu", "sigma", "10%", "90%"]
        for par in self.user_parameters:
            dims = self.get_parameter_shape(par)
            if len(dims) == 0 or (len(dims) == 1 and dims[0] == 1):
                par_name = par
                par_value = await self.get_parameter_estimate(par_name)
                ci = par_value.get_CI(0.8)
                table.add_row([par_name, "", str(par_value.estimateMean()), str(par_value.estimateSE()),
                               str(ci.pretty_lower), str(ci.pretty_upper)])
            else:
                max_idx = math.prod(dims)
                idx = [0 for _ in dims]
                i = 0
                par_txt = par
                while i < max_idx:
                    idx_txt = "[" + ",".join([str(i + 1) for i in idx]) + "]"
                    par_name = f"{par}{idx_txt}"
                    par_value = await self.get_parameter_estimate(par_name)
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


    async def repr_without_sampling_errors(self):
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
                par_value = await self.get_parameter_estimate(par_name)
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
                    par_value = await self.get_parameter_estimate(par_name)
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

    def formatted_runtime(self) -> str:
        if self.runtime is None:
            return "Run time: not available"
        else:
            return f"Run taken: {humanize.precisedelta(self.runtime)}"


    @abstractmethod
    async def get_progress(self) -> tuple[str, list[float]]:
        """Gets progress: each item in the list is a progress of a particular chain.
        First tuple element is the description of the phase.
        """
        ...


    def __repr__(self):
        if self.sample_count is None:
            return self.repr_without_sampling_errors()
        else:
            return self.repr_with_sampling_errors()


class ILocalInferenceResult(IInferenceResult):
    _runtime: timedelta | None
    _messages: dict[str, str] | None = None
    _user2onedim: dict[str, list[str]] | None  # Translates user parameter names to one-dim parameter names

    def __init__(self, messages: dict[str, str], runtime: timedelta | None):
        self._runtime = runtime
        self._messages = messages
        self._user2onedim = None

    # This member function is to be treated as protected, only accessible by method of derived classes, and not by the user
    def _make_dict(self):
        if self._user2onedim is not None:
            return
        self._user2onedim = {}
        for user_par in self.user_parameters:
            dims = self.get_parameter_shape(user_par)
            if len(dims) == 0:
                self._user2onedim[user_par] = [user_par]
            else:
                max_idx = math.prod(dims)
                idx = [0 for _ in dims]
                i = 0
                self._user2onedim[user_par] = []
                while i < max_idx:
                    idx_txt = "[" + ",".join([str(i + 1) for i in idx]) + "]"
                    par_name = f"{user_par}{idx_txt}"
                    self._user2onedim[user_par].append(par_name)
                    i += 1
                    idx[-1] += 1
                    for j in range(len(dims) - 1, 0, -1):
                        if idx[j] >= dims[j]:
                            idx[j] = 0
                            idx[j - 1] += 1

    @property
    def is_error(self) -> bool:
        return "error" in self._messages

    @property
    def one_dim_parameters_count(self) -> int:
        self._make_dict()
        return len(self.onedim_parameters)

    def get_onedim_parameter_names(self, user_parameter_name: str) -> list[str]:
        self._make_dict()
        return self._user2onedim[user_parameter_name]

    @property
    @abstractmethod
    def result_type(self) -> StanResultEngine:
        ...

    @property
    @abstractmethod
    def result_scope(self) -> StanOutputScope:
        ...

    @abstractmethod
    def serialize(self, output_scope: StanOutputScope)->bytes:
        ...

    @property
    def user_parameters(self) -> list[str]:
        self._make_dict()
        return list(self._user2onedim.keys())

    @property
    def onedim_parameters(self) -> list[str]:
        self._make_dict()
        return [item for sublist in self._user2onedim.values() for item in sublist]

    @abstractmethod
    def get_parameter_shape(self, user_parameter_name: str) -> tuple[int, ...]:
        ...

    @abstractmethod
    def sample_count(self, onedim_parameter_name: str = None) -> float | int | None:
        ...

    @abstractmethod
    def draws(self, incl_raw: bool = True) -> np.ndarray:
        ...

    @abstractmethod
    def get_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> IValueWithError:
        ...

    @abstractmethod
    def get_parameter_mu(self, user_parameter_name: str) -> np.ndarray:
        ...

    @abstractmethod
    def get_parameter_sigma(self, user_parameter_name: str) -> np.ndarray:
        ...

    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        if user_parameter_names is None:
            user_parameter_names = self.onedim_parameters
        elif isinstance(user_parameter_names, str):
            user_parameter_names = [user_parameter_names]

        onedim_parameter_names = [self.get_onedim_parameter_names(key) for key in user_parameter_names]
        onedim_parameter_names = [name for sublist in onedim_parameter_names for name in sublist]

        cov_matrix = np.zeros((len(onedim_parameter_names), len(onedim_parameter_names)))

        for i, name1 in enumerate(onedim_parameter_names):
            for j, name2 in enumerate(onedim_parameter_names):
                cov_matrix[i, j] = self.get_cov_onedim_par(name1, name2)

        return cov_matrix, onedim_parameter_names

    def pretty_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> str:
        cov_matrix, one_dim_names = self.get_cov_matrix(user_parameter_names)
        out = prettytable.PrettyTable()
        out.field_names = [""] + one_dim_names

        # Calculate the smallest standard error of variances
        factor = np.sqrt(0.5 / (self.sample_count() - 1))
        se = np.sqrt(min(np.diag(cov_matrix))) * factor
        digits = int(np.ceil(-np.log10(se))) + 1

        cov_matrix_txt = np.round(cov_matrix, digits)

        for i in range(len(one_dim_names)):
            # Suppres scientific notation
            out.add_row([one_dim_names[i]] + [f"{cov_matrix_txt[i, j]:.4f}" for j in range(len(one_dim_names))])
        return str(out)

    def repr_with_sampling_errors(self):
        # Table example:
        #         mean   se_mean       sd       10%      90%
        # mu  7.751103 0.1113406 5.199004 1.3286256 14.03575
        # tau 6.806410 0.1785522 6.044944 0.9572097 14.48271
        out = self.method_name + "\n"

        out += self.formatted_runtime() + "\n"

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "mu", "sigma", "10%", "90%"]
        for par in self.user_parameters:
            dims = self.get_parameter_shape(par)
            if len(dims) == 0 or (len(dims) == 1 and dims[0] == 1):
                par_name = par
                par_value = self.get_parameter_estimate(par_name)
                ci = par_value.get_CI(0.8)
                table.add_row([par_name, "", str(par_value.estimateMean()), str(par_value.estimateSE()),
                               str(ci.pretty_lower), str(ci.pretty_upper)])
            else:
                max_idx = math.prod(dims)
                idx = [0 for _ in dims]
                i = 0
                par_txt = par
                while i < max_idx:
                    idx_txt = "[" + ",".join([str(i + 1) for i in idx]) + "]"
                    par_name = f"{par}{idx_txt}"
                    par_value = self.get_parameter_estimate(par_name)
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

    @property
    @abstractmethod
    def method_name(self) -> str:
        ...

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
                par_value = self.get_parameter_estimate(par_name)
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
                    par_value = self.get_parameter_estimate(par_name)
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

    def formatted_runtime(self) -> str:
        if self._runtime is None:
            return "Run time: not available"
        else:
            return f"Run taken: {humanize.precisedelta(self._runtime)}"

    def __repr__(self):
        if self.sample_count is None:
            return self.repr_without_sampling_errors()
        else:
            return self.repr_with_sampling_errors()

    @abstractmethod
    def all_main_effects(self) -> dict[str, IValueWithError]:
        ...

    @abstractmethod
    def get_cov_onedim_par(self, one_dim_par1: str, one_dim_par2: str) -> float | np.ndarray:
        ...



class IStanRunner(ABC):

    @property
    @abstractmethod
    def error_state(self) -> StanErrorType:
        ...

    @property
    @abstractmethod
    def is_model_loaded(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_model_compiled(self) -> bool:
        ...

    @abstractmethod
    def get_messages(self, error_only: bool) -> str:
        ...

    @abstractmethod
    async def load_model_by_file(self, stan_file: str | Path, model_name: str | None = None, pars_list: list[str] = None):
        ...

    @abstractmethod
    async def load_model_by_str(self, model_code: str | list[str], model_name: str, pars_list: list[str] = None):
        ...

    @abstractmethod
    async def load_data_by_file(self, data_file: str | Path):
        ...

    @abstractmethod
    async def load_data_by_dict(self, data: dict[str, float | int | np.ndarray]):
        ...

    @abstractmethod
    async def sampling(self, num_chains: int, iter_sampling: int = None,
                 iter_warmup: int = None, thin: int = 1, max_treedepth: int = None,
                 seed: int = None, inits: dict[str, Any] | float | list[str] = None) -> ILocalInferenceResult:
        ...

    @abstractmethod
    async def variational_bayes(self, output_samples: int = 1000, **kwargs) -> ILocalInferenceResult:
        ...

    @abstractmethod
    async def pathfinder(self, output_samples: int = 1000, **kwargs) -> ILocalInferenceResult:
        ...

    @abstractmethod
    async def laplace_sample(self, output_samples: int = 1000, **kwargs) -> ILocalInferenceResult:
        ...

    @property
    @abstractmethod
    def model_code(self) -> str | None:
        ...

