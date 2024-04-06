from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any
from cmdstanpy import CmdStanMCMC

import numpy as np
from ValueWithError import ValueWithError


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



class IInferenceResult(ABC):
    @property
    @abstractmethod
    def is_error(self) -> bool:
        ...

    @abstractmethod
    def get_onedim_parameter_names(self, user_parameter_name: str) -> list[str]:
        ...

    @abstractmethod
    def get_onedim_parameter_index(self, onedim_parameter_name: str) -> int:
        ...

    @property
    @abstractmethod
    def result_type(self) -> StanResultEngine:
        ...

    @property
    @abstractmethod
    def result_scope(self) -> StanOutputScope:
        ...

    @abstractmethod
    def serialize_to_file(self, output_type: str, file_name: str):
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

    @property
    @abstractmethod
    def draws(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> ValueWithError:
        ...

    @abstractmethod
    def get_parameter_mu(self, user_parameter_name: str) -> np.ndarray:
        ...

    @abstractmethod
    def get_parameter_sigma(self, user_parameter_name: str) -> np.ndarray:
        ...

    @abstractmethod
    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        ...

    @abstractmethod
    def pretty_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> str:
        ...

    @abstractmethod
    def repr_with_sampling_errors(self):
        ...

    @abstractmethod
    def method_name(self) -> str:
        ...

    @abstractmethod
    def repr_without_sampling_errors(self):
        ...

    @abstractmethod
    def formatted_runtime(self) -> str:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def all_main_effects(self) -> dict[str, ValueWithError]:
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
    def load_model_by_file(self, stan_file: str | Path, model_name: str | None = None, pars_list: list[str] = None):
        ...

    @abstractmethod
    def load_model_by_str(self, model_code: str | list[str], model_name: str, pars_list: list[str] = None):
        ...

    @abstractmethod
    def load_data_by_file(self, data_file: str | Path):
        ...

    @abstractmethod
    def load_data_by_dict(self, data: dict[str, float | int | np.ndarray]):
        ...

    @abstractmethod
    def sampling(self, num_chains: int, iter_sampling: int = None,
                 iter_warmup: int = None, thin: int = 1, max_treedepth: int = None,
                 seed: int = None, inits: dict[str, Any] | float | list[str] = None) -> IInferenceResult:
        ...

    @abstractmethod
    def variational_bayes(self, output_samples: int = 1000, **kwargs) -> IInferenceResult:
        ...

    @abstractmethod
    def pathfinder(self, output_samples: int = 1000, **kwargs) -> IInferenceResult:
        ...

    @abstractmethod
    def laplace_sample(self, output_samples: int = 1000, **kwargs) -> IInferenceResult:
        ...

    @property
    @abstractmethod
    def model_code(self) -> str | None:
        ...


def test():
    mcmc = CmdStanMCMC()