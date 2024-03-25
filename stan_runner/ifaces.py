from abc import ABC, abstractmethod
import prettytable
import math
from ValueWithError import ValueWithError
import numpy as np
from enum import Enum

class StanErrorType(Enum):
    NO_ERROR = 0
    SYNTAX_ERROR = 1
    COMPILE_ERROR = 2
    SAMPLING_ERROR = 3

class StanResultType(Enum):
    Nothing = 0
    Sampling = 1
    ADVI = 2
    Laplace = 3


class IStanResult(ABC):
    @property
    @abstractmethod
    def error_state(self) -> StanErrorType:
        ...

    @property
    @abstractmethod
    def messages(self) -> dict[str,str]:
        ...

    @property
    @abstractmethod
    def user_parameters(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def onedim_parameters(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def user_parameter_count(self) -> int:
        ...

    @property
    @abstractmethod
    def onedim_parameter_count(self) -> int:
        ...

    @abstractmethod
    def get_parameter_shape(self, user_par_name: str) -> list[int]:
        ...

    @abstractmethod
    def get_parameter_estimate(self, onedim_par_name: str) -> ValueWithError:
        ...

    @abstractmethod
    def get_parameter_mu(self, user_par_name: str) -> np.ndarray:
        """Returns expected value of the parameter. If the parameter is multidimensional,
        the dimension of the result is the same as that of .parameter_shape()"""
        ...

    @abstractmethod
    def get_parameter_sigma(self, user_par_name: str) -> np.ndarray:
        """Returns standard deviation of the parameter. If the parameter is multidimensional,
        the dimensionality of the result is the same as that of .parameter_shape() squared"""
        ...

    @property
    @abstractmethod
    def result_type(self) -> StanResultType:
        ...

    @property
    @abstractmethod
    def is_data_set(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_model_compiled(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_model_loaded(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_model_sampled(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_error(self)->bool:
        ...

    @abstractmethod
    def clear_last_results(self):
        """Clears the last results, so that the next results can be stored"""
        ...

    @abstractmethod
    def clear_last_data(self):
        """Clears the last data, so that the next data can be stored"""
        ...

    @abstractmethod
    def clear_last_model(self):
        """Clears the last model, so that the next model can be stored"""
        ...


    @abstractmethod
    def get_messages(self, error_only:bool)->str:
        pass

    def __repr__(self):
        # Table example:
        #         mean   se_mean       sd       10%      90%
        # mu  7.751103 0.1113406 5.199004 1.3286256 14.03575
        # tau 6.806410 0.1785522 6.044944 0.9572097 14.48271

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "mu", "sigma", "10%", "90%"]
        for par in self.user_parameters:
            dims = self.get_parameter_shape(par)
            if len(dims) == 0:
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

        return str(table)


class IStanResultCov(IStanResult):
    @abstractmethod
    def get_covariances(self) -> tuple[np.ndarray, dict[str, int]]:
        """Returns the covariance matrix of the parameters and a dictionary of the parameter names"""
        ...
