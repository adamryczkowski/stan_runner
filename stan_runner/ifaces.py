from abc import ABC, abstractmethod
import prettytable
import math
from ValueWithError import ValueWithError
import numpy as np

class IStanResult(ABC):
    @abstractmethod
    @property
    def user_parameters(self) -> list[str]:
        ...

    @abstractmethod
    @property
    def onedim_parameters(self) -> list[str]:
        ...

    @abstractmethod
    @property
    def user_parameter_count(self) -> int:
        ...

    @abstractmethod
    @property
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

    def __repr__(self):
        # Table example:
        #         mean   se_mean       sd       10%      90%
        # mu  7.751103 0.1113406 5.199004 1.3286256 14.03575
        # tau 6.806410 0.1785522 6.044944 0.9572097 14.48271

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "mu", "sigma", "10%", "90%"]
        for par in self.user_parameters:
            dims = self.get_parameter_shape(par)
            if len(dims) == 1 and dims[0] == 1:
                par_name = par
                par_value = self.get_parameter_estimate(par_name)
                ci = par_value.get_CI(0.8)
                table.add_row([par_name, "", str(par_value.estimateMean), str(par_value.estimateSE),
                               str(ci.pretty_lower), str(ci.pretty_upper)])
            else:
                max_idx = math.prod(dims)
                idx = [0 for _ in dims]
                i = 0
                while i < max_idx:
                    idx_txt = "[" + "][".join([str(i + 1) for i in idx]) + "]"
                    par_name = f"{par}{idx_txt}"
                    par_value = self.get_parameter_estimate(par_name)
                    ci = par_value.get_CI(0.8)
                    table.add_row([par, idx_txt, str(par_value.estimateMean), str(par_value.estimateSE),
                                   str(ci.pretty_lower), str(ci.pretty_upper)])
                    par = ""
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
