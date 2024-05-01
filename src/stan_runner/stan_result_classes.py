import math

import numpy as np
import prettytable
from ValueWithError import ValueWithError, ValueWithErrorVec
from overrides import overrides

from .ifaces import IStanResult, IStanResultCov


class StanResultMainEffects(IStanResult):
    _one_dim_pars: dict[str, ValueWithError]
    _par_dimensions: dict[str, list[int]]

    def __init__(self, pars: dict[str, ValueWithError], par_dimensions: dict[str, list[int]]):
        assert isinstance(pars, dict)
        assert isinstance(par_dimensions, dict)
        self._one_dim_pars = pars
        self._par_dimensions = par_dimensions

    @overrides
    def __repr__(self):
        # Table example:
        #         mean   se_mean       sd       10%      90%
        # mu  7.751103 0.1113406 5.199004 1.3286256 14.03575
        # tau 6.806410 0.1785522 6.044944 0.9572097 14.48271

        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "index", "mu", "sigma", "10%", "90%"]
        for par, dims in self._par_dimensions.items():
            if len(dims) == 1 and dims[0] == 1:
                par_name = par
                par_value = self._one_dim_pars[par_name]
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
                    par_value = self._one_dim_pars[par_name]
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

    @overrides
    def user_parameter_count(self) -> int:
        return len(self._par_dimensions)

    @overrides
    def get_parameter_shape(self, user_par_name: str) -> list[int]:
        return self._par_dimensions[user_par_name]

    @overrides
    def get_parameter_estimate(self, onedim_par_name: str) -> ValueWithError:
        if len(self._par_dimensions[onedim_par_name]) == 1:
            return self._one_dim_pars[onedim_par_name]
        else:
            if isinstance(idx, int):
                idx = [idx]
            if len(idx) != len(self._par_dimensions[onedim_par_name]):
                raise ValueError(
                    f"Index {idx} has wrong length for parameter {onedim_par_name} with shape {self._par_dimensions[onedim_par_name]}")

            one_par_name = onedim_par_name + "[" + "][".join([str(i) for i in idx]) + "]"
            return self._one_dim_pars[one_par_name]

    @overrides
    def get_parameter_mu(self, user_par_name: str) -> np.ndarray:
        if len(self._par_dimensions[user_par_name]) == 1:
            return np.array([self._one_dim_pars[user_par_name].value])
        else:
            return np.array([self._one_dim_pars[user_par_name + "[" + "][".join([str(i) for i in idx]) + "]"].value
                             for idx in np.ndindex(*self._par_dimensions[user_par_name])])

    @overrides
    def get_parameter_sigma(self, user_par_name: str) -> np.ndarray:
        if len(self._par_dimensions[user_par_name]) == 1:
            return np.array([self._one_dim_pars[user_par_name].SE])
        else:
            return np.array([self._one_dim_pars[user_par_name + "[" + "][".join([str(i) for i in idx]) + "]"].SE
                             for idx in np.ndindex(*self._par_dimensions[user_par_name])])

    def mus(self) -> tuple[np.ndarray, dict[str, int]]:
        """Returns the means of the parameters"""
        par_count = len(self._one_dim_pars)
        ans = np.ndarray(par_count)
        keys = {}
        for i, (one_par, value) in enumerate(self._one_dim_pars.items()):
            ans[i] = value.value
            keys[one_par] = i
        return ans, keys


class StanResultFull(StanResultMainEffects, IStanResultCov):
    _one_dim_pars: dict[str, ValueWithErrorVec]

    def __init__(self, pars: dict[str, ValueWithErrorVec], par_dimensions: dict[str, list[int]]):
        # noinspection PyTypeChecker
        super().__init__(pars, par_dimensions)

    @overrides
    def get_covariances(self) -> tuple[np.ndarray, dict[str, int]]:
        """Returns the covariance matrix of the parameters and a dictionary of the parameter names"""
        par_names = list(self._par_dimensions.keys())
        par_count = len(self._one_dim_pars)
        cov = np.ndarray((par_count, par_count))
        keys = {}
        for i in range(par_count):
            for j in range(i, par_count):
                cov[i, j] = np.cov(self._one_dim_pars[par_names[i]].vector, self._one_dim_pars[par_names[j]].vector)
                cov[j, i] = cov[i, j]
                keys[par_names[i]] = i
        return cov, keys


class StanResultMultiNormal(IStanResultCov):
    _main_effects = np.ndarray
    _covariances = np.ndarray
    _keys = dict[str, int]
    _par_dimensions: dict[str, list[int]]

    def __init__(self, main_effects: np.ndarray, covariances: np.ndarray, keys: dict[str, int],
                 par_dimensions: dict[str, list[int]]):
        assert isinstance(main_effects, np.ndarray)
        assert isinstance(covariances, np.ndarray)
        assert isinstance(keys, dict)
        assert isinstance(par_dimensions, dict)
        assert main_effects.shape[0] == covariances.shape[0]
        assert main_effects.shape[0] == covariances.shape[1]
        assert main_effects.shape[0] == len(keys)

        self._main_effects = main_effects
        self._covariances = covariances
        self._keys = keys
        self._par_dimensions = par_dimensions

    @overrides
    def __repr__(self):
        pass

    @overrides
    def user_parameter_count(self) -> int:
        return len(self._keys)

    @overrides
    def get_parameter_shape(self, user_par_name: str) -> list[int]:
        return self._par_dimensions[user_par_name]

    @overrides
    def get_parameter_estimate(self, onedim_par_name: str) -> ValueWithError:
        if onedim_par_name not in self._keys:
            raise ValueError(f"Parameter {onedim_par_name} not found in the result")
        if idx != 0:
            raise ValueError(f"Parameter {onedim_par_name} is not one-dimensional")
        idx = self._keys[onedim_par_name]
        return ValueWithError(self._main_effects[idx], np.sqrt(self._covariances[idx, idx]))

    @overrides
    def get_parameter_mu(self, user_par_name: str) -> np.ndarray:
        if self.get_parameter_shape(user_par_name) == [1]:
            return np.array([self._main_effects[self._keys[user_par_name]]])

        return np.array([self._main_effects[user_par_name + "[" + "][".join([str(i) for i in idx]) + "]"]
                         for idx in np.ndindex(*self._par_dimensions[user_par_name])])

    @overrides
    def get_parameter_sigma(self, user_par_name: str) -> np.ndarray:
        if self.get_parameter_shape(user_par_name) == [1]:
            return np.array([np.sqrt(self._covariances[self._keys[user_par_name], self._keys[user_par_name]])])

        return np.array([np.sqrt(self._covariances[user_par_name + "[" + "][".join([str(i) for i in idx]) + "]",
                                                   user_par_name + "[" + "][".join([str(i) for i in idx]) + "]"])
                         for idx in np.ndindex(*self._par_dimensions[user_par_name])])

    @overrides
    def get_covariances(self) -> tuple[np.ndarray, dict[str, int]]:
        return self._covariances, self._keys
