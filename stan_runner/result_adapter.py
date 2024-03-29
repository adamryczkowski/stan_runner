import math
from enum import Enum

import humanize
import numpy as np
import prettytable
from ValueWithError import ValueWithError, ValueWithErrorVec
from cmdstanpy import CmdStanLaplace, CmdStanVB, CmdStanMCMC, CmdStanPathfinder
from datetime import timedelta


class StanResultType(Enum):
    NONE = 0
    LAPLACE = 1
    VB = 2
    MCMC = 3
    PATHFINDER = 4


class InferenceResult:
    _result: CmdStanLaplace | CmdStanVB | CmdStanMCMC | CmdStanPathfinder | None
    _messages: dict[str, str] | None = None
    _draws: np.ndarray | None
    _user2onedim: dict[str, list[str]] | None  # Translates user parameter names to one-dim parameter names
    _runtime: timedelta | None  # Time taken to run the inference

    def __init__(self, result: CmdStanLaplace | CmdStanVB | CmdStanMCMC | CmdStanPathfinder | None,
                 messages: dict[str, str], runtime: timedelta = None) -> None:
        if result is not None:
            assert isinstance(result, (CmdStanLaplace, CmdStanVB, CmdStanMCMC, CmdStanPathfinder))
            assert isinstance(messages, dict)

        self._result = result
        self._draws = None
        self._messages = messages
        self._user2onedim = None
        if runtime is not None:
            assert isinstance(runtime, timedelta)
            self._runtime = runtime

    def make_dict(self):
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
        return self._result is None

    def get_onedim_parameter_names(self, user_parameter_name: str) -> list[str]:
        self.make_dict()
        return self._user2onedim[user_parameter_name]

    def get_onedim_parameter_index(self, onedim_parameter_name: str) -> int:
        return self._result.column_names.index(onedim_parameter_name)

    @property
    def result_type(self) -> StanResultType:
        if self._result is None:
            return StanResultType.NONE
        elif isinstance(self._result, CmdStanLaplace):
            return StanResultType.LAPLACE
        elif isinstance(self._result, CmdStanVB):
            return StanResultType.VB
        elif isinstance(self._result, CmdStanMCMC):
            return StanResultType.MCMC
        elif isinstance(self._result, CmdStanPathfinder):
            return StanResultType.PATHFINDER
        else:
            raise ValueError("Unknown result type")

    @property
    def messages(self) -> dict[str, str] | None:
        return self._messages

    @property
    def user_parameters(self) -> list[str]:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultType.LAPLACE:
            return list(self._result.metadata.stan_vars.keys())
        elif self.result_type == StanResultType.VB:
            return list(self._result.metadata.stan_vars.keys())
        elif self.result_type == StanResultType.MCMC:
            return list(self._result.metadata.stan_vars.keys())
        elif self.result_type == StanResultType.PATHFINDER:
            return list(self._result.metadata.stan_vars.keys())
        else:
            raise ValueError("Unknown result type")

    @property
    def onedim_parameters(self) -> list[str]:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultType.LAPLACE:
            return list(self._result.column_names[2:])
        elif self.result_type == StanResultType.VB:
            return list(self._result.column_names[3:])
        elif self.result_type == StanResultType.MCMC:
            return list(self._result.column_names[7:])
        elif self.result_type == StanResultType.PATHFINDER:
            return list(self._result.column_names[2:])
        else:
            raise ValueError("Unknown result type")

    def get_parameter_shape(self, user_parameter_name: str) -> tuple[int, ...]:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultType.LAPLACE:
            return self._result.metadata.stan_vars[user_parameter_name].dimensions
        elif self.result_type == StanResultType.VB:
            return self._result.metadata.stan_vars[user_parameter_name].dimensions
        elif self.result_type == StanResultType.MCMC:
            return self._result.metadata.stan_vars[user_parameter_name].dimensions
        elif self.result_type == StanResultType.PATHFINDER:
            return self._result.metadata.stan_vars[user_parameter_name].dimensions
        else:
            raise ValueError("Unknown result type")

    def sample_count(self, onedim_parameter_name: str = None) -> float | int | None:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultType.LAPLACE:
            return self.draws.shape[0]
        elif self.result_type == StanResultType.VB:
            return self.draws.shape[0]
        elif self.result_type == StanResultType.MCMC:
            if onedim_parameter_name is not None:
                s = self._result.summary()
                return s["N_Eff"][onedim_parameter_name]
            else:
                return self.draws.shape[0]
        elif self.result_type == StanResultType.PATHFINDER:
            return self.draws.shape[0]
        else:
            raise ValueError("Unknown result type")

    @property
    def draws(self) -> np.ndarray:
        if self._draws is None:
            if self._result is None:
                raise ValueError("No result available")
            elif self.result_type == StanResultType.LAPLACE:
                self._draws = self._result.draws()
            elif self.result_type == StanResultType.VB:
                self._draws = self._result.variational_sample
            elif self.result_type == StanResultType.MCMC:
                self._draws = self._result.draws(concat_chains=True)
            elif self.result_type == StanResultType.PATHFINDER:
                self._draws = self._result.draws()
        return self._draws

    def get_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> ValueWithError:
        if self._result is None:
            raise ValueError("No result available")
        var_index = self._result.column_names.index(onedim_parameter_name)
        if store_values:
            ans = ValueWithErrorVec(self.draws[:, var_index])
        else:
            ans = ValueWithError.CreateFromVector(self.draws[:, var_index], N=self.sample_count(onedim_parameter_name))
        return ans

    def get_parameter_mu(self, user_parameter_name: str) -> np.ndarray:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultType.LAPLACE:
            return self._result.stan_variable(user_parameter_name)
        elif self.result_type == StanResultType.VB:
            return np.mean(self._result.stan_variable(user_parameter_name), axis=0)
        elif self.result_type == StanResultType.MCMC:
            return np.mean(self._result.stan_variable(user_parameter_name), axis=0)
        elif self.result_type == StanResultType.PATHFINDER:
            return self._result.stan_variable(user_parameter_name)
        else:
            raise ValueError("Unknown result type")

    def get_parameter_sigma(self, user_parameter_name: str) -> np.ndarray:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type != StanResultType.NONE:
            return np.std(self._result.stan_variable(user_parameter_name), axis=0)
        else:
            raise ValueError("Unknown result type")

    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type != StanResultType.NONE:
            if isinstance(user_parameter_names, str):
                user_parameter_names = [user_parameter_names]
                one_dim_names = []
                for name in user_parameter_names:
                    one_dim_names.extend(self.get_onedim_parameter_names(name))
            else:
                one_dim_names = self.onedim_parameters

            indices = [self.get_onedim_parameter_index(name) for name in one_dim_names]

            return np.cov(self.draws[:, indices], rowvar=False), one_dim_names
        else:
            raise ValueError("Unknown result type")

    def pretty_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> str:
        cov_matrix, one_dim_names = self.get_cov_matrix(user_parameter_names)
        out = prettytable.PrettyTable()
        out.field_names = [""] + one_dim_names

        # Calculate the smallest standard error of variances
        factor = np.sqrt(0.5 / (self.sample_count() - 1))
        se = np.sqrt(min(np.diag(cov_matrix)))* factor
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
        out = self.method_name() + "\n"

        out += self.formatted_runtime() + "\n"

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

        return out + str(table)

    def method_name(self) -> str:
        if self.result_type == StanResultType.NONE:
            return "No result available"
        elif self.result_type == StanResultType.LAPLACE:
            return "Laplace approximation around the posterior mode"
        elif self.result_type == StanResultType.PATHFINDER:
            return "Pathfinder variational inference algorithm."
        elif self.result_type == StanResultType.VB:
            return f"Variational inference algorithm {self._result.metadata.cmdstan_config['algorithm']}"
        elif self.result_type == StanResultType.MCMC:
            return f"MCMC algorithm {self._result.metadata.cmdstan_config['algorithm']}, engine {self._result.metadata.cmdstan_config['engine']}"

    def repr_without_sampling_errors(self):
        # Table example:
        #        value        10%      90%
        # mu  7.751103  1.3286256 14.03575
        # tau 6.806410  0.9572097 14.48271
        out = self.method_name() + "\n"

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
