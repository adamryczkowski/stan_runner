from __future__ import annotations

import base64
import json
import math
import pickle
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import humanize
import jsonpickle
import numpy as np
import prettytable
from ValueWithError import ValueWithError, ValueWithErrorVec
from cmdstanpy import CmdStanLaplace, CmdStanVB, CmdStanMCMC, CmdStanPathfinder
from cmdstanpy.cmdstan_args import CmdStanArgs
from cmdstanpy.stanfit.vb import RunSet
from overrides import overrides

from .ifaces import IInferenceResult, StanResultEngine, StanOutputScope

_fallback = json._default_encoder.default
json._default_encoder.default = lambda obj: getattr(obj.__class__, "to_json", _fallback)(obj)


class InferenceResult(IInferenceResult):
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
    @overrides
    def is_error(self) -> bool:
        return self._result is None

    @overrides
    def get_onedim_parameter_names(self, user_parameter_name: str) -> list[str]:
        self._make_dict()
        return self._user2onedim[user_parameter_name]

    @overrides
    def get_onedim_parameter_index(self, onedim_parameter_name: str) -> int:
        return self._result.column_names.index(onedim_parameter_name)

    @property
    @overrides
    def result_type(self) -> StanResultEngine:
        if self._result is None:
            return StanResultEngine.NONE
        elif isinstance(self._result, CmdStanLaplace):
            return StanResultEngine.LAPLACE
        elif isinstance(self._result, CmdStanVB):
            return StanResultEngine.VB
        elif isinstance(self._result, CmdStanMCMC):
            return StanResultEngine.MCMC
        elif isinstance(self._result, CmdStanPathfinder):
            return StanResultEngine.PATHFINDER
        else:
            raise ValueError("Unknown result type")

    @property
    @overrides
    def result_scope(self) -> StanOutputScope:
        return StanOutputScope.RawOutput

    @overrides
    def serialize_to_file(self, output_type: str, file_name: str):
        raise NotImplementedError()

    @property
    def messages(self) -> dict[str, str] | None:
        return self._messages

    @property
    @overrides
    def user_parameters(self) -> list[str]:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultEngine.LAPLACE:
            return list(self._result.metadata.stan_vars.keys())
        elif self.result_type == StanResultEngine.VB:
            return list(self._result.metadata.stan_vars.keys())
        elif self.result_type == StanResultEngine.MCMC:
            return list(self._result.metadata.stan_vars.keys())
        elif self.result_type == StanResultEngine.PATHFINDER:
            return list(self._result.metadata.stan_vars.keys())
        else:
            raise ValueError("Unknown result type")

    @property
    @overrides
    def onedim_parameters(self) -> list[str]:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultEngine.LAPLACE:
            return list(self._result.column_names[2:])
        elif self.result_type == StanResultEngine.VB:
            return list(self._result.column_names[3:])
        elif self.result_type == StanResultEngine.MCMC:
            return list(self._result.column_names[7:])
        elif self.result_type == StanResultEngine.PATHFINDER:
            return list(self._result.column_names[2:])
        else:
            raise ValueError("Unknown result type")

    @overrides
    def get_parameter_shape(self, user_parameter_name: str) -> tuple[int, ...]:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultEngine.LAPLACE:
            return self._result.metadata.stan_vars[user_parameter_name].dimensions
        elif self.result_type == StanResultEngine.VB:
            return self._result.metadata.stan_vars[user_parameter_name].dimensions
        elif self.result_type == StanResultEngine.MCMC:
            return self._result.metadata.stan_vars[user_parameter_name].dimensions
        elif self.result_type == StanResultEngine.PATHFINDER:
            return self._result.metadata.stan_vars[user_parameter_name].dimensions
        else:
            raise ValueError("Unknown result type")

    @overrides
    def sample_count(self, onedim_parameter_name: str = None) -> float | int | None:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultEngine.LAPLACE:
            return self.draws.shape[0]
        elif self.result_type == StanResultEngine.VB:
            return self.draws.shape[0]
        elif self.result_type == StanResultEngine.MCMC:
            if onedim_parameter_name is not None:
                s = self._result.summary()
                return s["N_Eff"][onedim_parameter_name]
            else:
                return self.draws.shape[0]
        elif self.result_type == StanResultEngine.PATHFINDER:
            return self.draws.shape[0]
        else:
            raise ValueError("Unknown result type")

    @property
    @overrides
    def draws(self) -> np.ndarray:
        if self._draws is None:
            if self._result is None:
                raise ValueError("No result available")
            elif self.result_type == StanResultEngine.LAPLACE:
                self._draws = self._result.draws()
            elif self.result_type == StanResultEngine.VB:
                self._draws = self._result.variational_sample
            elif self.result_type == StanResultEngine.MCMC:
                self._draws = self._result.draws(concat_chains=True)
            elif self.result_type == StanResultEngine.PATHFINDER:
                self._draws = self._result.draws()
        return self._draws

    @overrides
    def get_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> ValueWithError:
        if self._result is None:
            raise ValueError("No result available")
        var_index = self._result.column_names.index(onedim_parameter_name)
        if store_values:
            ans = ValueWithErrorVec(self.draws[:, var_index])
        else:
            ans = ValueWithError.CreateFromVector(self.draws[:, var_index], N=self.sample_count(onedim_parameter_name))
        return ans

    @overrides
    def get_parameter_mu(self, user_parameter_name: str) -> np.ndarray:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type == StanResultEngine.LAPLACE:
            return self._result.stan_variable(user_parameter_name)
        elif self.result_type == StanResultEngine.VB:
            return np.mean(self._result.stan_variable(user_parameter_name), axis=0)
        elif self.result_type == StanResultEngine.MCMC:
            return np.mean(self._result.stan_variable(user_parameter_name), axis=0)
        elif self.result_type == StanResultEngine.PATHFINDER:
            return self._result.stan_variable(user_parameter_name)
        else:
            raise ValueError("Unknown result type")

    @overrides
    def get_parameter_sigma(self, user_parameter_name: str) -> np.ndarray:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type != StanResultEngine.NONE:
            return np.std(self._result.stan_variable(user_parameter_name), axis=0)
        else:
            raise ValueError("Unknown result type")

    @overrides
    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        if self._result is None:
            raise ValueError("No result available")
        elif self.result_type != StanResultEngine.NONE:
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

    @overrides
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

    @overrides
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

    @overrides
    def method_name(self) -> str:
        if self.result_type == StanResultEngine.NONE:
            return "No result available"
        elif self.result_type == StanResultEngine.LAPLACE:
            return "Laplace approximation around the posterior mode"
        elif self.result_type == StanResultEngine.PATHFINDER:
            return "Pathfinder variational inference algorithm."
        elif self.result_type == StanResultEngine.VB:
            return f"Variational inference algorithm {self._result.metadata.cmdstan_config['algorithm']}"
        elif self.result_type == StanResultEngine.MCMC:
            return f"MCMC algorithm {self._result.metadata.cmdstan_config['algorithm']}, engine {self._result.metadata.cmdstan_config['engine']}"

    @overrides
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

    @overrides
    def formatted_runtime(self) -> str:
        if self._runtime is None:
            return "Run time: not available"
        else:
            return f"Run taken: {humanize.precisedelta(self._runtime)}"

    @overrides
    def __repr__(self):
        if self.sample_count is None:
            return self.repr_without_sampling_errors()
        else:
            return self.repr_with_sampling_errors()

    @overrides
    def all_main_effects(self) -> dict[str, ValueWithError]:
        out = {}
        for par in self.onedim_parameters:
            out[par] = self.get_parameter_estimate(par)
        return out

    def serialize_to_dict(self, output_type: str):
        if output_type == "main_effects":
            ans = self.all_main_effects()
            return {key: {"value": ans[key].value, "error": ans[key].SE, "N": ans[key].N} for key in ans}
        if output_type == "covariances":
            means = self._result.stan_variables()
            cov_matrix, one_dim_names = self.get_cov_matrix()
            out = {}
            out["cov"] = cov_matrix.tolist()
            out["names"] = one_dim_names
            out["means"] = means
            return out
        if output_type == "draws":
            out = {}
            out["draws"] = self.draws.tolist()
            out["names"] = self.onedim_parameters
            return out
        if output_type == "raw":
            file = self.serialize()
            # Encode file as base64
            with open(file, "rb") as f:
                data = f.read()
                return base64.b64encode(data).decode("utf-8")


        raise ValueError("Unknown output type")

    def to_json(self, output_type: str) -> str:
        if output_type == "all":
            return jsonpickle.encode(self)
        return json.dumps(self.serialize_to_dict(output_type))

    def serialize(self) -> Path:
        """Serializes the method into a single file that is readable."""
        rs: RunSet = self._result._runset

        a: CmdStanArgs = rs._args
        output_dir = Path("/") / a.output_dir
        obj = {"runset": rs, "messages": self._messages, "runtime": self._runtime}
        pickle.dump(obj, open(output_dir / "runset.pkl", "wb"))

        # Compress the output directory
        zip_file = output_dir.parent / (output_dir.name + ".zip")
        shutil.make_archive(str(output_dir), 'zip', zip_file)
        return zip_file

    @staticmethod
    def Deserialize(zip_path: Path, bDeleteAfterwards: bool = False) -> InferenceResult:

        dest_dir = Path(tempfile.TemporaryDirectory().name)

        # Unzip the zip_path into the dest_dir

        shutil.unpack_archive(str(zip_path), dest_dir)

        shutil.rmtree(dest_dir, ignore_errors=True)
        shutil.unpack_archive(str(zip_path), dest_dir)

        runset_path = Path(dest_dir) / "runset.pkl"
        assert runset_path.exists()

        obj: dict = pickle.load(open(dest_dir / "runset.pkl", "rb"))
        rs2: RunSet = obj["runset"]
        rs2._args.output_dir = str(dest_dir)
        rs2._csv_files = [str(dest_dir / Path(item).name) for item in rs2.csv_files]
        rs2._stdout_files = [str(dest_dir / Path(item).name) for item in rs2.stdout_files]

        # 'SAMPLE', 'VARIATIONAL', 'LAPLACE', PATHFINDER
        if rs2._args.method.name == "SAMPLE":
            stanObj = CmdStanMCMC(rs2)
        elif rs2._args.method.name == "VARIATIONAL":
            stanObj = CmdStanVB(rs2)
        elif rs2._args.method.name == "LAPLACE":
            stanObj = CmdStanLaplace(rs2)
        elif rs2._args.method.name == "PATHFINDER":
            stanObj = CmdStanPathfinder(rs2)
        else:
            raise ValueError("Unknown method")

        output = InferenceResult(stanObj, messages=obj["messages"], runtime=obj["runtime"])

        if bDeleteAfterwards:
            shutil.rmtree(zip_path)

        return output
