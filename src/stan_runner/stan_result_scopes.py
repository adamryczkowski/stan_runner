import base64
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ValueWithError import ValueWithError, IValueWithError, ValueWithErrorVec
from cmdstanpy import CmdStanLaplace, CmdStanVB, CmdStanMCMC, CmdStanPathfinder
from cmdstanpy.cmdstan_args import CmdStanArgs
from cmdstanpy.stanfit.vb import RunSet
from nats_foundation import create_dict_serializable_copy
from overrides import overrides

from .ifaces import StanResultEngine
from .ifaces2 import IStanRun, IStanResultBase, IStanResultCovariances, IStanResultFullSamples, \
    IStanResultRawResult
from .stan_result_base import ImplStanResultMetaWithUser2Onedim, ImplStanResultBase, ImplCovarianceInterface, \
    ImplValueWithError
from .utils import infer_param_shapes


class StanResultMainEffects(ImplStanResultBase, ImplStanResultMetaWithUser2Onedim, ImplValueWithError, IStanResultBase):
    _one_dim_pars: dict[str, ValueWithError]

    _user2dim: dict[str, tuple[int, ...]]  # Caches the shape of the parameters including the sample count
    _user2onedim: dict[str, list[str]]  # Caches the one-dim parameter names

    def __init__(self, run: IStanRun, output: str, warnings: str, errors: str, runtime: float,
                 one_dim_pars: dict[str, ValueWithError]):
        super().__init__(run=run, output=output, warnings=warnings, errors=errors, runtime=runtime)

        assert isinstance(one_dim_pars, dict)
        for k, v in one_dim_pars.items():
            assert isinstance(k, str)
            assert isinstance(v, ValueWithError)
        self._user2dim, self._user2onedim = infer_param_shapes(list(one_dim_pars.keys()))
        for k in self._user2onedim.keys():
            self._user2dim[k] = self._user2dim[k] + (one_dim_pars[k].N,)

        self._one_dim_pars = one_dim_pars

    @overrides
    def _get_user2onedim(self) -> dict[str, list[str]]:
        return self._user2onedim

    @overrides
    def _get_user2dim(self) -> dict[str, tuple[int, ...]]:
        return self._user2dim

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["one_dim_pars"] = {k: v.__getstate__() for k, v in self._one_dim_pars.items()}
        return d

    @overrides
    def get_onedim_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> IValueWithError:
        if store_values:
            raise ValueError("store_values is not supported for this result scope")

        if onedim_parameter_name not in self._one_dim_pars:
            raise ValueError(f"Parameter {onedim_parameter_name} not found in the result")

        return self._one_dim_pars[onedim_parameter_name]


class StanResultCovariances(ImplStanResultMetaWithUser2Onedim, ImplStanResultBase, ImplCovarianceInterface,
                            ImplValueWithError, IStanResultCovariances):
    _main_effects = np.ndarray
    _covariances = np.ndarray
    _onedim2idx: dict[str, int]  # Translates onedim name into the index of the arrays above

    _user2dim: dict[str, tuple[int, ...]]  # Caches the shape of the parameters including the sample count
    _user2onedim: dict[str, list[str]]  # Caches the one-dim parameter names

    def __init__(self, run: IStanRun, output: str, warnings: str, errors: str, runtime: float,
                 main_effects: np.ndarray, covariances: np.ndarray, effective_sample_sizes: np.ndarray,
                 onedim_names: list[str] | np.ndarray):
        super().__init__(run, output, warnings, errors, runtime)

        assert isinstance(main_effects, np.ndarray)
        assert isinstance(main_effects.dtype, float)
        assert len(main_effects.shape) == 1
        assert isinstance(covariances, np.ndarray)
        assert isinstance(covariances.dtype, float)
        assert len(covariances.shape) == 2  # Matrix
        assert covariances.shape[0] == covariances.shape[1]
        assert main_effects.shape[0] == covariances.shape[0]
        assert isinstance(effective_sample_sizes, np.ndarray)
        assert isinstance(effective_sample_sizes.dtype, (int, float))
        assert len(effective_sample_sizes.shape) == 1
        assert effective_sample_sizes.shape[0] == main_effects.shape[0]
        if isinstance(onedim_names, np.ndarray):
            assert onedim_names.dtype == str
            onedim_names = onedim_names.tolist()
        assert isinstance(onedim_names, list)
        assert len(onedim_names) == main_effects.shape[0]

        self._onedim2idx = {k: idx for idx, k in enumerate(onedim_names)}

        self._user2dim, self._user2onedim = infer_param_shapes(onedim_names)

        for k in self._user2onedim.keys():
            self._user2dim[k] = self._user2dim[k] + (effective_sample_sizes[k],)

    @overrides
    def _get_user2onedim(self) -> dict[str, list[str]]:
        return self._user2onedim

    @overrides
    def _get_user2dim(self) -> dict[str, tuple[int, ...]]:
        return self._user2dim

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["main_effects"] = self._main_effects
        # Translates covariances to triangular form to save space
        d["covariances"] = self._covariances[np.triu_indices(self._covariances.shape[0])]

        d["effective_sizes"] = [self.sample_count(one_dim_name) for one_dim_name in self.onedim_parameters]
        return create_dict_serializable_copy(d)

    @overrides
    def get_cov_onedim_par(self, one_dim_par1: str, one_dim_par2: str) -> float | np.ndarray:
        idx1 = self._onedim2idx[one_dim_par1]
        idx2 = self._onedim2idx[one_dim_par2]
        return float(self._covariances[idx1, idx2])

    @overrides
    def get_onedim_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> IValueWithError:
        if store_values:
            raise ValueError("store_values is not supported for this result scope")

        if onedim_parameter_name not in self._onedim2idx:
            raise ValueError(f"Parameter {onedim_parameter_name} not found in the result")

        idx = self._onedim2idx[onedim_parameter_name]
        return ValueWithError(value=self._main_effects[idx],
                              SE=np.sqrt(self._covariances[idx, idx]),
                              N=self.sample_count(onedim_parameter_name))


class StanResultFullSamples(ImplStanResultMetaWithUser2Onedim, ImplStanResultBase, ImplCovarianceInterface,
                            ImplValueWithError, IStanResultFullSamples):
    _one_dim_pars: dict[str, ValueWithErrorVec]

    _user2dim: dict[str, tuple[int, ...]]  # Caches the shape of the parameters including the sample count
    _user2onedim: dict[str, list[str]]  # Caches the one-dim parameter names

    def __init__(self, run: IStanRun, output: str, warnings: str, errors: str, runtime: float, draws: np.ndarray,
                 onedim_names: list[str] | np.ndarray, effective_sample_sizes: np.ndarray = None):
        super().__init__(run=run, output=output, warnings=warnings, errors=errors, runtime=runtime)

        assert isinstance(draws, np.ndarray)
        assert isinstance(draws.dtype, float)
        assert len(draws.shape) == 2

        if effective_sample_sizes is None:
            effective_sample_sizes = np.ones(draws.shape[1]) * draws.shape[0]
        else:
            assert isinstance(effective_sample_sizes, np.ndarray)
            assert isinstance(effective_sample_sizes.dtype, (int, float))
            assert len(effective_sample_sizes.shape) == 1
            assert effective_sample_sizes.shape[0] == draws.shape[1]

        if isinstance(onedim_names, np.ndarray):
            assert onedim_names.dtype == str
            onedim_names = onedim_names.tolist()
        assert isinstance(onedim_names, list)
        assert len(onedim_names) == draws.shape[1]

        self._user2dim, self._user2onedim = infer_param_shapes(onedim_names)

        self._one_dim_pars = {onedim_names[i]:
                                  ValueWithErrorVec(draws[i], override_len=float(effective_sample_sizes[i]))
                              for i in range(draws.shape[0])}

    @overrides
    def _get_user2onedim(self) -> dict[str, list[str]]:
        return self._user2onedim

    @overrides
    def _get_user2dim(self) -> dict[str, tuple[int, ...]]:
        return self._user2dim

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["draws"] = self.draws(None)
        d["names"] = list(self._one_dim_pars.keys())
        d["effective_sizes"] = [self.sample_count(one_dim_name) for one_dim_name in self.onedim_parameters]
        return d

    @overrides
    def draws(self, onedim_varname: str) -> np.ndarray:
        return self._one_dim_pars[onedim_varname].vector

    @overrides
    def get_cov_onedim_par(self, one_dim_par1: str, one_dim_par2: str) -> float | np.ndarray:
        idx1 = self._user2onedim[one_dim_par1]
        idx2 = self._user2onedim[one_dim_par2]
        return np.cov(self._one_dim_pars[idx1].vector, self._one_dim_pars[idx2].vector)

    @overrides
    def get_onedim_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> IValueWithError:
        if onedim_parameter_name not in self._one_dim_pars:
            raise ValueError(f"Parameter {onedim_parameter_name} not found in the result")
        if store_values:
            return self._one_dim_pars[onedim_parameter_name]
        else:
            return ValueWithError.CreateFromVector(self._one_dim_pars[onedim_parameter_name].vector,
                                                   N=self._one_dim_pars[onedim_parameter_name].N)


class StanResultRawResult(ImplStanResultBase, ImplCovarianceInterface, ImplValueWithError,
                          ImplStanResultMetaWithUser2Onedim, IStanResultRawResult):
    _result: CmdStanLaplace | CmdStanVB | CmdStanMCMC | CmdStanPathfinder | None
    _draws_cache: np.ndarray | None = None
    _draws_names: tuple[str, ...] | None = None
    _names_dict: dict[str, int] | None = None
    _draws_offset: int | None
    _user2onedim: dict[str, list[str]] | None = None
    _user2dim: dict[str, tuple[int, ...]] | None = None

    def __init__(self, run: IStanRun, output: str, warnings: str, errors: str, runtime: float,
                 result: CmdStanLaplace | CmdStanVB | CmdStanMCMC | CmdStanPathfinder | None):
        super().__init__(run=run, output=output, warnings=warnings, errors=errors, runtime=runtime)
        if result is not None:
            assert isinstance(result, (CmdStanLaplace, CmdStanVB, CmdStanMCMC, CmdStanPathfinder))

        self._result = result
        self._draws_cache = None
        self._draws_names = None
        self._draws_offset = None
        self._names_dict = None
        self._user2onedim = None
        self._user2dim = None

    @overrides
    def serialize_to_file(self) -> Path:
        """Serializes the method into a single file that is readable."""
        try:
            rs: RunSet = self._result.runset
        except AttributeError:
            try:
                rs = self._result._runset
            except AttributeError:
                raise ValueError("No result available")

        a: CmdStanArgs = rs._args
        output_dir = Path("/") / a.output_dir
        obj = {"runset": rs, "messages": self.messages, "warnings": self.warnings, "errors": self.errors,
               "runtime": self._runtime}
        if self.result_type == StanResultEngine.LAPLACE:
            obj["laplace_mode"] = self._result.mode
        pickle.dump(obj, open(output_dir / "runset.pkl", "wb"))

        # Compress the output directory
        zip_file = output_dir.parent / output_dir.name
        zip_file = shutil.make_archive(str(output_dir), 'zip', zip_file)
        return Path(zip_file)

    def serialize_result(self) -> bytes:
        """Serializes the result folder into a zip-compressed byte array."""
        try:
            rs: RunSet = self._result.runset
        except AttributeError:
            try:
                rs = self._result._runset
            except AttributeError:
                raise ValueError("No result available")

        a: CmdStanArgs = rs._args
        source_dir = Path("/") / a.output_dir

        assert self._result is not None
        with tempfile.NamedTemporaryFile(delete_on_close=True, delete=True) as tmp_file:
            shutil.make_archive(str(tmp_file.name), 'zip', source_dir)
            with open(tmp_file.name + ".zip", "rb") as f:
                return f.read()

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["result"] = base64.b64encode(self.serialize_result())
        return d

    @property
    @overrides
    def get_raw_result(self) -> Any:
        return self._result

    @overrides
    def draws_of_all_variables(self, bInclSpecial=False) -> tuple[np.ndarray, tuple[str, ...]]:
        if self._result is None:
            raise ValueError("No result available")
        self._get_draws()
        if not bInclSpecial:
            return self._draws_cache, self._draws_names
        else:
            return self._draws_cache[:, self._draws_offset:], self._draws_names[self._draws_offset:]

    def _get_meta(self):
        if self._result is None:
            return None
        if self._draws_offset is not None:
            return
        if isinstance(self._result, CmdStanLaplace):
            offset = 2
        elif isinstance(self._result, CmdStanVB):
            offset = 3
        elif isinstance(self._result, CmdStanMCMC):
            offset = 7
        elif isinstance(self._result, CmdStanPathfinder):
            offset = 2
        else:
            assert False

        self._draws_offset = offset
        self._names_dict = {name: idx for idx, name in enumerate(self._draws_names)}
        self._draws_names = self._result.column_names
        self._user2dim, self._user2onedim = infer_param_shapes(self._draws_names[offset:])

    def _get_draws(self):
        if self._result is None:
            return
        if self._draws_cache is not None:
            return
        if isinstance(self._result, CmdStanLaplace):
            draws = self._result.draws()
        elif isinstance(self._result, CmdStanVB):
            draws = self._result.variational_sample
        elif isinstance(self._result, CmdStanMCMC):
            draws = self._result.draws(concat_chains=True)
        elif isinstance(self._result, CmdStanPathfinder):
            draws = self._result.draws()
        else:
            assert False

        self._draws_cache = draws
        self._get_meta()

    @overrides
    def draws(self, onedim_varname: str) -> np.ndarray:
        if self._result is None:
            raise ValueError("No result available")
        self._get_draws()

        idx = self._names_dict[onedim_varname]
        return self._draws_cache[:, idx]

    @overrides
    def get_cov_onedim_par(self, one_dim_par1: str, one_dim_par2: str) -> float | np.ndarray:
        self._get_draws()
        return np.cov(self.draws(one_dim_par1), self.draws(one_dim_par2))

    @overrides
    def get_onedim_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> IValueWithError:
        vector = self.draws(onedim_parameter_name)
        if store_values:
            return ValueWithErrorVec(vector)
        else:
            return ValueWithError.CreateFromVector(vector)

    @property
    @overrides
    def one_dim_parameters_count(self) -> int:
        if self._result is None:
            return 0
        self._get_meta()
        return len(self._result.column_names) - self._draws_offset

    @overrides
    def _get_user2onedim(self) -> dict[str, list[str]]:
        self._get_meta()
        return self._user2onedim

    @overrides
    def _get_user2dim(self) -> dict[str, tuple[int, ...]]:
        self._get_meta()
        return self._user2dim
