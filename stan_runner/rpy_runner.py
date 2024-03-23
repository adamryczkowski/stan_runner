import contextlib
import io
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import rpy2
import rpy2.rinterface_lib.callbacks
from ValueWithError import ValueWithErrorCI, ValueWithErrorVec, ValueWithError
from overrides import overrides
from rpy2.robjects.methods import RS4
from rpy2.robjects.packages import importr

from .ifaces import IStanResult, StanErrorType, StanResultType
from .r_utils import rcode_load_library, normalize_stan_code, convert_dict_to_r

stan_CI_levels_dict = {0: 0.95, 1: 0.9, 2: 0.8, 3: 0.5}


class RPyRunner(IStanResult):
    _number_of_cores: int
    _stanc_opts: dict[str, Any]
    _model_cache: Path
    _initialized: bool
    _rstan: rpy2.robjects.packages.InstalledSTPackage | None

    _model_filename: Path | None
    _last_model_hash: str
    _last_model_code: RS4 | None
    _last_model_obj: RS4 | None
    _pars_of_interest: list[str]

    _data: Any

    _result: RS4 | Any
    _result_type: StanResultType
    _stan_extract: Any | None

    _user_parameters: list[str] | None
    _onedim_parameters: list[str] | None

    _messages: dict[str, str]

    def __init__(self, model_cache: Path, number_of_cores: int = 1, allow_optimizations_for_stanc: bool = True):
        assert isinstance(model_cache, Path)
        if not model_cache.exists():
            model_cache.mkdir(parents=True)
        assert model_cache.is_dir()
        assert isinstance(number_of_cores, int)
        assert number_of_cores > 0

        self._number_of_cores = number_of_cores
        self._model_cache = model_cache

        self.clear_last_model()
        self.clear_last_data()
        self.clear_last_results()

        self._initialized = False

        self._stanc_opts = {"allow_optimizations": allow_optimizations_for_stanc}
        self._pars_of_interest = rpy2.robjects.NA_Logical
        self._rstan = None

    def install_dependencies(self):
        r_str = """if(!dir.exists(Sys.getenv("R_LIBS_USER"))) {
  dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE)
}"""
        rpy2.robjects.reval(r_str)

        r_str = rcode_load_library('pacman')
        rpy2.robjects.reval(r_str)

        pacman = importr('pacman')
        pacman.p_load('rstan', 'purrr')
        self._rstan = importr("rstan")

    @overrides
    def clear_last_model(self):
        """Also clears results and data"""
        self._last_model_hash = ""
        self._last_model_code = None
        self._last_model_obj = None
        self._model_filename = None
        self._messages = {}
        self._pars_of_interest = rpy2.robjects.NA_Logical

        if "stanc_output" in self._messages:
            del self._messages["stanc_output"]
        if "stanc_warnings" in self._messages:
            del self._messages["stanc_warnings"]
        if "stanc_error" in self._messages:
            del self._messages["stanc_error"]
        if "compile_output" in self._messages:
            del self._messages["compile_output"]
        if "compile_warnings" in self._messages:
            del self._messages["compile_warnings"]
        if "compile_error" in self._messages:
            del self._messages["compile_error"]

        self.clear_last_data()
        self.clear_last_results()

    @overrides
    def clear_last_data(self):
        self._data = None

        self.clear_last_results()

    @overrides
    def clear_last_results(self):
        self._result = None
        self._result_type = StanResultType.Nothing
        self._onedim_parameters = None
        self._user_parameters = None
        self._stan_extract = None

        if "sampling_output" in self._messages:
            del self._messages["sampling_output"]
        if "sampling_warnings" in self._messages:
            del self._messages["sampling_warnings"]
        if "sampling_error" in self._messages:
            del self._messages["sampling_error"]

    @property
    @overrides
    def result_type(self) -> StanResultType:
        return self._result_type

    @property
    @overrides
    def error_state(self) -> StanErrorType:
        if "stanc_error" in self._messages:
            return StanErrorType.SYNTAX_ERROR
        if "compile_error" in self._messages:
            return StanErrorType.COMPILE_ERROR
        if "sampling_error" in self._messages:
            return StanErrorType.SAMPLING_ERROR
        return StanErrorType.NO_ERROR

    @property
    @overrides
    def messages(self) -> dict[str, str]:
        return self._messages

    @property
    @overrides
    def initialized(self) -> bool:
        return self._initialized

    @property
    @overrides
    def is_model_loaded(self) -> bool:
        return self._model_filename is not None

    @property
    @overrides
    def is_model_compiled(self) -> bool:
        return self._last_model_obj is not None

    def set_data(self, data: dict[str, int | float | np.ndarray]):
        data = convert_dict_to_r(data)
        self._data = data

    @property
    @overrides
    def is_data_set(self) -> bool:
        return self._data is not None

    def load_model_by_str(self, model_code: str | list[str], model_name: str, pars_list: list[str] = None):
        if isinstance(model_code, list):
            model_code = "\n".join(model_code)
        assert isinstance(model_code, str)
        assert isinstance(model_name, str)
        model_code = normalize_stan_code(model_code)
        # purrr = importr("purrr")
        self.clear_last_model()

        stdout = []
        stderr = []

        def add_to_stdout(line):
            stdout.append(line)

        def add_to_stderr(line):
            stderr.append(line)

        stdout_orig = rpy2.rinterface_lib.callbacks.consolewrite_print
        stderr_orig = rpy2.rinterface_lib.callbacks.consolewrite_warnerror

        rpy2.rinterface_lib.callbacks.consolewrite_print = add_to_stdout
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = add_to_stderr

        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                self._last_model_code = self._rstan.stanc(model_code=model_code, model_name=model_name, verbose=True,
                                                          use_opencl=self._stanc_opts.get("allow_optimizations", False),
                                                          allow_optimizations=self._stanc_opts["allow_optimizations"])

            # result =  purrr.quietly(rstan.stanc)(model_code=model_code, model_name=model_name, verbose=True,
            #                                     use_opencl=self._stanc_opts.get("allow_optimizations", False),
            #                                     allow_optimizations=self._stanc_opts["allow_optimizations"])
        except rpy2.rinterface_lib.embedded.RRuntimeError as e:
            self._messages["sampling_output"] = "\n".join(stdout)
            self._messages["sampling_warnings"] = "\n".join(stderr)
            self._messages["sampling_error"] = str(e)
            return
        finally:
            rpy2.rinterface_lib.callbacks.consolewrite_print = stdout_orig
            rpy2.rinterface_lib.callbacks.consolewrite_warnerror = stderr_orig

        output = "\n".join(stdout)
        warnings = "\n".join(stderr)

        self._messages["stanc_output"] = output
        self._messages["stanc_warnings"] = warnings

        model_code = list(self._last_model_code.rx2["model_code"])[0]
        # model_cpp = list(self._last_model_code.rx2["cppcode"])[0]
        model_hash = sha256(model_code.encode()).hexdigest()
        if model_hash == self._last_model_hash:
            return

        # Save the model to a file in a cache
        model_filename = find_model_in_cache(self._model_cache, model_name, model_hash)
        if not model_filename.exists():
            with model_filename.open("w") as f:
                f.write(model_code)

        if pars_list is None:
            self._pars_of_interest = rpy2.robjects.NA_Logical
        else:
            self._pars_of_interest = rpy2.robjects.StrVector(pars_list)

        self._model_filename = model_filename
        self._last_model_hash = model_hash

    def compile_model(self):
        if self._last_model_obj is not None:
            return
        assert self.is_model_loaded

        model_rds_filename = self._model_filename.with_suffix(".rds")
        if model_rds_filename.exists():
            model_obj = rpy2.robjects.r.readRDS(str(model_rds_filename))
            modej_hash = sha256(str(list(model_obj.slots["model_code"])[0]).encode()).hexdigest()
            if modej_hash == self._last_model_hash:
                self._last_model_obj = model_obj
                return
            else:
                raise RuntimeError(
                    f"Model hash mismatch for model {self._model_filename}! User should delete that model and inspect the reason for hash collision.")

        stdout = []
        stderr = []

        def add_to_stdout(line):
            stdout.append(line)

        def add_to_stderr(line):
            stderr.append(line)

        stdout_orig = rpy2.rinterface_lib.callbacks.consolewrite_print
        stderr_orig = rpy2.rinterface_lib.callbacks.consolewrite_warnerror

        rpy2.rinterface_lib.callbacks.consolewrite_print = add_to_stdout
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = add_to_stderr
        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                self._last_model_obj = self._rstan.stan_model(file=str(self._model_filename),
                                                        stanc_ret=self._last_model_code,
                                                        auto_write=False,
                                                        allow_optimizations=self._stanc_opts[
                                                            "allow_optimizations"],
                                                        verbose=True)
        except rpy2.rinterface_lib.embedded.RRuntimeError as e:
            self._messages["sampling_output"] = "\n".join(stdout)
            self._messages["sampling_warnings"] = "\n".join(stderr)
            self._messages["sampling_error"] = str(e)
            return
        finally:
            rpy2.rinterface_lib.callbacks.consolewrite_print = stdout_orig
            rpy2.rinterface_lib.callbacks.consolewrite_warnerror = stderr_orig

        self._messages["compile_output"] = "\n".join(stdout)
        self._messages["compile_warnings"] = "\n".join(stderr)

        # Save the model to a file in a cache
        rpy2.robjects.r.saveRDS(self._last_model_obj, str(model_rds_filename))

    def sampling(self, num_samples: int, num_chains: int, warmup: int = 1000,
                 seed: int = None):
        assert self.is_data_set
        assert self.is_model_loaded
        # Load the model in R
        if seed is not None:
            rpy2.robjects.r.set_seed(seed)
        else:
            seed = np.random.randint(0, 2 ** 31 - 1, dtype=np.int32)

        stdout = []
        stderr = []

        def add_to_stdout(line):
            stdout.append(line)

        def add_to_stderr(line):
            stderr.append(line)

        stdout_orig = rpy2.rinterface_lib.callbacks.consolewrite_print
        stderr_orig = rpy2.rinterface_lib.callbacks.consolewrite_warnerror

        rpy2.rinterface_lib.callbacks.consolewrite_print = add_to_stdout
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = add_to_stderr

        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                fit = self._rstan.sampling(self._last_model_obj, data=self._data, chains=num_chains,
                                     iter=int(num_samples + warmup),
                                     warmup=int(warmup), seed=seed, cores=int(self._number_of_cores),
                                     verbose=True, pars=self._pars_of_interest)
        except rpy2.rinterface_lib.embedded.RRuntimeError as e:
            self._messages["sampling_output"] = "\n".join(stdout)
            self._messages["sampling_warnings"] = "\n".join(stderr)
            self._messages["sampling_error"] = str(e)
            return
        finally:
            rpy2.rinterface_lib.callbacks.consolewrite_print = stdout_orig
            rpy2.rinterface_lib.callbacks.consolewrite_warnerror = stderr_orig
        self._messages["sampling_output"] = "\n".join(stdout)
        self._messages["sampling_warnings"] = "\n".join(stderr)

        self._result = fit
        self._result_type = StanResultType.Sampling

        summary = self._rstan.summary_sim(fit.slots['sim'])
        self._onedim_parameters = list(rpy2.robjects.reval("(function(x){rownames(x$msd)})")(summary))
        user_pars = [p for p in fit.slots["model_pars"] if p != "lp__"]
        self._user_parameters = user_pars

    def variational_bayes(self, seed: int = None,
                          algorithm: str = "meanfield",
                          importance_resampling: bool = False, iter: int = 10000, grad_samples: int = 1,
                          elbo_samples: int = 100, eta: float = None, adapt_engaged: bool = True,
                          tol_rel_obj: float = 0.01, eval_elbo: int = 100, output_samples: int = 1000,
                          adapt_iter: int = 50):
        assert self.is_data_set
        assert self.is_model_loaded
        assert False
        # Load the model in R
        rstan = importr("rstan")
        purrr = importr("purrr")
        if seed is not None:
            rpy2.robjects.r.set_seed(seed)

        result = purrr.quietly(rstan.vb)(self._last_model_obj, data=self._data, seed=seed, algorithm=algorithm,
                                         importance_resampling=importance_resampling,
                                         iter=iter, grad_samples=grad_samples, elbo_samples=elbo_samples, eta=eta,
                                         adapt_engaged=adapt_engaged, tol_rel_obj=tol_rel_obj, eval_elbo=eval_elbo,
                                         output_samples=output_samples, adapt_iter=adapt_iter,
                                         cores=self._number_of_cores,
                                         verbose=True, pars=self._pars_of_interest)
        output = "\n".join(list(result.rx2("output")))
        warnings = "\n".join(list(result.rx2("warnings")))
        fit = result.rx2("result")
        self._messages["sampling_output"] = output
        self._messages["sampling_warnings"] = warnings

        self._result = fit
        self._result_type = StanResultType.ADVI

        stan = importr("rstan")

        summary = stan.summary_sim(result.slots['sim'])
        self._onedim_parameters = list(rpy2.robjects.reval("(function(x){rownames(x$msd)})")(summary))
        user_pars = [p for p in fit.slots["model_pars"] if p != "lp__"]
        self._user_parameters = user_pars

    def MAP_estimate(self, seed: int = None, algorithm: str = "LBFGS",
                     importance_resampling: bool = False, iter: int = 2000, init_alpha: float = 0.001,
                     tol_obj: float = 1e-12, tol_rel_obj: float = 1e4, tol_grad: float = 1e-8,
                     tol_rel_grad: float = 1e8, tol_param: float = 1e-7, history_size: int = 5):
        assert self.is_data_set
        assert self.is_model_loaded
        # Load the model in R
        rstan = importr("rstan")
        model = self._last_model_obj
        if seed is not None:
            rpy2.robjects.r.set_seed(seed)

        fit = rstan.optimizing(model, data=self._data, seed=seed, verbose=True, algorithm=algorithm, hessian=True,
                               importance_resampling=importance_resampling, iter=iter, init_alpha=init_alpha,
                               tol_obj=tol_obj, tol_rel_obj=tol_rel_obj, tol_grad=tol_grad, tol_rel_grad=tol_rel_grad,
                               tol_param=tol_param, history_size=history_size)
        # TODO: Honor the self._pars_of_interest
        self._result = fit
        self._result_type = StanResultType.Laplace

    def _get_result(self, use_vb: bool, store_vectors: bool, CI_level_nr: int | dict[str, int] = 2) -> dict[
        str, ValueError]:
        assert self.is_data_set
        assert self.is_model_compiled
        if use_vb:
            result = self._vb
        else:
            result = self._fit
        assert result is not None

        stan = importr("rstan")

        summary = stan.summary_sim(result.slots['sim'])
        par_names_R = list(rpy2.robjects.reval("(function(x){rownames(x$msd)})")(summary))

        pars = {}

        if store_vectors:
            simulations = stan.extract(result)
        else:
            msd = summary.rx2['msd']
            quan = summary.rx2['quan']
        # model_names = list(result.slots['model_pars'])

        for i, key in enumerate(par_names_R):
            if key == "lp__":
                break  # This is the last parameter.

            # Remove '[\d+]' from the end of the name
            key_bare = key.split("[")[0]
            key_index = int(key.split("[")[1][:-1])
            if isinstance(CI_level_nr, dict):
                if key_bare in CI_level_nr:
                    CI_level = CI_level_nr[key_bare]
                else:
                    CI_level = 2
            else:
                CI_level = CI_level_nr
            if CI_level not in stan_CI_levels_dict:
                raise ValueError(f"CI_level {CI_level} must be an integer from 0 (95% CI) to 3 (50% CI).")

            if not store_vectors:
                par = ValueWithErrorCI(value=msd[i, 0], SE=msd[i, 1], ci_level=0.80, ci_lower=quan[i, CI_level],
                                       ci_upper=quan[i, -CI_level])
            if key_bare in pars:
                if store_vectors:
                    continue
                if isinstance(pars[key_bare], list):
                    pars[key_bare].append(par)
                else:
                    pars[key_bare] = [pars[key_bare], par]
                assert len(pars[key_bare]) == key_index
            else:
                if store_vectors:
                    arr: np.ndarray = simulations.rx2(key_bare)
                    pars[key_bare] = [ValueWithErrorVec(arr[:, i]) for i in range(arr.shape[1])]
                else:
                    pars[key_bare] = par
        return par
        # if self._MAP is not None:
        # # base = importr("base")
        # hessian = self._MAP.rx2["hessian"]
        # par_names = set(rpy2.robjects.r.colnames(self._MAP.rx2["hessian"]))
        # pars = [p for p in self._MAP.rx2["par"] if p in par_names]

    @property
    @overrides
    def user_parameters(self) -> list[str]:
        return self._user_parameters

    @property
    @overrides
    def onedim_parameters(self) -> list[str]:
        return self._onedim_parameters

    @property
    @overrides
    def user_parameter_count(self) -> int:
        return len(self._user_parameters)

    @property
    @overrides
    def onedim_parameter_count(self) -> int:
        return len(self._onedim_parameters)

    @overrides
    def get_parameter_shape(self, user_par_name: str) -> list[int]:
        raise "Not implemented yet"

    def _get_stan_extract(self):
        if self._stan_extract and self._result == StanResultType.Sampling or self._result == StanResultType.ADVI:
            stan = importr("rstan")
            self._stan_extract = stan.extract(self._result)
        return self._stan_extract

    @overrides
    def get_parameter_estimate(self, onedim_par_name: str, store_values: bool = False) -> ValueWithError:
        if self._result_type == StanResultType.Sampling or self._result_type == StanResultType.ADVI:
            if store_values:
                stan_extract = self._get_stan_extract()
                arr: np.ndarray = stan_extract.rx2(onedim_par_name)
                return ValueWithErrorVec(arr)
            else:
                stan = importr("rstan")
                summary = stan.summary_sim(self._result.slots['sim'])
                par_names_R = list(rpy2.robjects.reval("(function(x){rownames(x$msd)})")(summary))
                i = par_names_R.index(onedim_par_name)
                msd = summary.rx2['msd']
                quan = summary.rx2['quan']
                return ValueWithErrorCI(value=msd[i, 0], SE=msd[i, 1], ci_level=0.80, ci_lower=quan[i, 2],
                                        ci_upper=quan[i, -2])
        elif self._result_type == StanResultType.Laplace:
            if store_values:
                raise ValueError("Laplace approximation does not provide samples.")
            else:  # TODO
                raise NotImplementedError("Laplace approximation not implemented yet.")
        else:
            raise ValueError("No results available.")

    @overrides
    def get_parameter_mu(self, user_par_name: str) -> np.ndarray:
        if self._result_type == StanResultType.Sampling or self._result_type == StanResultType.ADVI:
            stan = importr("rstan")
            summary = stan.summary_sim(self._result.slots['sim'])
            par_names_R = list(rpy2.robjects.reval("(function(x){rownames(x$msd)})")(summary))
            i = par_names_R.index(user_par_name)
            return np.array([summary.rx2['msd'][i, 0]])
        elif self._result_type == StanResultType.Laplace:
            raise NotImplementedError("Laplace approximation not implemented yet.")
        else:
            raise ValueError("No results available.")

    @overrides
    def get_parameter_sigma(self, user_par_name: str) -> np.ndarray:
        if self._result_type == StanResultType.Sampling or self._result_type == StanResultType.ADVI:
            stan = importr("rstan")
            summary = stan.summary_sim(self._result.slots['sim'])
            par_names_R = list(rpy2.robjects.reval("(function(x){rownames(x$msd)})")(summary))
            i = par_names_R.index(user_par_name)
            return np.array([summary.rx2['msd'][i, 1]])
        elif self._result_type == StanResultType.Laplace:
            raise NotImplementedError("Laplace approximation not implemented yet.")
        else:
            raise ValueError("No results available.")


def find_model_in_cache(model_cache: Path, model_name: str, model_hash: str) -> Path:
    best_model_filename = None
    for hash_char_count in range(0, len(model_hash)):
        if hash_char_count == 0:
            model_filename = model_cache / f"{model_name}.stan"
        else:
            model_filename = model_cache / f"{model_name} {model_hash[:hash_char_count]}.stan"
        if model_filename.exists():
            # load the model and check its hash if it matches
            with model_filename.open("r") as f:
                model_code = f.read()
            if sha256(model_code.encode()).hexdigest() == model_hash:
                if hash_char_count == len(model_hash) - 1:
                    raise RuntimeError(
                        f"Hash collision for model {model_name} with hash {model_hash} in cache file {model_filename}!")
                return model_filename
        else:
            if best_model_filename is None:
                best_model_filename = model_filename

        hash_char_count += 1

    assert best_model_filename is not None
    return best_model_filename
