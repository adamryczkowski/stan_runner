import hashlib
import io
import subprocess
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from subprocess import run
from typing import Any, Optional

import cmdstanpy
import numpy as np
from overrides import overrides

from .ifaces import IStanResult, StanErrorType
from .utils import find_model_in_cache

stan_CI_levels_dict = {0: 0.95, 1: 0.9, 2: 0.8, 3: 0.5}


def initialize(cpu_cores: int = 1):
    # Check if `stanc` is installed in the correct path:
    stanc_expected_path = Path(cmdstanpy.cmdstan_path()) / "bin" / "stanc"
    if stanc_expected_path.exists():
        # Run stanc and get its version
        output = run([str(stanc_expected_path), "--version"], capture_output=True)
        if output.returncode == 0:
            # Check if stdoutput starts with "stanc3"
            if output.stdout.decode().strip().startswith("stanc3"):
                return

    cmdstanpy.install_cmdstan(verbose=True, overwrite=False, cores=cpu_cores)


class CmdStanRunner(IStanResult):
    _number_of_cores: int
    _stanc_opts: dict[str, Any]
    _cpp_opts: dict[str, Any]
    _other_opts: dict[str, Any]
    _model_cache: Path
    _output_dir: Path
    _initialized: bool

    _stan_model: cmdstanpy.CmdStanModel | None | str
    _model_filename: Path | None

    _data: dict[str, float | int | np.ndarray] | None | Path  # Either dict or JSON file

    _last_model_hash: str
    _pars_of_interest: list[str] | None  # None is auto

    _messages: dict[str, str]

    def __init__(self, model_cache: Path, number_of_cores: int = None, allow_optimizations_for_stanc: bool = True,
                 stan_threads: bool = True, output_dir: Path = None, sig_figs: int = None):
        if isinstance(model_cache, str):
            model_cache = Path(model_cache)
        assert isinstance(model_cache, Path)
        if not model_cache.exists():
            model_cache.mkdir(parents=True)
        assert model_cache.is_dir()
        if number_of_cores is not None:
            assert isinstance(number_of_cores, int)
            assert number_of_cores > 0

        if output_dir is not None:
            assert isinstance(output_dir, Path)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            assert output_dir.is_dir()
        else:
            output_dir = tempfile.TemporaryDirectory()

        self._output_dir = output_dir

        self._number_of_cores = number_of_cores
        self._model_cache = model_cache

        self.clear()

        self._initialized = False

        if allow_optimizations_for_stanc:
            self._stanc_opts = {"O": True}
        self._cpp_opts = {"stan_threads": stan_threads}
        self._pars_of_interest = None

        self._stan_model = None
        self._model_filename = None
        self._other_opts = {}
        if sig_figs is not None:
            assert isinstance(sig_figs, int)
            self._other_opts["sig_figs"] = sig_figs

    @overrides
    def install_dependencies(self):
        if self._initialized:
            return

        initialize(self._number_of_cores)
        self._initialized = True

    @overrides
    def clear(self):
        self._last_model_hash = ""
        self._messages = {}
        self._pars_of_interest = None

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

    def clear_last_data(self):
        self._data = None

    @property
    @overrides
    def error_state(self) -> StanErrorType:
        if "stanc_error" in self._messages:
            return StanErrorType.SYNTAX_ERROR
        if "compile_error" in self._messages:
            return StanErrorType.COMPILE_ERROR
        return StanErrorType.NO_ERROR

    @property
    @overrides
    def messages(self) -> dict[str, str]:
        return self._messages

    @property
    @overrides
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    @overrides
    def is_model_loaded(self) -> bool:
        return self._model_filename is not None

    @property
    @overrides
    def is_error(self) -> bool:
        return "stanc_error" in self._messages or "compile_error" in self._messages

    @property
    @overrides
    def is_model_compiled(self) -> bool:
        return isinstance(self._stan_model, cmdstanpy.CmdStanModel)

    @property
    @overrides
    def is_data_set(self) -> bool:
        return self._data is not None

    @overrides
    def get_messages(self, error_only: bool) -> str:
        output = []
        items = ["stanc_output", "stanc_warnings", "stanc_error",
                 "compile_output", "compile_warnings", "compile_error"]

        for item in items:
            if item in self._messages:
                if error_only and not item.endswith("error"):
                    continue
                output.append(self._messages[item])
        return "\n".join(output)

    def load_model_by_file(self, stan_file: str | Path, model_name: str | None = None, pars_list: list[str] = None):
        if not self.is_initialized:
            self.install_dependencies()
        if isinstance(stan_file, str):
            stan_file = Path(stan_file)
        assert isinstance(stan_file, Path)
        assert stan_file.exists()
        assert stan_file.is_file()
        if pars_list is not None:
            assert isinstance(pars_list, list)
            assert all(isinstance(p, str) for p in pars_list)

        stdout = io.StringIO()
        stderr = io.StringIO()

        # Copy stan_file to temporary location
        tmpfile = tempfile.NamedTemporaryFile("w", delete=False)
        tmpfile.write(stan_file.read_bytes().decode())
        tmpfile.close()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                cmdstanpy.format_stan_file(tmpfile.name, overwrite_file=True,
                                           canonicalize=["deprecations", "parentheses", "braces", "includes",
                                                         "strip-comments"])
            except subprocess.CalledProcessError as e:
                self._messages["stanc_output"] = stdout.getvalue() + e.stdout
                self._messages["stanc_warning"] = stderr.getvalue()
                self._messages["stanc_error"] = e.stderr
                return

        self._messages["stanc_output"] = stdout.getvalue()
        self._messages["stanc_warning"] = stderr.getvalue()

        with open(tmpfile.name, 'rb', buffering=0) as f:
            model_hash = hashlib.file_digest(f, 'sha256').hexdigest()

        if model_hash == self._last_model_hash:
            return

        model_filename = find_model_in_cache(self._model_cache, model_name, model_hash)
        if not model_filename.exists():
            model_filename.write_bytes(Path(tmpfile.name).read_bytes())

        self._pars_of_interest = pars_list

        self._model_filename = model_filename
        self._last_model_hash = model_hash

    def load_model_by_str(self, model_code: str | list[str], model_name: str, pars_list: list[str] = None):
        if not self.is_initialized:
            self.install_dependencies()

        if isinstance(model_code, list):
            model_code = "\n".join(model_code)

        # Write the model to a disposable temporary location
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(model_code)
            model_filename = Path(f.name)

        self.load_model_by_file(model_filename, model_name, pars_list)
        self._stan_model = model_name

    def compile_model(self, force_recompile: bool = False):
        assert self.is_model_loaded
        if self.is_model_compiled:
            return

        assert isinstance(self._stan_model, str)

        exe_file = self._model_filename.with_suffix("")
        # hash_file = self._model_filename.with_suffix(".hash")

        stdout = io.StringIO()
        stderr = io.StringIO()

        # Recompile also, if size of the file is zero.
        if not exe_file.exists() or exe_file.stat().st_size == 0:
            force_recompile = True
            exe_file.touch()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                model = cmdstanpy.CmdStanModel(stan_file=self._model_filename, exe_file=str(exe_file),
                                               force_compile=force_recompile, stanc_options=self._stanc_opts,
                                               cpp_options=self._cpp_opts)
                model.compile(force=force_recompile)
            except subprocess.CalledProcessError as e:
                self._messages["compile_output"] = stdout.getvalue() + e.stdout
                self._messages["compile_warning"] = stderr.getvalue()
                self._messages["compile_error"] = e.stderr
                return
        self._messages["compile_output"] = stdout.getvalue()
        self._messages["compile_warning"] = stderr.getvalue()
        self._stan_model = model

    def load_data_by_file(self, data_file: str | Path):
        """Loads data from a structureal file. Right now, the supported format is only JSON."""
        if isinstance(data_file, str):
            data_file = Path(data_file)
        assert isinstance(data_file, Path)
        assert data_file.exists()
        assert data_file.is_file()
        self._data = data_file

    def load_data_by_dict(self, data: dict[str, float | int | np.ndarray]):
        assert isinstance(data, dict)
        self._data = data

    def sampling(self, num_chains: int, iter_sampling: int = None,
                 iter_warmup: int = None, thin: int = 1, max_treedepth: int = None,
                 seed: int = None, inits: dict[str, Any] | float | list[str] = None) -> tuple[
        Optional[cmdstanpy.CmdStanMCMC], dict[str, str]]:

        assert self.is_model_compiled

        threads_per_chain = 1
        if "STAN_THREADS" in self._cpp_opts:
            if self._cpp_opts["STAN_THREADS"] == "true":
                if self._number_of_cores > num_chains:
                    threads_per_chain = self._number_of_cores // num_chains

        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._stan_model.sample(data=self._data, chains=num_chains, parallel_chains=1,
                                              threads_per_chain=threads_per_chain, seed=seed,
                                              inits=inits, iter_warmup=iter_warmup,
                                              iter_sampling=iter_sampling, thin=thin,
                                              max_treedepth=max_treedepth, output_dir=self._output_dir.name,
                                              sig_figs=self._other_opts.get("sig_figs", None))
            except subprocess.CalledProcessError as e:
                messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
                return None, messages

        messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
        return ans, messages

    def variational_bayes(self, output_samples: int = 1000, **kwargs) -> tuple[Optional[cmdstanpy.CmdStanVB], dict[str, str]]:

        assert self.is_model_compiled

        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._stan_model.variational(data=self._data, output_dir=self._output_dir.name,
                                                   sig_figs=self._other_opts.get("sig_figs", None),
                                                   draws=output_samples,
                                                   **kwargs)
            except subprocess.CalledProcessError as e:
                messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
                return None, messages
        return ans, {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}

    def pathfinder(self,
                   output_samples: int = 1000, **kwargs) -> tuple[Optional[cmdstanpy.CmdStanPathfinder], dict[str, str]]:
        assert self.is_model_compiled

        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._stan_model.pathfinder(data=self._data, draws=output_samples, output_dir=self._output_dir.name,
                                                  sig_figs=self._other_opts.get("sig_figs", None),
                                                  **kwargs)
            except subprocess.CalledProcessError as e:
                messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
                return None, messages

        return ans, {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}

    def laplace_sample(self, output_samples: int = 1000, **kwargs) -> tuple[Optional[cmdstanpy.CmdStanLaplace], dict[str, str]]:
        assert self.is_model_compiled

        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._stan_model.laplace_sample(data=self._data, output_dir=self._output_dir.name,
                                                      sig_figs=self._other_opts.get("sig_figs", None),
                                                      draws=output_samples,
                                                      **kwargs)
            except subprocess.CalledProcessError as e:
                messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
                return None, messages
        return ans, {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}

    def optimize(self, **kwargs) -> tuple[Optional[cmdstanpy.CmdStanMLE], dict[str, str]]:
        assert self.is_model_compiled

        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                ans = self._stan_model.optimize(data=self._data, output_dir=self._output_dir.name,
                                                sig_figs=self._other_opts.get("sig_figs", None),
                                                **kwargs)
            except subprocess.CalledProcessError as e:
                messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
                return None, messages
        return ans, {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}

    @property
    def model_code(self) -> str | None:
        if self._model_filename is not None:
            with self._model_filename.open("r") as f:
                return f.read()
        return None
