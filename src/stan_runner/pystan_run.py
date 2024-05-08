import io
import subprocess
import time
from contextlib import redirect_stdout, redirect_stderr
from multiprocessing import cpu_count
from pathlib import Path
from subprocess import run
from typing import Any

import cmdstanpy

from .ifaces import StanResultEngine, StanOutputScope
from .ifaces2 import IStanRun, IStanModel, IStanData, IStanResultBase
from .stan_result_scopes import StanResultRawResult


# from .result_adapter import InferenceResult

def install_all_dependencies(cpu_cores: int = 1):
    # Check if `stanc` is installed in the correct path:
    stanc_expected_path = Path(cmdstanpy.cmdstan_path()) / "bin" / "stanc"
    if stanc_expected_path.exists():
        # Run stanc and get its version
        output = subprocess.run([str(stanc_expected_path), "--version"], capture_output=True)
        if output.returncode == 0:
            # Check if stdoutput starts with "stanc3"
            if output.stdout.decode().strip().startswith("stanc3"):
                return

    cmdstanpy.install_cmdstan(verbose=True, overwrite=False, cores=cpu_cores)


def compile_model(model: IStanModel) -> tuple[float, Path | None, dict, cmdstanpy.CmdStanModel | None]:
    exe_file = model.model_filename.with_suffix("")
    stdout = io.StringIO()
    stderr = io.StringIO()
    force_recompile = False
    if not exe_file.exists() or exe_file.stat().st_size == 0:
        force_recompile = True
        exe_file.touch()

    time_now = time.time()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            model = cmdstanpy.CmdStanModel(stan_file=model.model_filename, exe_file=str(exe_file),
                                           force_compile=force_recompile, stanc_options=model.stanc_options,
                                           cpp_options=model.compilation_options)
            model.compile(force=force_recompile)
        except subprocess.CalledProcessError as e:
            time_taken = time.time() - time_now
            err = {"output": stdout.getvalue() + e.stdout,
                   "compile_warning": stderr.getvalue(),
                   "compile_error": e.stderr}
            return time_taken, None, err, None

    time_taken = time.time() - time_now
    # Write the time taken to compile the model into the file {model_filename}.time
    with open(exe_file.with_suffix(".time"), "w") as f:
        f.write(str(time_taken))

    return time_taken, exe_file, {"output": stdout.getvalue(), "compile_warning": stderr.getvalue()}, model


def run(obj: IStanRun) -> IStanResultBase:
    model = obj.get_model_meta()
    assert isinstance(model, IStanModel)
    model.make_sure_is_compiled()

    data = obj.get_data_meta()
    assert isinstance(data, IStanData)

    stdout = io.StringIO()
    stderr = io.StringIO()

    pystan_model = model.make_sure_is_compiled()

    time_now = time.time()

    engine: Any
    if obj.run_engine == StanResultEngine.PATHFINDER:
        engine = pystan_model.pathfinder
        obj.run_opts["draws"] = obj.run_opts["sample_count"]
        del obj.run_opts["sample_count"]
    elif obj.run_engine == StanResultEngine.LAPLACE:
        engine = pystan_model.laplace_sample
        obj.run_opts["draws"] = obj.run_opts["sample_count"]
        del obj.run_opts["sample_count"]
    elif obj.run_engine == StanResultEngine.VB:
        engine = pystan_model.variational
        obj.run_opts["draws"] = obj.run_opts["sample_count"]
        del obj.run_opts["sample_count"]
    elif obj.run_engine == StanResultEngine.MCMC:
        threads_per_chain = 1
        if "chains" in obj.run_opts:
            num_chains = obj.run_opts["chains"]
        else:
            num_chains = 1
        number_of_cores = cpu_count()
        parallel_chains = min(num_chains, number_of_cores)
        threads_per_chain = 1
        if "STAN_THREADS" in model.compilation_options and model.compilation_options["STAN_THREADS"] == True:
            if number_of_cores > num_chains:
                threads_per_chain = number_of_cores // num_chains

        obj.run_opts["threads_per_chain"] = threads_per_chain
        obj.run_opts["chains"] = parallel_chains
        obj.run_opts["iter_sampling"] = obj.run_opts["sample_count"] // parallel_chains
        del obj.run_opts["sample_count"]


        engine = pystan_model.sample
    else:
        raise ValueError(f"Unknown engine {obj.run_engine}")

    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            ans = engine(data=data.data_json_file,
                         output_dir=obj.run_folder,
                         **obj.run_opts)
        except subprocess.CalledProcessError as e:
            messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
            return StanResultRawResult(run=obj, output=stdout.getvalue(), warnings="", errors=stderr.getvalue(),
                                       runtime=time.time() - time_now, result=None)

    time_taken = time.time() - time_now

    messages = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
    out = StanResultRawResult(run=obj, output=stdout.getvalue(), warnings="", errors=stderr.getvalue(),
                              runtime=time_taken, result=ans)

    if obj.output_scope == StanOutputScope.MainEffects:
        return out.downcast_to_main_effects()
    elif obj.output_scope == StanOutputScope.Covariances:
        return out.downcast_to_covariances()
    elif obj.output_scope == StanOutputScope.FullSamples:
        return out.downcast_to_full_samples()
    else:
        assert obj.output_scope == StanOutputScope.RawOutput
        return out
