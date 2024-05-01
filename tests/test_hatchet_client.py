# Tests full serialization of the cmdStanPy model object

import pickle
import shutil
from pathlib import Path
import json

import numpy as np
from cmdstanpy.cmdstan_args import CmdStanArgs
from cmdstanpy.stanfit.vb import CmdStanVB
from cmdstanpy.stanfit.vb import RunSet

from src.stan_runner import *

register_server()


def model_pars() -> dict:
    model_pars = {
        "model_code": """data {
  int N;
  vector[N] arr;
}
parameters {
  real par;
}
model {
  arr ~ normal(par, 1);
}
generated quantities {
  real par3;
  par3 = par + 2;
}
""",
        "model_name": "test_model",
        "pars": None,
        "data": {
            "arr": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "N": 8
        },
        "sampling_args": {
            "iter_sampling": 1000,
            "num_chains": 4},
        "vb_args": {},
        "laplace_args": {},
        "pathfinder_args": {},
        "output_type": "main_effects"
    }
    return model_pars


def dict2json(d: dict) -> str:
    return json.dumps(d)


def model_generator_cov(nrow: int = 1000) -> tuple[str, dict[str, np.ndarray]]:
    # Returns stan model that recovers multivariate normal variable with par_count size from
    # gets input data of dimension data_dim from nrow samples, and returns model string and the data dictionary.

    model_str = """
data {
   int<lower=1> N;
   vector[N] rows;
}
parameters {
   real mu1;
   real mu2;
   real<lower=0> sigma;
}
model {
   mu1 ~ normal(0, 1);
   mu2 ~ normal(0, 1);
   sigma ~ cauchy(0, 1);
   rows ~ normal(mu1+mu2+1, sigma);
}
"""

    true_mu = np.random.randn(1)[0]
    # Exp(1) distributed
    true_sigma = np.random.exponential(1)

    data = np.random.normal(loc=true_mu, scale=true_sigma, size=nrow)

    data_dict = {
        "N": nrow,
        "rows": data,
        "true_mu": true_mu,
        "true_sigma": true_sigma
    }

    return model_str, data_dict


def make_model() -> CmdStanRunner:
    model_str, data_dict = model_generator_cov(nrow=10000)
    model_cache_dir = Path(__file__).parent / "model_cache"

    runner = CmdStanRunner(model_cache_dir)
    runner.load_model_by_str(model_str, "cov_model")
    assert runner.is_model_loaded
    runner.compile_model()
    assert runner.is_model_compiled
    runner.load_data_by_dict(data_dict)
    assert runner.is_data_set
    return runner


def make_remote_model(output_type: StanOutputScope) -> RemoteStanRunner:
    model_str, data_dict = model_generator_cov(nrow=10000)
    runner = RemoteStanRunner("http://localhost:37184", output_type=output_type)
    runner.load_model_by_str(model_str, "cov_model")
    assert runner.is_model_loaded
    runner.load_data_by_dict(data_dict)
    assert runner.is_data_set
    return runner


def test1():
    runner = make_model()
    # runner.sampling(16000, num_chains=16)
    bayes = runner.pathfinder()
    print(bayes)
    obj: CmdStanVB = bayes._result
    print(obj)
    rs: RunSet = obj._runset

    a: CmdStanArgs = rs._args
    output_dir = Path("/") / a.output_dir
    pickle.dump(rs, open(output_dir / "runset.pkl", "wb"))
    print(rs._args.method.name)  # 'SAMPLE', 'VARIATIONAL', 'LAPLACE', PATHFINDER

    dest_dir = Path("/home/Adama-docs/Adam/MyDocs/praca/TrainerEngine/tmp")
    # Copy output_dir to dest_dir
    shutil.rmtree(dest_dir)
    shutil.copytree(output_dir, dest_dir)
    # Delete the output_dir
    shutil.rmtree(output_dir)

    rs2: RunSet = pickle.load(open(dest_dir / "runset.pkl", "rb"))
    rs2._args.output_dir = str(dest_dir)
    rs2._csv_files = [str(dest_dir / Path(item).name) for item in rs2.csv_files]
    rs2._stdout_files = [str(dest_dir / Path(item).name) for item in rs2.stdout_files]

    # rs2:RunSet = RunSet(obj)

    vb2 = CmdStanVB(rs2)
    bayes2 = InferenceResult(vb2, messages={})
    print(bayes2)


def test_remote_run(output_type: StanOutputScope):
    runner = make_remote_model(output_type)
    runner.schedule_laplace_sample()
    runner.schedule_pathfinder()
    runner.schedule_variational_bayes(algorithm="fullrank")
    runner.schedule_sampling(4, 500)

    delayed_result = runner.run()
    ans = delayed_result.wait()
    flattened_ans = [(f"{key1} {key2}", item) for key2, items in ans.items() for key1, item in items.items()]
    print(ans)
    if output_type != StanOutputScope.MainEffects:
        for key, item in flattened_ans:
            print(f"{key}: {item.pretty_cov_matrix()}")


def test2():
    print("Testing MainEffects output...")
    test_remote_run(StanOutputScope.MainEffects)

    print("Testing Covariance output...")
    test_remote_run(StanOutputScope.Covariances)

    print("Testing Full output...")
    test_remote_run(StanOutputScope.FullSamples)

    print("Testing Raw output...")
    test_remote_run(StanOutputScope.RawOutput)

def test3(output_type: StanOutputScope):
    runner = make_remote_model(output_type)
    runner.schedule_laplace_sample()
    # runner.schedule_pathfinder()
    # runner.schedule_variational_bayes(algorithm="fullrank")
    # runner.schedule_sampling(4, 500)

    delayed_result = runner.run()
    ans = delayed_result.wait()
    flattened_ans = [(f"{key1} {key2}", item) for key2, items in ans.items() for key1, item in items.items()]
    print(ans)
    if output_type != StanOutputScope.MainEffects:
        for key, item in flattened_ans:
            print(f"{key}: {item.pretty_cov_matrix()}")


if __name__ == '__main__':
    test2()
    # test3(StanOutputScope.FullSamples)
