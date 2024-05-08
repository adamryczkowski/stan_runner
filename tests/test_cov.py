from pathlib import Path
from typing import Any

import numpy as np

from stan_runner import *


def model_generator_cov(nrow: int = 1000) -> tuple[StanModel, StanData]:
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
   rows ~ normal(mu1+mu2, sigma);
}
"""
    model_obj = StanModel(model_folder=Path(__file__).parent / "model_cache", model_name="cov_model",
                          model_code=model_str)

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
    data_obj = StanData(run_folder=Path(__file__).parent / "data_cache", data=data_dict)

    return model_obj, data_obj


def test1(output_scope: StanOutputScope):
    model_obj, data_obj = model_generator_cov(nrow=10000)

    install_all_dependencies()
    # runner = CmdStanRunner(model_cache=model_cache_dir)
    # runner.install_dependencies()
    try:
        model_obj.make_sure_is_compiled()
    except ValueError as v:
        print(f"Compilation error: {str(v)}")
        return

    assert model_obj.is_compiled

    print(data_obj.data_dict["true_mu"])
    print(data_obj.data_dict["true_sigma"])

    def make_a_run(output_scope: StanOutputScope, run_engine: StanResultEngine, run_opts: dict[str, Any] = None):
        runner = StanRun(run_folder=Path(__file__).parent.parent / "run_cache",
                         data=data_obj, model=model_obj, output_scope=output_scope,
                         sample_count=4000, run_engine=run_engine, run_opts=run_opts)

        result = runner.run()
        if result.is_error:
            print(result.errors)
        print(result.repr_without_sampling_errors())
        if output_scope > StanOutputScope.MainEffects:
            assert isinstance(result, IStanResultCovariances)

            print(result.pretty_cov_matrix(["mu1", "mu2"]))

    make_a_run(output_scope=output_scope, run_engine=StanResultEngine.MCMC)
    make_a_run(output_scope=output_scope, run_engine=StanResultEngine.VB, run_opts={"algorithm": "fullrank"})
    make_a_run(output_scope=output_scope, run_engine=StanResultEngine.VB, run_opts={"algorithm": "meanfield"})
    make_a_run(output_scope=output_scope, run_engine=StanResultEngine.LAPLACE)
    make_a_run(output_scope=output_scope, run_engine=StanResultEngine.PATHFINDER)


if __name__ == '__main__':
    test1(output_scope=StanOutputScope.RawOutput)
    test1(output_scope=StanOutputScope.FullSamples)
    test1(output_scope=StanOutputScope.Covariances)
    test1(output_scope=StanOutputScope.MainEffects)
