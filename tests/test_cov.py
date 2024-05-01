from pathlib import Path

import numpy as np

from src.stan_runner import CmdStanRunner

from scipy.stats import random_correlation


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
   rows ~ normal(mu1+mu2, sigma);
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


def test1():
    model_str, data_dict = model_generator_cov(nrow=10000)
    model_cache_dir = Path(__file__).parent / "model_cache"

    runner = CmdStanRunner(model_cache_dir)
    runner.load_model_by_str(model_str, "cov_model")
    assert runner.is_model_loaded
    runner.compile_model()
    assert runner.is_model_compiled
    runner.load_data_by_dict(data_dict)
    assert runner.is_data_set
    # runner.sampling(16000, num_chains=16)
    print(data_dict["true_mu"])
    print(data_dict["true_sigma"])

    mcmc = runner.sampling(iter_sampling=4000, num_chains=8)
    if mcmc.is_error:
        print(mcmc.messages)
    else:
        print(mcmc.repr_without_sampling_errors())
        print(mcmc.pretty_cov_matrix(["mu1", "mu2"]))

    vb = runner.variational_bayes(algorithm="fullrank")
    if vb.is_error:
        print(vb.messages)
    else:
        print(vb.repr_without_sampling_errors())
        print(vb.pretty_cov_matrix(["mu1", "mu2"]))

    map = runner.laplace_sample()
    if map.is_error:
        print(map.messages)
    else:
        print(map.repr_without_sampling_errors())
        print(map.pretty_cov_matrix(["mu1", "mu2"]))

    pf = runner.pathfinder(output_samples=4000)
    if pf.is_error:
        print(pf.messages)
    else:
        print(pf.repr_without_sampling_errors())
        print(pf.pretty_cov_matrix(["mu1", "mu2"]))


if __name__ == '__main__':
    test1()
