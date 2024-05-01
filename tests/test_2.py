from pathlib import Path

import numpy as np

from src.stan_runner import CmdStanRunner

from scipy.stats import random_correlation


def random_corr_matrix(n: int) -> np.ndarray:
    # Returns a random covariance matrix of size n. Eigenvalues are exponentially distributed with mean 1.
    eigenvalues = np.random.exponential(scale=1, size=n)
    eigenvalues /= (np.sum(eigenvalues) / n)
    return random_correlation(eigs=eigenvalues)


def random_variance(n: int, mu: float) -> np.ndarray:
    # Returns a random variance of size n, that is exponentially distributed with mean mu
    return np.random.exponential(scale=mu, size=n)


def random_cov_matrix(n: int, mu: float) -> np.ndarray:
    # Returns a random covariance matrix of size n. Eigenvalues are exponentially distributed with mean mu.
    stdev = np.sqrt(random_variance(n, mu))
    corr = random_corr_matrix(n).rvs()
    return np.diag(stdev) @ corr @ np.diag(stdev)


def model_generator_cov(par_count: int = 2, nrow: int = 100) -> tuple[str, dict[str, np.ndarray]]:
    # Returns stan model that recovers multivariate normal variable with par_count size from
    # gets input data of dimension data_dim from nrow samples, and returns model string and the data dictionary.

    model_code = ["data {", ]
    model_code.append(f"   int<lower=1> nrow;")
    model_code.append(f"   int<lower=1> par_count;")
    model_code.append(f"   array[nrow] vector[par_count] rows;")
    model_code.append("}")
    model_code.append("parameters {")
    model_code.append(f"   vector[par_count] mu;")
    model_code.append(f"   cov_matrix[par_count] sigma;")
    model_code.append("}")
    model_code.append("model {")
    model_code.append("   mu ~ normal(0, 1);")
    # model_code.append("   sigma ~ lkj_corr(1);")
    model_code.append("   for (i in 1:nrow) {")
    model_code.append("      rows[i] ~ multi_normal(mu, sigma);")
    model_code.append("   }")
    model_code.append("}")

    true_mu = np.random.randn(par_count)
    true_sigma = random_cov_matrix(par_count, 1)

    data = np.random.multivariate_normal(true_mu, true_sigma, nrow)

    data_dict = {
        "nrow": nrow,
        "par_count": par_count,
        "rows": data,
        "true_mu": true_mu,
        "true_sigma": true_sigma
    }

    model_str = "\n".join(model_code)

    return model_str, data_dict


def test1():
    model_str, data_dict = model_generator_cov(par_count=2, nrow=10000)
    model_cache_dir = Path(__file__).parent / "model_cache"

    runner = CmdStanRunner(model_cache_dir)
    runner.load_model_by_str(model_str, "test2_model")
    assert runner.is_model_loaded
    runner.compile_model()
    assert runner.is_model_compiled
    runner.load_data_by_dict(data_dict)
    assert runner.is_data_set
    # runner.sampling(16000, num_chains=16)
    print(data_dict["true_mu"])
    print(data_dict["true_sigma"])


    map = runner.laplace_sample()
    if map is None:
        print(map.messages["stderr"])
    print(map)
    print(map.get_cov_matrix())

    pf = runner.pathfinder(output_samples=4000)
    if pf is None:
        print(pf.messages["stderr"])
    print(pf)
    print(pf.get_cov_matrix())

    vb = runner.variational_bayes(grad_samples=20)
    if vb is None:
        print(vb.messages["stderr"])
    print(vb)
    print(vb.get_cov_matrix())

    mcmc = runner.sampling(iter_sampling=4000, num_chains=8)
    if mcmc is None:
        print(mcmc.messages["stderr"])
    print(mcmc)
    print(mcmc.get_cov_matrix())

if __name__ == '__main__':
    test1()
