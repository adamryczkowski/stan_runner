from pathlib import Path
from stan_runner import *

import numpy as np

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


def model_generator_cov(par_count: int = 2, nrow: int = 100) -> tuple[StanModel, StanData]:
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
    model_str = "\n".join(model_code)

    model_name = f"test_cov_{str(par_count)}-{str(nrow)}"
    model_obj = StanModel(model_folder=Path(__file__).parent.parent / "model_cache",
                          model_name=model_name, model_code=model_str)

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

    data_obj = StanData(run_folder=Path(__file__).parent.parent / "data_cache", data=data_dict)

    return model_obj, data_obj


def test1(output_scope: StanOutputScope, single_engine:StanResultEngine = None):
    model_obj, data_obj = model_generator_cov(par_count=2, nrow=1000)
    # model_cache_dir = Path(__file__).parent / "model_cache"

    if single_engine is None:
        single_engine = {StanResultEngine.MCMC, StanResultEngine.PATHFINDER, StanResultEngine.LAPLACE, StanResultEngine.VB}
    elif isinstance(single_engine, StanResultEngine):
        single_engine = {single_engine}

    install_all_dependencies()
    try:
        model_obj.make_sure_is_compiled()
    except ValueError as v:
        print(f"Compilation error: {str(v)}")
        return
    assert model_obj.is_compiled

    # runner.sampling(16000, num_chains=16)
    print(data_obj.data_dict["true_mu"])
    print(data_obj.data_dict["true_sigma"])

    if StanResultEngine.LAPLACE in single_engine:
        runner_map = StanRun(run_folder=Path(__file__).parent.parent / "run_cache",
                             data=data_obj, model=model_obj, output_scope=output_scope, sample_count=1000,
                             run_engine=StanResultEngine.LAPLACE)
        map = runner_map.run()
        if map.is_error:
            print(map.errors)
        print(map)
        if output_scope > StanOutputScope.MainEffects:
            assert isinstance(map, IStanResultCovariances)
            print(map.pretty_cov_matrix())

    if StanResultEngine.PATHFINDER in single_engine:
        runner_pf = StanRun(run_folder=Path(__file__).parent.parent / "run_cache",
                            data=data_obj, model=model_obj, output_scope=output_scope, sample_count=1000,
                            run_engine=StanResultEngine.PATHFINDER)
        pf = runner_pf.run()
        if pf.is_error:
            print(pf.errors)
        print(pf)
        if output_scope > StanOutputScope.MainEffects:
            assert isinstance(pf, IStanResultCovariances)
            print(pf.pretty_cov_matrix())

    if StanResultEngine.VB in single_engine:
        runner_vb = StanRun(run_folder=Path(__file__).parent.parent / "run_cache",
                            data=data_obj, model=model_obj, output_scope=output_scope, sample_count=1000,
                            run_engine=StanResultEngine.VB, run_opts={"grad_samples": 20})
        vb = runner_vb.run()
        if vb.is_error:
            print(vb.errors)
        print(vb)
        if output_scope > StanOutputScope.MainEffects:
            assert isinstance(vb, IStanResultCovariances)
            print(vb.pretty_cov_matrix())

        runner_vb = StanRun(run_folder=Path(__file__).parent.parent / "run_cache",
                            data=data_obj, model=model_obj, output_scope=output_scope,
                            run_engine=StanResultEngine.VB, sample_count=1000,
                            run_opts={"grad_samples": 20, "algorithm": "fullrank"})
        vb = runner_vb.run()
        if vb.is_error:
            print(vb.errors)
        print(vb)
        if output_scope > StanOutputScope.MainEffects:
            assert isinstance(vb, IStanResultCovariances)
            print(vb.pretty_cov_matrix())

    if StanResultEngine.MCMC in single_engine:
        runner_mcmc = StanRun(run_folder=Path(__file__).parent.parent / "run_cache",
                              data=data_obj, model=model_obj, output_scope=output_scope,
                              run_engine=StanResultEngine.MCMC, sample_count=1000)
        mcmc = runner_mcmc.run()
        if mcmc.is_error:
            print(mcmc.errors)
        print(mcmc)
        if output_scope > StanOutputScope.MainEffects:
            assert isinstance(mcmc, IStanResultCovariances)
            print(mcmc.pretty_cov_matrix())


if __name__ == '__main__':
    test1(StanOutputScope.MainEffects)
    # test1(StanOutputScope.Covariances, StanResultEngine.MCMC)
    test1(StanOutputScope.Covariances)
    test1(StanOutputScope.FullSamples)
    test1(StanOutputScope.RawOutput)
