from pathlib import Path
import json
import numpy as np

from src.stan_runner import CmdStanRunner
import cmdstanpy

def model()->str:
    return """data {
  real arr;
}
parameters {
  real par;
}
model {
  arr ~ normal(par, 1);
}
generated quantities {
  real par3;
  par3 = par + 1;
}
"""


def test1():
    model_cache_dir = Path(__file__).parent / "model_cache"

    runner = CmdStanRunner(model_cache=model_cache_dir)
    runner.install_dependencies()

    model_code = model()
    data = {
        "arr": 1.0
    }

    runner.load_model_by_str(model_code, "test_model")
    if not runner.is_model_loaded:
        print(runner.model_code)
        print(runner.messages["stanc_error"])

        return
    runner.compile_model()
    if not runner.is_model_compiled:
        print(runner.messages["compile_error"])
        return
    runner.load_data_by_dict(data)
    if not runner.is_data_set:
        print(runner.messages["data_error"])
        return
    result = runner.sampling(iter_sampling=1000, num_chains=8)
    # print(messages)
    print(result)
    json.dump(result, open("result.json", "w"))



if __name__ == '__main__':
    test1()
