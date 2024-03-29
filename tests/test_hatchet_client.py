import json


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
  par3 = par + 1;
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


if __name__ == '__main__':
    print(dict2json(
        model_pars()))  # expect {"model_code": "data {\n  real arr;\n}\nparameters {\n  real par;\n}\nmodel {\n  arr ~ normal(par, 1);\n}\ngenerated quantities {\n  real par3;\n  par3 = par + 1;\n}\n", "model_name": "test_model", "pars": null, "data": {"arr": 1.0}, "iter_sampling": 1000, "num_chains": 4}
