from stan_runner import StanData, StanModel
from pathlib import Path

def data():
    data = StanData(run_folder=Path("data"), data={"arr": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "N": 8})
    print(data)
    print(data.get_metaobject())

def model():
    model = StanModel(model_folder=Path("model"), model_name="model", model_code="""
    data {
        int<lower=0> N;
        array[N] real myarr;
    }
    parameters {
        real<lower=0> mu;
    }
    model {
        myarr ~ normal(mu, 1);
    }""")
    print(model)
    print(model.get_metaobject())

if __name__ == '__main__':
    data()
    model()
    print("Done")