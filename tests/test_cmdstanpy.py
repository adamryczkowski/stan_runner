import os
from cmdstanpy import CmdStanModel


class MyCmdStanModel(CmdStanModel):
    pass

def test():
    my_stanfile = os.path.join('.', 'my_model.stan')
    my_model = MyCmdStanModel(stan_file="/home/Adama-docs/Adam/MyDocs/praca/TrainerEngine/lib/stan_runner/tests/model_cache/cov_model.stan", cpp_options={'STAN_THREADS':'true'})
