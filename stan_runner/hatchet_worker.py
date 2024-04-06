import os

os.environ["HATCHET_CLIENT_HOST_PORT"] = "192.168.42.5:7077"
os.environ["HATCHET_CLIENT_TLS_STRATEGY"] = "none"
os.environ[
    "HATCHET_CLIENT_TOKEN"] = "eyJhbGciOiJFUzI1NiIsICJraWQiOiIzV3NlencifQ.eyJhdWQiOiJodHRwOi8vbG9jYWxob3N0OjgwODAiLCAiZXhwIjoxNzE4NjI2MDE5LCAiZ3JwY19icm9hZGNhc3RfYWRkcmVzcyI6ImxvY2FsaG9zdDo3MDc3IiwgImlhdCI6MTcxMDg1MDAxOSwgImlzcyI6Imh0dHA6Ly9sb2NhbGhvc3Q6ODA4MCIsICJzZXJ2ZXJfdXJsIjoiaHR0cDovL2xvY2FsaG9zdDo4MDgwIiwgInN1YiI6IjcwN2QwODU1LTgwYWItNGUxZi1hMTU2LWYxYzQ1NDZjYmY1MiIsICJ0b2tlbl9pZCI6ImJhMjA4NjA3LTk2N2UtNDhiNC1iMjUyLTJkYmM3ZjhmMGNjMiJ9.83DD-usEIfLsJFQ82BHPSwq3Gd0MqZd9BbGfMPdgARjeekaNq5M10uLqBCcHVDbH6_LSulg2GclXrhuJksu6gw"

from hatchet_sdk import Hatchet, Context

from .cmdstan_runner import CmdStanRunner, InferenceResult
from pathlib import Path

hatchet = Hatchet()

@hatchet.workflow(on_events=["simple:create"])
class Hatchet_StanRunner:
    _runner: CmdStanRunner

    def __init__(self):
        # Set model cache to tests/model_cache relative to the repo's root
        model_cache = os.path.join(os.path.dirname(__file__), "model_cache")
        self._runner = CmdStanRunner(Path(model_cache))

    def common(self, context: Context):
        print("Loading model")
        context_inputs = context.workflow_input()
        model_code = context_inputs["model_code"]
        if "model_name" in context_inputs:
            model_name = context_inputs["model_name"]
        else:
            model_name = "test_model"

        if "pars" in context_inputs:
            pars = context_inputs["pars"]
        else:
            pars = None
        self._runner.load_model_by_str(model_code, model_name, pars)

        print("Compiling model")
        self._runner.compile_model()

        print("Loading data")
        context_inputs = context.workflow_input()
        data = context_inputs["data"]
        self._runner.load_data_by_dict(data)

    @hatchet.step("run", timeout="1h")
    def run(self, context: Context):
        self.common(context)
        print("Running sampling")
        context_inputs = context.workflow_input()
        output_type = context_inputs["output_type"]

        samplers = [{"fn": self._runner.sampling, "prefix": "sampling"},
                    {"fn": self._runner.variational_bayes, "prefix": "vb"},
                    {"fn": self._runner.laplace_sample, "prefix": "laplace"},
                    {"fn": self._runner.pathfinder, "prefix": "pathfinder"}]

        output = {}

        for args in samplers:
            fn = args["fn"]
            prefix = args["prefix"]
            args_key = f"{prefix}_args"
            if args_key in context_inputs:
                print(f"Running {prefix}")
                result: InferenceResult = fn(**context_inputs[args_key])
                output[prefix] = result.serialize_to_dict(output_type)

        return {"result": output}
