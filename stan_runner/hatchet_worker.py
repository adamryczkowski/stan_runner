import json
import os
from pathlib import Path

from .cmdstan_runner import CmdStanRunner, InferenceResult
from .hatchet_utils import register_server


def register_worker(server_address: str = "192.168.42.5:7077", server_token: str = None,
                    worker_name: str = "CmdStanWorker") -> "Worker":
    print("Loading Hatched sdk...")
    from hatchet_sdk import Hatchet, Context
    print("Registering worker...")

    register_server(server_address, server_token)
    hatchet = Hatchet()

    @hatchet.workflow(on_events=["simple:create"], schedule_timeout="1h")
    class Hatchet_StanRunner:
        _runner: CmdStanRunner

        def __init__(self):
            # Set model cache to tests/model_cache relative to the repo's root
            model_cache = os.path.join(os.path.dirname(__file__), "model_cache")
            self._runner = CmdStanRunner(Path(model_cache))
            self._runner.install_dependencies()

        def common(self, context: Context):
            print("Loading model")
            context_inputs = context.workflow_input()
            data_str = context_inputs["request"]
            if isinstance(data_str, str):
                data_dict = json.loads(data_str)
            else:
                data_dict = data_str
            assert isinstance(data_dict, dict)
            model_code = data_dict["model_code"]
            if "model_name" in data_dict:
                model_name = data_dict["model_name"]
            else:
                model_name = "test_model"

            if "pars" in data_dict:
                pars = data_dict["pars"]
            else:
                pars = None
            self._runner.load_model_by_str(model_code, model_name, pars)

            print("Compiling model")
            self._runner.compile_model()

            print("Loading data")
            data = data_dict["data"]
            self._runner.load_data_by_dict(data)

        @hatchet.step("run", timeout="1h")
        def run(self, context: Context):
            self.common(context)
            print("Running sampling")
            context_inputs = context.workflow_input()
            data_str = context_inputs["request"]
            if isinstance(data_str, str):
                data_dict = json.loads(data_str)
            else:
                data_dict = data_str
            assert isinstance(data_dict, dict)
            output_type = data_dict["output_type"]

            samplers = [{"fn": self._runner.sampling, "prefix": "sampling"},
                        {"fn": self._runner.variational_bayes, "prefix": "vb"},
                        {"fn": self._runner.laplace_sample, "prefix": "laplace"},
                        {"fn": self._runner.pathfinder, "prefix": "pathfinder"}]

            output = {}

            for args in samplers:
                fn = args["fn"]
                prefix = args["prefix"]
                args_key = f"{prefix}_args"
                if args_key in data_dict:
                    print(f"Running {prefix}")
                    result: InferenceResult = fn(**data_dict[args_key])
                    if result is not None:
                        output[prefix] = result.serialize_to_dict(output_type)
                    else:
                        print(f"Result unavailable: {data_dict[args_key]}")

            return {"result": output}

    worker = hatchet.worker(worker_name)
    worker.register_workflow(Hatchet_StanRunner())

    return worker


def run_worker(server_address: str = "192.168.42.5:7077", server_token: str = None):
    from hatchet_sdk import Worker
    worker:Worker = register_worker(server_address, server_token)
    worker.contexts
    print("Starting worker...")
    try:
        worker.start()
    except Exception as e:
        print(e)
    print("Worker quit")
