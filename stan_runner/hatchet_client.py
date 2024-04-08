from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from ValueWithError import ValueWithError
from .cmdstan_runner import InferenceResult

import numpy as np

from .ifaces import StanOutputScope, IInferenceResult


def denumpy_dict(d: dict[str, Any]) -> dict[str, Any]:
    ans = {}
    for key in d:
        if isinstance(d[key], np.ndarray):
            ans[key] = d[key].tolist()
        elif isinstance(d[key], dict):
            ans[key] = denumpy_dict(d[key])
        else:
            ans[key] = d[key]
    return ans


def model2json(stan_code: str, model_name: str, data: dict, output_type: StanOutputScope,
               **kwargs) -> str:
    assert isinstance(stan_code, str)
    assert isinstance(model_name, str)
    assert isinstance(data, dict)
    # assert isinstance(engine, StanResultEngine)
    assert isinstance(output_type, StanOutputScope)

    out = {}
    out["model_code"] = stan_code
    out["model_name"] = model_name
    out["data"] = denumpy_dict(data)
    out["output_type"] = output_type.txt_value()
    out.update(kwargs)

    # Convert out to json
    return str(json.dumps(out))


def post_model_to_server(hatchet: "Hatchet", stan_code: str, model_name: str, data: dict,
                         output_type: StanOutputScope, **kwargs) -> str:
    model_json = model2json(stan_code, model_name, data, output_type, **kwargs)
    messageID = hatchet.client.admin.run_workflow("Hatchet_StanRunner", {"request": model_json})

    return messageID

from hatchet_sdk import Hatchet
class RemoteStanRunner:
    _hatchet: "Hatchet" | None
    _server_url: str

    _stan_model: str | Path | None
    _stan_model_name: str | None

    _data: dict | Path | None

    _output_type: StanOutputScope

    _kwargs: dict

    def __init__(self, server_url: str, sig_figs: int = None, output_type: StanOutputScope = StanOutputScope.RawOutput):
        from hatchet_sdk import Hatchet

        assert isinstance(server_url, str)
        assert isinstance(sig_figs, int) or sig_figs is None
        assert sig_figs is None or sig_figs > 0
        assert server_url.startswith("http://") or server_url.startswith("https://")
        assert isinstance(output_type, StanOutputScope)

        self._server_url = server_url
        self._sig_figs = sig_figs

        self._stan_model = None
        self._stan_model_name = None
        self._data = None
        self._kwargs = {}

        self._output_type = output_type

        try:
            self._hatchet = Hatchet()
        except Exception as e:
            print("Hatchet not available")
            self._hatchet = None

    # @overrides
    def load_model_by_file(self, stan_file: str | Path, model_name: str | None = None, pars_list: list[str] = None):
        assert isinstance(stan_file, str) or isinstance(stan_file, Path)
        assert isinstance(model_name, str) or model_name is None
        assert isinstance(pars_list, list) or pars_list is None

        self._stan_model = stan_file
        self._stan_model_name = model_name

    # @overrides
    def load_model_by_str(self, model_code: str | list[str], model_name: str, pars_list: list[str] = None):
        assert isinstance(model_code, str) or isinstance(model_code, list)
        assert isinstance(model_name, str)
        assert isinstance(pars_list, list) or pars_list is None

        self._stan_model = model_code
        self._stan_model_name = model_name

    @property
    def is_server_alive(self) -> bool:
        return self._hatchet is not None

    @property
    # @overrides
    def is_model_loaded(self) -> bool:
        return self._stan_model is not None

    @property
    def is_data_set(self) -> bool:
        return self._data is not None

    # @overrides
    def load_data_by_file(self, data_file: str | Path):
        assert isinstance(data_file, str) or isinstance(data_file, Path)

        self._data = data_file

    # @overrides
    def load_data_by_dict(self, data: dict[str, float | int | np.ndarray]):
        assert isinstance(data, dict)

        self._data = data

    @property
    # @overrides
    def model_code(self) -> str | None:
        assert self._stan_model is not None
        if isinstance(self._stan_model, str):
            return self._stan_model
        else:
            with open(self._stan_model) as f:
                return f.read()

    # @overrides
    def add_sampling(self, num_chains: int, iter_sampling: int = None, iter_warmup: int = None, thin: int = 1,
                     max_treedepth: int = None, seed: int = None,
                     inits: dict[str, Any] | float | list[str] = None, **kwargs):
        assert self.is_model_loaded
        assert self._data is not None

        if "sampling_args" in self._kwargs:
            raise ValueError("Sampling already scheduled")

        self._kwargs["sampling_args"] = {"num_chains": num_chains, "iter_sampling": iter_sampling,
                                         "iter_warmup": iter_warmup, "thin": thin, "max_treedepth": max_treedepth,
                                         "seed": seed, "inits": inits, **kwargs}

        # message_ID = post_model_to_server(self.model_code, self._stan_model_name, self._data, StanResultEngine.MCMC,
        #                                   self._output_type, num_chains=num_chains, iter_sampling=iter_sampling,
        #                                   iter_warmup=iter_warmup, thin=thin, max_treedepth=max_treedepth, seed=seed,
        #                                   inits=inits)

    # @overrides
    def schedule_variational_bayes(self, output_samples: int = 1000, **kwargs):
        assert self.is_model_loaded
        assert self._data is not None

        if "vb_args" in self._kwargs:
            raise ValueError("Variational Bayes already scheduled")

        self._kwargs["vb_args"] = {"output_samples": output_samples, **kwargs}

        # message_ID = post_model_to_server(self.model_code, self._stan_model_name, self._data, StanResultEngine.VB,
        #                                   self._output_type, output_samples=output_samples, **kwargs)

        # return DelayedInferenceResult(message_ID)

    # @overrides
    def schedule_pathfinder(self, output_samples: int = 1000, **kwargs):
        assert self.is_model_loaded
        assert self._data is not None

        if "pathfinder_args" in self._kwargs:
            raise ValueError("Pathfinder already scheduled")

        self._kwargs["pathfinder_args"] = {"output_samples": output_samples, **kwargs}

        # message_ID = post_model_to_server(self.model_code, self._stan_model_name, self._data,
        #                                   StanResultEngine.PATHFINDER,
        #                                   self._output_type, output_samples=output_samples, **kwargs)

        # return DelayedInferenceResult(message_ID)

    # @overrides
    def schedule_laplace_sample(self, output_samples: int = 1000, **kwargs):
        assert self.is_model_loaded
        assert self._data is not None

        if "laplace_args" in self._kwargs:
            raise ValueError("Laplace already scheduled")

        self._kwargs["laplace_args"] = {"output_samples": output_samples, **kwargs}

        # message_ID = post_model_to_server(self.model_code, self._stan_model_name, self._data, StanResultEngine.LAPLACE,
        #                                   self._output_type, output_samples=output_samples, **kwargs)
        #
        # return DelayedInferenceResult(message_ID)

    def run(self) -> DelayedInferenceResult:
        message_ID = post_model_to_server(self._hatchet, self.model_code, self._stan_model_name, self._data,
                                          self._output_type, **self._kwargs)

        return DelayedInferenceResult(self._hatchet, message_ID)


class DelayedInferenceResult:
    _messageID: str
    _hatchet: Hatchet

    def __init__(self, hatchet: "Hatchet", messageID: str):

        self._messageID = messageID
        self._hatchet = hatchet

    def wait(self, timeout: int = None) -> dict[str, IInferenceResult] | None:
        from hatchet_sdk import StepRunEventType
        from hatchet_sdk.client import ClientImpl
        event_payload = None
        print(f"Waiting for result: {self._messageID}")

        # def on_event(event):
        #     nonlocal event_payload
        #     event_payload = event.payload

        # Hatchet_StanRunner-fcc542/run
        event:StepRunEventType

        client:ClientImpl = self._hatchet.client

        from hatchet_sdk.clients.rest.models.step_run import StepRun
        # sr:StepRun = client.rest_client.step_run_get(self._messageID)
        # sr = StepRun(1,2,3)


        from hatchet_sdk import WorkflowRun
        # workflow_run:WorkflowRun = self._hatchet.client.rest_client.workflow_run_get(self._messageID)


        # workflow_run.status
        # self._hatchet.client.rest().workflow_list()
        ans = {}

        for event in self._hatchet.client.listener.stream(self._messageID):
            print(event)
            payload = event.payload
            if payload is not None:
                event_payload = payload
                result = event_payload["result"]
                for key, payload in result.items():
                    assert isinstance(payload, dict)
                    assert len(payload) == 1
                    if "raw" in payload:
                        obj = InferenceResult.DeserializeFromString(payload["raw"])
                    elif "draws" in payload:
                        obj = InferenceResultFullSamples(payload["draws"], payload["runtime"])
                    elif "covariances" in payload:
                        obj = InferenceResultCovariances(payload["covariances"], payload["runtime"])
                    elif "main_effects" in payload:
                        obj = InferenceResultMainEffects(payload["main_effects"], payload["runtime"])
                    else:
                        raise ValueError("Unknown result type")
                    ans[key] = obj



        print(f"Found result: {ans}")

    @property
    def hatchet(self) -> "Hatchet":
        return self._hatchet

    def cancel(self):
        self._hatchet.client.listener.stream.abort(self._messageID)

    @property
    def is_computed(self) -> bool:
        import hatchet_sdk.clients.listener

        def get_metadata(token: str):
            return [('authorization', 'bearer ' + token)]

        try:
            listener = self._hatchet.client.listener.client.SubscribeToWorkflowEvents(
                hatchet_sdk.clients.listener.SubscribeToWorkflowEventsRequest(
                    workflowRunId=self._messageID,
                ), metadata=get_metadata(self._hatchet.client.listener.token))
        except Exception as e:
            return False

        return True

    # def get_result(self) -> IInferenceResult:


class InferenceResultMainEffects(IInferenceResult):
    _vars: dict[str, ValueWithError]
    _user2onedim: dict[str, list[str]] | None  # Translates user parameter names to one-dim parameter names

    def __init__(self, vars: dict[str, ValueWithError], user2onedim: dict[str, list[str]] | None = None):
        self._vars = vars
        self._user2onedim = user2onedim

    pass


class InferenceResultCovariances(IInferenceResult):
    _vars: dict[str, ValueWithError]
    _covariance: np.ndarray
    pass


class InferenceResultFullSamples(IInferenceResult):
    pass


class RemoteInferenceResultPromise:
    _messageID: str

    def __init__(self, messageID: str):
        self._messageID = messageID

    @property
    def is_computed(self) -> bool:
        # TODO

        return False

    def wait(self, timeout: int = None) -> IInferenceResult | None:
        # TODO
        pass
