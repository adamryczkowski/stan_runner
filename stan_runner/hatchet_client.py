from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hatchet_sdk.clients.listener
import numpy as np
from hatchet_sdk import Hatchet
from overrides import overrides

from .ifaces import StanOutputScope, StanResultEngine, IStanRunner, StanErrorType, IInferenceResult

hatchet = Hatchet()


def model2json(stan_code: str, model_name: str, data: dict, engine: StanResultEngine, output_type: StanOutputScope,
               **kwargs) -> str:
    assert isinstance(stan_code, str)
    assert isinstance(model_name, str)
    assert isinstance(data, dict)
    assert isinstance(engine, StanResultEngine)
    assert isinstance(output_type, StanOutputScope)

    out = {}
    out["model_code"] = stan_code
    out["model_name"] = model_name
    out["data"] = data
    out["output_type"] = output_type.value
    out.update(kwargs)

    # Convert out to json
    return str(json.dumps(out))


def post_model_to_server(stan_code: str, model_name: str, data: dict, engine: StanResultEngine,
                         output_type: StanOutputScope, **kwargs) -> str:
    model_json = model2json(stan_code, model_name, data, engine, output_type, **kwargs)
    messageID = hatchet.client.admin.run_workflow("Hatchet_StanRunner", {"request": model_json})

    return messageID


class RemoteStanRunner:
    _hatchet: Hatchet | None
    _server_url: str

    _stan_model: str | Path | None
    _stan_model_name: str | None

    _data: dict | Path | None

    _output_type: StanOutputScope

    def __init__(self, server_url: str, sig_figs: int = None, output_type: StanOutputScope = StanOutputScope.RawOutput):
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

    # @overrides
    def load_data_by_file(self, data_file: str | Path):
        assert isinstance(data_file, str) or isinstance(data_file, Path)

        self._data = data_file

    # @overrides
    def load_data_by_dict(self, data: dict[str, float | int | np.ndarray]):
        assert isinstance(data, dict)

        self._data = data

    @property
    @overrides
    def model_code(self) -> str | None:
        assert self._stan_model is not None
        if isinstance(self._stan_model, str):
            return self._stan_model
        else:
            with open(self._stan_model) as f:
                return f.read()

    @overrides
    def sampling(self, num_chains: int, iter_sampling: int = None, iter_warmup: int = None, thin: int = 1,
                 max_treedepth: int = None, seed: int = None,
                 inits: dict[str, Any] | float | list[str] = None) -> DelayedInferenceResult:
        assert self.is_model_loaded
        assert self._data is not None

        message_ID = post_model_to_server(self.model_code, self._stan_model_name, self._data, StanResultEngine.MCMC,
                                          self._output_type, num_chains=num_chains, iter_sampling=iter_sampling,
                                          iter_warmup=iter_warmup, thin=thin, max_treedepth=max_treedepth, seed=seed,
                                          inits=inits)

        return DelayedInferenceResult(message_ID)

    @overrides
    def variational_bayes(self, output_samples: int = 1000, **kwargs) -> DelayedInferenceResult:
        assert self.is_model_loaded
        assert self._data is not None

        message_ID = post_model_to_server(self.model_code, self._stan_model_name, self._data, StanResultEngine.VB,
                                          self._output_type, output_samples=output_samples, **kwargs)

        return DelayedInferenceResult(message_ID)

    @overrides
    def pathfinder(self, output_samples: int = 1000, **kwargs) -> DelayedInferenceResult:
        assert self.is_model_loaded
        assert self._data is not None

        message_ID = post_model_to_server(self.model_code, self._stan_model_name, self._data,
                                          StanResultEngine.PATHFINDER,
                                          self._output_type, output_samples=output_samples, **kwargs)

        return DelayedInferenceResult(message_ID)

    @overrides
    def laplace_sample(self, output_samples: int = 1000, **kwargs) -> DelayedInferenceResult:
        assert self.is_model_loaded
        assert self._data is not None

        message_ID = post_model_to_server(self.model_code, self._stan_model_name, self._data, StanResultEngine.LAPLACE,
                                          self._output_type, output_samples=output_samples, **kwargs)

        return DelayedInferenceResult(message_ID)


class DelayedInferenceResult:
    _messageID: str

    def __init__(self, messageID: str):
        self._messageID = messageID

    def wait(self, timeout: int = None) -> IInferenceResult | None:
        event_payload = None
        print(f"Waiting for result: {self._messageID}")

        def on_event(event):
            nonlocal event_payload
            event_payload = event.payload

        hatchet.client.listener.on(self._messageID, on_event)

        print(f"Found result: {event_payload}")

    def cancel(self):
        hatchet.client.listener.stream.abort(self._messageID)

    @property
    def is_computed(self) -> bool:
        def get_metadata(token: str):
            return [('authorization', 'bearer ' + token)]

        try:
            listener = hatchet.client.listener.client.SubscribeToWorkflowEvents(
                        hatchet_sdk.clients.listener.SubscribeToWorkflowEventsRequest(
                            workflowRunId=self._messageID,
                        ), metadata=get_metadata(hatchet.client.listener.token))
        except Exception as e:
            return False

        return True


class InferenceResultMainEffects(IInferenceResult):
    pass


class InferenceResultCovariances(IInferenceResult):
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
