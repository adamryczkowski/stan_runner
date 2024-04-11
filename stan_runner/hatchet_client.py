from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, List

import numpy as np
from ValueWithError import ValueWithError, ValueWithErrorVec, IValueWithError
from overrides import overrides

from .ifaces import StanResultEngine
from .cmdstan_runner import InferenceResult
from .ifaces import StanOutputScope, IInferenceResult
from .utils import infer_param_shapes


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
    def schedule_sampling(self, num_chains: int, iter_sampling: int = None, iter_warmup: int = None, thin: int = 1,
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

    def wait(self, timeout: int = None) -> dict[str, dict[str, IInferenceResult]] | None:
        if timeout is not None:
            raise NotImplementedError
        from hatchet_sdk import StepRunEventType
        from hatchet_sdk.client import ClientImpl
        from hatchet_sdk.clients.listener import StepRunEvent
        print(f"Waiting for result: {self._messageID}")

        # def on_event(event):
        #     nonlocal event_payload
        #     event_payload = event.payload

        # Hatchet_StanRunner-fcc542/run

        client: ClientImpl = self._hatchet.client

        # sr:StepRun = client.rest_client.step_run_get(self._messageID)
        # sr = StepRun(1,2,3)

        # workflow_run:WorkflowRun = self._hatchet.client.rest_client.workflow_run_get(self._messageID)

        # workflow_run.status
        # self._hatchet.client.rest().workflow_list()
        ans = {}
        event: StepRunEvent

        for event in self._hatchet.client.listener.stream(self._messageID):
            # print(event)
            payload = event.payload
            # event.payload
            if event.type == "STEP_RUN_EVENT_TYPE_FAILED":
                raise ValueError("Failed")

            if payload is not None:
                assert isinstance(payload, dict)
                event_payload = payload
                result = event_payload["result"]
                for alg_key, payload in result.items():
                    result_type_str = alg_key.split("_")[0]
                    result_type = StanResultEngine.from_str(result_type_str)
                    assert isinstance(payload, dict)
                    assert len(payload) == 1
                    for key, arg_dict in payload.items():
                        arg_dict["runtime"] = timedelta(seconds=arg_dict["runtime"]) if "runtime" in arg_dict else None
                        arg_dict["result_type"] = result_type
                        if key == "raw":
                            obj = InferenceResult.DeserializeFromString(**arg_dict)
                        elif key == "draws":
                            obj = InferenceResultFullSamples(**arg_dict)
                        elif key == "covariances":
                            obj = InferenceResultCovariances(**arg_dict)
                        elif key == "main_effects":
                            obj = InferenceResultMainEffects.FromDict(**arg_dict)
                        else:
                            raise ValueError("Unknown result type")
                        if key in ans:
                            ans[key][alg_key] = obj
                        else:
                            ans[key] = {alg_key: obj}

        # print(f"Found result: {ans}")
        return ans

    @property
    def hatchet(self) -> "Hatchet":
        return self._hatchet

    def cancel(self):
        self._hatchet.client.listener.stream.abort(self._messageID)

    @property
    def is_computed(self) -> bool:
        assert NotImplementedError
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
    _vars: dict[str, IValueWithError]
    _result_type: StanResultEngine
    _param_shapes: dict[str, tuple[int, ...]]
    _sample_count: int
    _method_name: str

    @staticmethod
    def FromDict(vars: dict[str, dict[str, float | int]],
                 messages: dict[str, str], runtime: timedelta | None,
                 result_type: StanResultEngine, sample_count: int,
                 method_name: str) -> InferenceResultMainEffects:
        param_shapes, user2onedim = infer_param_shapes(list(vars.keys()))
        err_vars = {key: ValueWithError(**value) for key, value in vars.items()}

        obj = InferenceResultMainEffects(vars=err_vars, user2onedim=user2onedim, param_shapes=param_shapes,
                                         messages=messages, runtime=runtime, result_type=result_type,
                                         sample_count=sample_count, method_name=method_name)
        return obj

    def __init__(self, vars: dict[str, IValueWithError],
                 user2onedim: dict[str, list[str]], param_shapes: dict[str, tuple[int, ...]],
                 messages: dict[str, str], runtime: timedelta | None, result_type: StanResultEngine, sample_count: int,
                 method_name: str):
        super().__init__(messages=messages, runtime=runtime)
        assert isinstance(vars, dict)
        assert all(isinstance(vars[key], IValueWithError) for key in vars)
        assert isinstance(user2onedim, dict) or user2onedim is None
        assert all(value in vars for values in user2onedim.values() for value in values)
        assert isinstance(result_type, StanResultEngine)
        assert isinstance(param_shapes, dict)
        assert isinstance(sample_count, int)
        assert isinstance(method_name, str)

        self._vars = vars
        self._user2onedim = user2onedim
        self._param_shapes = param_shapes
        self._result_type = result_type
        self._validate_parameter_shapes(param_shapes)
        self._sample_count = sample_count
        self._method_name = method_name

    def _validate_parameter_shapes(self, param_shapes: dict[str, tuple[int, ...]]):
        for key, shape in param_shapes.items():
            assert key in self._vars
            count = np.prod(shape)
            assert count == len(self.get_onedim_parameter_names(key))

    @property
    @overrides
    def result_type(self) -> StanResultEngine:
        return self._result_type

    @property
    @overrides
    def result_scope(self) -> StanOutputScope:
        return StanOutputScope.MainEffects

    @overrides
    def serialize_to_file(self, output_type: str, file_name: str):
        raise NotImplementedError

    @overrides
    def get_parameter_shape(self, user_parameter_name: str) -> tuple[int, ...]:
        assert user_parameter_name in self._param_shapes
        return self._param_shapes[user_parameter_name]

    @overrides
    def sample_count(self, onedim_parameter_name: str = None) -> float | int | None:
        if onedim_parameter_name is not None:
            return self._vars[onedim_parameter_name].N
        else:
            return self._sample_count

    @property
    def draws(self, incl_raw: bool = True) -> np.ndarray:
        # Does not make sense for main effects
        raise NotImplementedError

    @overrides
    def get_parameter_estimate(self, onedim_parameter_name: str, store_values: bool = False) -> IValueWithError:
        if store_values:
            raise NotImplementedError
        return self._vars[onedim_parameter_name]

    @overrides
    def get_parameter_mu(self, user_parameter_name: str) -> np.ndarray:
        onedim_parameter_names = self.get_onedim_parameter_names(user_parameter_name)
        return np.array([self._vars[key].estimateMean() for key in onedim_parameter_names])

    @overrides
    def get_parameter_sigma(self, user_parameter_name: str) -> np.ndarray:
        onedim_parameter_names = self.get_onedim_parameter_names(user_parameter_name)
        return np.array([self._vars[key].estimateSE() for key in onedim_parameter_names])

    @overrides
    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError

    @property
    @overrides
    def method_name(self) -> str:
        return self._method_name

    @overrides
    def all_main_effects(self) -> dict[str, IValueWithError]:
        return self._vars

    @overrides
    def get_cov(self, one_dim_par1: str, one_dim_par2: str) -> float | np.ndarray:
        raise NotImplementedError


class InferenceResultCovariances(InferenceResultMainEffects):
    _covariance: np.ndarray

    def __init__(self, names: list[str], cov: list[list[float]], means: dict[str, float], N: list[float | int],
                 runtime: timedelta | None, messages: dict[str, str], result_type: StanResultEngine,
                 method_name: str, sample_count: int = 0):
        cov = np.asarray(cov)
        N = np.asarray(N)
        assert isinstance(cov, np.ndarray)
        assert isinstance(N, np.ndarray)
        assert cov.shape[0] == cov.shape[1]
        assert cov.shape[0] == len(names)
        assert len(means) == len(names)
        assert N.shape[0] == len(names)

        super_vars = {name: ValueWithError(value=means[name], SE=SE, N=int(N)) for name, SE, N in
                      zip(names, np.sqrt(np.diag(cov)), N)}

        param_shapes, user2onedim = infer_param_shapes(names)

        super().__init__(vars=super_vars, user2onedim=user2onedim, param_shapes=param_shapes,
                         messages=messages, runtime=runtime, result_type=result_type,
                         method_name=method_name, sample_count=sample_count)

    @overrides
    def get_cov(self, one_dim_par1: str, one_dim_par2: str) -> float | np.ndarray:
        assert one_dim_par1 in self._vars
        assert one_dim_par2 in self._vars

        return self._covariance[one_dim_par1, one_dim_par2]

    @property
    @overrides
    def result_scope(self) -> StanOutputScope:
        return StanOutputScope.Covariances


class InferenceResultFullSamples(InferenceResultMainEffects):
    _vars: dict[str, ValueWithErrorVec]

    def __init__(self, draws: list[list[float | int]], names: list[str],
                 messages: dict[str, str], runtime: timedelta | None, result_type: StanResultEngine,
                 method_name: str):
        draws = np.asarray(draws)
        assert isinstance(names, list)
        assert draws.shape[1] == len(names)
        assert len(draws) > 0
        sample_len = draws.shape[0]
        assert isinstance(all(len(draw) == sample_len for draw in draws), bool)

        param_shapes, user2onedim = infer_param_shapes(names)
        vars = {name: ValueWithErrorVec(draws[:, i]) for i, name in enumerate(names)}

        super().__init__(vars=vars, user2onedim=user2onedim, param_shapes=param_shapes,
                         messages=messages, runtime=runtime, result_type=result_type,
                         sample_count=sample_len, method_name=method_name)

    @property
    @overrides
    def result_scope(self) -> StanOutputScope:
        return StanOutputScope.FullSamples

    @overrides
    def draws(self, incl_raw: bool = True) -> np.ndarray:
        onedim_varnames = self.onedim_parameters
        ans = np.array([self._vars[name].vector for name in onedim_varnames])
        ans.dtype.names = onedim_varnames
        return ans

    @overrides
    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        if user_parameter_names is None:
            user_parameter_names = self.onedim_parameters

        onedim_parameter_names = [name for user_name in user_parameter_names for name in
                                  self.get_onedim_parameter_names(user_name)]
        cov_matrix = np.array(
            [[np.cov(self._vars[name].vector, self._vars[name2].vector) for name in onedim_parameter_names] for name2 in
             onedim_parameter_names])
        return cov_matrix, onedim_parameter_names


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
