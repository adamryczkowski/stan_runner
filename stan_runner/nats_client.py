from __future__ import annotations

import asyncio
import json
import pickle
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from ValueWithError import ValueWithError, ValueWithErrorVec, IValueWithError
from nats.js import JetStreamContext
from nats.js.api import StreamInfo, RawStreamMsg
from overrides import overrides

from .cmdstan_runner import InferenceResult
from .ifaces import StanOutputScope, ILocalInferenceResult, StanErrorType, IStanRunner, StanResultEngine
from .nats_utils import create_stream, name_topic_datadef, name_topic_modeldef, name_topic_run, STREAM_NAME, \
    connect_to_nats
from .utils import infer_param_shapes, normalize_stan_model_by_str


class RemoteStanRunner(IStanRunner):
    _js: JetStreamContext
    _normalize_code: bool

    _stan_model: str | Path | None
    _stan_model_name: str | None

    _data: dict | None

    _output_type: StanOutputScope

    _kwargs: dict

    @staticmethod
    async def Create(server_url: str, user: str, password: str,
                     sig_figs: int = None, output_type: StanOutputScope = StanOutputScope.RawOutput,
                     normalize_code: bool = True) -> RemoteStanRunner:
        ns = await connect_to_nats(server_url, user, password)
        js, _ = await create_stream(ns, permanent_storage=True, stream_name=STREAM_NAME)
        return RemoteStanRunner(server_url, js, sig_figs, output_type, normalize_code)


    def __init__(self, server_url: str, ns: JetStreamContext,
                 sig_figs: int = None, output_type: StanOutputScope = StanOutputScope.RawOutput,
                 normalize_code: bool = True):

        assert isinstance(server_url, str)
        assert isinstance(sig_figs, int) or sig_figs is None
        assert sig_figs is None or sig_figs > 0
        assert isinstance(output_type, StanOutputScope)
        assert isinstance(normalize_code, bool)

        self._normalize_code = normalize_code

        self._sig_figs = sig_figs

        self._stan_model = None
        self._stan_model_name = None
        self._data = None
        self._kwargs = {}

        self._output_type = output_type


    @overrides
    def load_model_by_file(self, stan_file: str | Path, model_name: str | None = None, pars_list: list[str] = None):
        self.load_model_by_str(stan_file.read_text(), model_name, pars_list)

    @overrides
    def load_model_by_str(self, model_code: str | list[str], model_name: str, pars_list: list[str] = None):
        assert isinstance(model_code, str) or isinstance(model_code, list)
        assert isinstance(model_name, str)
        assert isinstance(pars_list, list) or pars_list is None

        self._stan_model = model_code
        self._stan_model_name = model_name
        self.normalize_model_code()

    def normalize_model_code(self):
        if self._normalize_code:
            self._stan_model, msg = normalize_stan_model_by_str(self._stan_model)

            if self._stan_model is None:
                raise ValueError(f"Model normalization failed: {msg}")

            if msg["stanc_output"] != "":
                print(f"STANC output: {msg['stanc_output']}")
            if msg["stanc_warning"] != "":
                print(f"STANC warning: {msg['stanc_warning']}")

    @property
    def is_server_alive(self) -> bool:
        async def check():
            try:
                await self._server_context.stream_info(name=STREAM_NAME)
                return True
            except Exception as e:
                print(f"Stream {STREAM_NAME} not found: {e}")
                return False

        try:
            ans = asyncio.run(check())
        except asyncio.TimeoutError:
            print("Server check timed out")
            return False

        return ans

    @property
    @overrides
    def is_model_loaded(self) -> bool:
        return self._stan_model is not None

    @property
    def is_data_set(self) -> bool:
        return self._data is not None

    @overrides
    def load_data_by_file(self, data_file: str | Path):
        if isinstance(data_file, str):
            data_file = Path(data_file)

        assert isinstance(data_file, Path)
        assert data_file.exists()
        assert data_file.is_file()

        if data_file.suffix == ".json":
            json_data = data_file.read_text()
            self._data = json.loads(json_data)
        elif data_file.suffix == ".pkl":
            import pickle
            with data_file.open("rb") as f:
                self._data = pickle.load(f)
        else:
            raise ValueError("Unknown data file type")

    @property
    @overrides
    def error_state(self) -> StanErrorType:
        return StanErrorType.NO_ERROR

    @property
    @overrides
    def is_model_compiled(self) -> bool:
        return self.is_model_loaded

    def get_messages(self, error_only: bool) -> str:
        return ""

    @overrides
    def load_data_by_dict(self, data: dict[str, float | int | np.ndarray]):
        assert isinstance(data, dict)

        self._data = data

    @property
    def data_hash(self) -> bytes:
        return sha256(json.dumps(self._data).encode()).digest()

    @property
    def model_hash(self) -> bytes:
        return sha256(self._stan_model.encode()).digest()

    def _get_run_hash(self, run_opts: dict) -> bytes:
        assert "model_hash" not in run_opts
        assert "data_hash" not in run_opts
        run_dict = {"model_hash": self.model_hash, "data_hash": self.data_hash, "run_opts": run_opts}
        return sha256(json.dumps(run_dict).encode()).digest()

    @property
    @overrides
    def model_code(self) -> str | None:
        return self._stan_model

    async def _post_data(self):
        """Makes sure that the datadef message is posted"""
        if self._data is None:
            raise ValueError("No data loaded")

        data_hash = self.data_hash

        data_topic = name_topic_datadef(data_hash)

        if not self.is_server_alive:
            raise ValueError("Server not alive")

        msg = await self._server_context.get_last_msg(stream_name=STREAM_NAME, subject=data_topic)

        if msg is None:
            # Serialize data using "pickle"
            serialized_data = pickle.dumps(self._data)
            await self._server_context.publish(subject=data_topic, payload=serialized_data, stream=STREAM_NAME,
                                               headers={"format": "pickle"})

    async def _post_model(self):
        """
        Makes sure that the modeldef message is posted
        """
        if self._stan_model is None:
            raise ValueError("No model loaded")

        model_hash = self.model_hash

        model_topic = name_topic_modeldef(model_hash)

        if not self.is_server_alive:
            raise ValueError("Server not alive")

        msg = await self._server_context.get_last_msg(stream_name=STREAM_NAME, subject=model_topic)

        if msg is None:
            serialized_model = pickle.dumps({"code": self._stan_model, "name": self._stan_model_name})
            await self._server_context.publish(subject=model_topic, payload=serialized_model, stream=STREAM_NAME,
                                               headers={"format": "pickle"})

    def _get_run(self, run_opts: dict, timeout: int | float = 20) -> str:
        if not self.is_model_loaded:
            raise ValueError("No model loaded")

        if not self.is_data_set:
            raise ValueError("No data loaded")

        engine: StanResultEngine = run_opts["engine"]
        args = run_opts["args"]

        run_dict = {"model_hash": self.model_hash, "data_hash": self.data_hash, "run_opts": args, "engine": str(engine)}

        run_hash = self._get_run_hash(run_opts)
        run_topic, run_hash = name_topic_run(run_hash, "rundef")
        assert self.is_server_alive

        async def run():
            msg = await self._server_context.get_last_msg(stream_name=STREAM_NAME, subject=run_topic)
            await self._post_data()
            await self._post_model()

            if msg is None:
                run_payload = pickle.dumps(run_dict)
                await self._server_context.publish(subject=run_topic, payload=run_payload, stream=STREAM_NAME,
                                                   headers={"format": "pickle", "scope": self._output_type.txt_value()})

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.wait_for(run(), timeout=timeout))
        except asyncio.TimeoutError:
            print("Run send timed out")

        async def close():
            loop.stop()

        loop.run_until_complete(close())
        return run_hash

    def _wait_for_result(self, run_hash: str, timeout: int = None) -> tuple[dict, ILocalInferenceResult | None]:
        """
        Waits for the result of the run
        """
        run_topic = name_topic_run(run_hash, "runresult")
        loop = asyncio.new_event_loop()
        msg: RawStreamMsg | None = None
        messages = {}

        async def wait() -> RawStreamMsg:
            while True:
                msg: RawStreamMsg = await self._server_context.get_last_msg(stream_name=STREAM_NAME, subject=run_topic)
                if msg is None:
                    raise ValueError("No message found")

                if msg.headers["status"] == "finished":
                    return msg

        try:
            msg = loop.run_until_complete(asyncio.wait_for(wait(), timeout=timeout))
        except asyncio.TimeoutError:
            messages = {"error": "Waiting for result timed out"}
            result = None

        async def close():
            loop.stop()

        loop.run_until_complete(close())

        if msg is None:
            return messages, None

        if msg.headers["format"] == "pickle":
            output = pickle.loads(msg.data)
        else:
            assert False  # We do not support other formats for now

        output_scope = StanOutputScope.FromStr(msg.headers["output_scope"])
        if output_scope == StanOutputScope.RawOutput:
            result = InferenceResult.DeserializeFromString(raw_str=output)
        elif output_scope == StanOutputScope.FullSamples:
            result = InferenceResultFullSamples(**output)
        elif output_scope == StanOutputScope.Covariances:
            result = InferenceResultCovariances(**output)
        elif output_scope == StanOutputScope.MainEffects:
            result = InferenceResultMainEffects.FromDict(**output)
        else:
            raise ValueError("Unknown output scope")

        return messages, result

    def _serialize_sampling_optons(self, num_chains: int, iter_sampling: int = None, iter_warmup: int = None,
                                   thin: int = 1,
                                   max_treedepth: int = None, seed: int = None,
                                   inits: dict[str, Any] | float | list[str] = None, **kwargs) -> dict:
        ans = {"num_chains": num_chains, "iter_sampling": iter_sampling,
               "iter_warmup": iter_warmup, "thin": thin, "max_treedepth": max_treedepth,
               "seed": seed, "inits": inits, **kwargs}
        return {"engine": StanResultEngine.MCMC, "args": ans}

    @overrides
    def sampling(self, num_chains: int, iter_sampling: int = None,
                 iter_warmup: int = None, thin: int = 1, max_treedepth: int = None,
                 seed: int = None, inits: dict[str, Any] | float | list[str] = None) -> ILocalInferenceResult:
        """
        Sends the request to sample, and waits for the results.
        """
        opts = self._serialize_sampling_optons(num_chains, iter_sampling, iter_warmup, thin, max_treedepth, seed, inits)
        run_hash = self._get_run(opts)

        messages, result = self._wait_for_result(run_hash)
        if result is None:
            raise ValueError(f"Error in sampling: {messages}")

        return result

    def _serialize_variational_bayes_optons(self, output_samples: int, **kwargs) -> dict:
        ans = {"draws": output_samples, **kwargs}
        return {"engine": StanResultEngine.VB, "args": ans}

    @overrides
    def variational_bayes(self, output_samples: int = 1000, **kwargs) -> ILocalInferenceResult:
        opts = self._serialize_variational_bayes_optons(output_samples=output_samples, **kwargs)
        run_hash = self._get_run(opts)

        messages, result = self._wait_for_result(run_hash)
        if result is None:
            raise ValueError(f"Error in sampling: {messages}")

        return result

    def _serialize_pathfinder_optons(self, output_samples: int, **kwargs) -> dict:
        ans = {"draws": output_samples, **kwargs}
        return {"engine": StanResultEngine.PATHFINDER, "args": ans}

    @overrides
    def pathfinder(self, output_samples: int = 1000, **kwargs) -> ILocalInferenceResult:
        opts = self._serialize_pathfinder_optons(output_samples, **kwargs)
        run_hash = self._get_run(opts)

        messages, result = self._wait_for_result(run_hash)
        if result is None:
            raise ValueError(f"Error in sampling: {messages}")

        return result

    def _serialize_laplace_optons(self, output_samples: int, **kwargs) -> dict:
        ans = {"draws": output_samples, **kwargs}
        return {"engine": StanResultEngine.LAPLACE, "args": ans}

    @overrides
    def laplace_sample(self, output_samples: int = 1000, **kwargs) -> ILocalInferenceResult:
        opts = self._serialize_laplace_optons(output_samples, **kwargs)
        run_hash = self._get_run(opts)

        messages, result = self._wait_for_result(run_hash)
        if result is None:
            raise ValueError(f"Error in sampling: {messages}")

        return result


class DelayedInferenceResult(ILocalInferenceResult):
    _messageID: str
    _hatchet: Hatchet

    def __init__(self, hatchet: "Hatchet", messageID: str):

        self._messageID = messageID
        self._hatchet = hatchet

    def wait(self, timeout: int = None) -> dict[str, dict[str, ILocalInferenceResult]] | None:
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
            if event.type == "STEP_RUN_EVENT_TYPE_STARTED":
                print(f"Task {self._messageID} started")
            elif payload is None:
                print(f"Step run probably failed: {event.type}")
                if event.type == StepRunEventType.STEP_RUN_EVENT_TYPE_FAILED:
                    return None

            if payload is not None:
                assert isinstance(payload, dict)
                event_payload = payload
                if "error" in event_payload:
                    print(f"Worker error: {event_payload['error']}")
                    print()
                    print(f"Call stack: {event_payload['call_stack']}")
                    break
                result = event_payload["result"]
                for alg_key, payload in result.items():
                    result_type_str = alg_key.split("_")[0]
                    result_type = StanResultEngine.FromStr(result_type_str)
                    assert isinstance(payload, dict)
                    assert len(payload) == 1
                    for key, arg_dict in payload.items():
                        print(f"Got payload {key} {alg_key}")
                        arg_dict["runtime"] = timedelta(seconds=arg_dict["runtime"]) if "runtime" in arg_dict else None
                        arg_dict["result_type"] = result_type
                        if key == "raw":
                            obj = InferenceResult.DeserializeFromString(raw_str=arg_dict["raw"])
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


class InferenceResultMainEffects(ILocalInferenceResult):
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
    def serialize(self, output_scope: StanOutputScope) -> bytes:
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
        if user_parameter_names is None:
            user_parameter_names = self.onedim_parameters

        onedim_parameter_names = [name for user_name in user_parameter_names for name in
                                  self.get_onedim_parameter_names(user_name)]
        cov_matrix = np.asarray(
            [[self.get_cov(one_dim_par1, one_dim_par2) for one_dim_par1 in onedim_parameter_names] for
             one_dim_par2 in onedim_parameter_names])
        return cov_matrix, onedim_parameter_names

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
    _idx2name: list[str]

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
        self._covariance = cov
        self._idx2name = names

    @overrides
    def get_cov(self, one_dim_par1: str, one_dim_par2: str) -> float | np.ndarray:
        assert one_dim_par1 in self._vars
        assert one_dim_par2 in self._vars

        return self._covariance[self._idx2name.index(one_dim_par1), self._idx2name.index(one_dim_par2)]

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
        ans = np.asarray([self._vars[name].vector for name in onedim_varnames])
        # ans.dtype.names = onedim_varnames
        return ans

    @overrides
    def get_cov_matrix(self, user_parameter_names: list[str] | str | None = None) -> tuple[np.ndarray, list[str]]:
        if user_parameter_names is None:
            user_parameter_names = self.onedim_parameters

        onedim_parameter_names = [name for user_name in user_parameter_names for name in
                                  self.get_onedim_parameter_names(user_name)]

        cov_matrix = np.asarray(
            [[np.cov(self._vars[name].vector, self._vars[name2].vector)[0, 0] for name in onedim_parameter_names] for
             name2 in
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

    def wait(self, timeout: int = None) -> ILocalInferenceResult | None:
        # TODO
        pass
