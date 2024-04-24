from __future__ import annotations

import asyncio
import pickle
import signal
import traceback
import uuid
from pathlib import Path

from nats.aio.msg import Msg
from nats.js import JetStreamContext
from nats.js.api import RawStreamMsg
from nats.js.errors import NotFoundError
import nats
import time

from .cmdstan_runner import CmdStanRunner
from .ifaces import StanOutputScope, StanResultEngine
from .nats_utils import name_topic_datadef, name_topic_modeldef, name_topic_run, STREAM_NAME, connect_to_nats, \
    WORKER_TIMEOUT_SECONDS, KeepAliver
from .worker_capacity_info import WorkerCapacityInfo


class NatsWorker:
    _runner: CmdStanRunner
    _js: JetStreamContext
    _uid: str
    _self_capacity: WorkerCapacityInfo

    _output_dir: Path
    _subscriptions: list
    _keep_aliver: KeepAliver
    _keep_aliver_task: asyncio.Task | None

    @staticmethod
    async def Create(server_url: str, user: str, password: str = None, model_cache_dir: Path = None,
                     output_dir: Path = None) -> NatsWorker:
        nc = await connect_to_nats(nats_connection=server_url, user=user, password=password)
        worker = NatsWorker(nc.jetstream(), model_cache_dir, output_dir)
        return worker

    def __init__(self, js: JetStreamContext, model_cache_dir: Path = None, output_dir: Path = None):
        if model_cache_dir is None:
            model_cache_dir = Path("model_cache")
        if output_dir is None:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True, parents=True)

        self._runner = CmdStanRunner(model_cache_dir)
        self._js = js
        self._uid = "worker_" + str(uuid.uuid4())[0:8]
        self._self_capacity = WorkerCapacityInfo.BenchmarkSelf(output_dir)
        self._subscriptions = []
        self._keep_aliver = KeepAliver(self._js, subject=f"stan.worker_advert.{self._uid}",
                                       timeout=WORKER_TIMEOUT_SECONDS, unique_id=self._uid,
                                       serialized_content=self._self_capacity.serialize(), serialized_format="json")
        self._keep_aliver_task = None

    async def handle_task(self, msg: Msg):
        """Broker asks the worker to run a Stan model."""
        if msg.headers["format"] == "pickle":
            data = pickle.loads(msg.data)
        else:
            raise Exception(f"Unknown format {msg.headers['format']}")
        output_scope = StanOutputScope.FromStr(msg.headers["output_scope"])
        _stan, _topic, run_hash = msg.subject.split(".")
        assert _stan == "stan"
        assert _topic == "rundef"

        assert isinstance(data, dict)

        # Get the model code
        assert "model_hash" in data
        model_code_dict = await get_model_code(data["model_hash"], self._js)
        self._runner.load_model_by_str(model_code_dict["code"], model_code_dict["name"], None)

        # Get the data
        assert "data_hash" in data
        data = await get_data(data["data_hash"], self._js)
        self._runner.load_data_by_dict(data)

        # Run the model
        assert "run_opts" in data
        assert "engine" in data
        run_opts = data["run_opts"]
        engine = StanResultEngine.FromStr(data["engine"])

        if engine == StanResultEngine.MCMC:
            result = self._runner.sampling(**run_opts)
        elif engine == StanResultEngine.VB:
            result = self._runner.variational_bayes(**run_opts)
        elif engine == StanResultEngine.LAPLACE:
            result = self._runner.laplace_sample(**run_opts)
        elif engine == StanResultEngine.PATHFINDER:
            result = self._runner.pathfinder(**run_opts)
        else:
            raise Exception(f"Unknown engine {engine}")

        # Serialize the result
        result_payload = result.serialize(output_scope)

        # Publish the result
        result_topic, _ = name_topic_run(run_hash, "runresult")
        await msg.ack()
        await self._js.publish(subject=result_topic, payload=result_payload,
                               headers={"format": "pickle", "output_scope": output_scope.txt_value(),
                                        "status": "success"})  # Three statuses: success, failure, exception. Exceptions will be retried.


    async def the_loop(self):
        # Attach the shutdown coroutine to SIGINT and SIGTERM
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(self.shutdown(loop)))
        loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(self.shutdown(loop)))

        sub1 = await self._js.subscribe(f"stan.work_order.{self._uid}", cb=self.handle_task)

        self._subscriptions = [sub1]

        self._keep_aliver_task = asyncio.create_task(self._keep_aliver.keep_alive())

    async def shutdown(self):
        print("Received exit signal, shutting down...")

        print("Canceling the keep-aliver task...")
        self._keep_aliver_task.cancel()
        try:
            await self._keep_aliver_task
        except asyncio.CancelledError:
            pass

        print("Unsubscribing from events...")
        for s in self._subscriptions:
            await s.unsubscribe()



async def get_model_code(model_hash: str, js: JetStreamContext) -> dict[str, str]:
    """
    Get the Stan model code from the NATS server.
    """
    # Get the model code
    modeldef = await js.get_last_msg(stream_name=STREAM_NAME, subject=name_topic_modeldef(model_hash))
    model_code = modeldef.data
    if modeldef.headers["format"] == "pickle":
        model_code = pickle.loads(model_code)
    else:
        raise Exception(f"Unknown format {modeldef.headers['format']}")
    assert isinstance(model_code, dict)

    return model_code


async def get_data(data_hash: str, js: JetStreamContext) -> dict:
    """
    Get the data from the NATS server.
    """
    # Get the data
    datadef = await js.get_last_msg(stream_name=STREAM_NAME, subject=name_topic_datadef(data_hash))
    data = datadef.data
    if datadef.headers["format"] == "pickle":
        data = pickle.loads(data)
    else:
        raise Exception(f"Unknown format {datadef.headers['format']}")

    return data
