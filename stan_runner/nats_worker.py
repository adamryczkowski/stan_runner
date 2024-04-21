from __future__ import annotations

import asyncio
import pickle
import signal
import traceback
import uuid
from pathlib import Path

from nats.aio.msg import Msg
from nats.js import JetStreamContext

from .cmdstan_runner import CmdStanRunner
from .ifaces import StanOutputScope, StanResultEngine
from .nats_utils import name_topic_datadef, name_topic_modeldef, name_topic_run, STREAM_NAME, connect_to_nats
from .worker_capacity_info import WorkerCapacityInfo

class NatsWorker:
    _runner: CmdStanRunner
    _js: JetStreamContext
    _uid: str
    _self_capacity: WorkerCapacityInfo

    def __init__(self, nats_connection: str, user: str, password: str):
        self._runner = CmdStanRunner(Path("model_cache"))
        nats = connect_to_nats(nats_connection, user, password)
        self._js = nats.jetstream()
        self._uid = "worker_" + str(uuid.uuid4())[0:8]
        self._self_capacity = WorkerCapacityInfo.BenchmarkSelf(self._runner.model_code)

    async def handle_task(self, msg: Msg):
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

    async def wait_for_tasks(self):
        while True:
            try:
                subscription = await self._js.subscribe(stream=STREAM_NAME, subject="stan.rundef.>", manual_ack=True,
                                                        idle_heartbeat=30., pending_msgs_limit=1)

                while True:
                    msg = await subscription.next_msg()
                    await self.handle_task(msg)


            except Exception as e:
                print(e)
                traceback.print_exc()
                break

    async def send_heartbeat(self):
        while True:
            await self._js.publish(f"stan.worker_heartbeat.{self._uid}", b"1")
            await asyncio.sleep(30)

    def main_loop(self):
        loop = asyncio.new_event_loop()



        async def shutdown(signal, loop):
            print(f"Received exit signal {signal.name}...")
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

            [task.cancel() for task in tasks]

            print(f"Cancelling {len(tasks)} tasks")
            await asyncio.gather(*tasks, return_exceptions=True)
            loop.stop()

        # Attach the shutdown coroutine to SIGINT

        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown(signal.SIGINT, loop)))

        loop.create_task(self.wait_for_tasks())
        loop.create_task(self.send_heartbeat())
        try:
            loop.run_forever()
        finally:
            loop.close()
            print("Successfully shutdown the loop")


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

