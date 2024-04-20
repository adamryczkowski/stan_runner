from __future__ import annotations

import asyncio
import json
import pickle
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
import traceback

import numpy as np
from ValueWithError import ValueWithError, ValueWithErrorVec, IValueWithError
from hatchet_sdk import Hatchet
from nats.js import JetStreamContext
from nats.js.api import StreamInfo, RawStreamMsg
from nats import NATS
from nats.js.errors import NotFoundError
from .cmdstan_runner import CmdStanRunner

from overrides import overrides

from .cmdstan_runner import InferenceResult
from .ifaces import StanOutputScope, IInferenceResult, StanErrorType, IStanRunner, StanResultEngine
from .nats_utils import create_stream, name_topic_datadef, name_topic_modeldef, name_topic_run, STREAM_NAME
from .utils import infer_param_shapes, normalize_stan_model_by_str


async def get_model_code(model_hash: str, js: JetStreamContext) -> dict[str,str]:
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

def spawn_worker(nats_connection: str, user: str, password: str):
    nc = NATS()
    js: JetStreamContext = nc.jetstream()
    runner = CmdStanRunner(Path("model_cache"))

    async def main_loop():
        try:
            await nc.connect(nats_connection, user=user, password=password, max_reconnect_attempts=1)

            try:
                stream = await js.stream_info(name=STREAM_NAME)
            except NotFoundError as e:
                raise Exception(f"Stream {STREAM_NAME} not found.")
        except Exception as e:
            # Prints error and the stack trace:
            print(e)
            traceback.print_exc()
            return None

        while True:
            try:
                subscription = await js.subscribe(stream=STREAM_NAME, subject="stan.rundef.>", manual_ack=True,
                                         idle_heartbeat=30., pending_msgs_limit=1)

                while True:
                    msg = await subscription.next_msg()
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
                    model_code_dict = await get_model_code(data["model_hash"], js)
                    runner.load_model_by_str(model_code_dict["code"], model_code_dict["name"], None)

                    # Get the data
                    assert "data_hash" in data
                    data = await get_data(data["data_hash"], js)
                    runner.load_data_by_dict(data)

                    # Run the model
                    assert "run_opts" in data
                    assert "engine" in data
                    run_opts = data["run_opts"]
                    engine = StanResultEngine.FromStr(data["engine"])

                    if engine == StanResultEngine.MCMC:
                        result = runner.sampling(**run_opts)
                    elif engine == StanResultEngine.VB:
                        result = runner.variational_bayes(**run_opts)
                    elif engine == StanResultEngine.LAPLACE:
                        result = runner.laplace_sample(**run_opts)
                    elif engine == StanResultEngine.PATHFINDER:
                        result = runner.pathfinder(**run_opts)
                    else:
                        raise Exception(f"Unknown engine {engine}")

                    # Serialize the result
                    result_payload = result.serialize(output_scope)

                    # Publish the result
                    result_topic, _ = name_topic_run(run_hash, "result")
                    await msg.ack()
                    await js.publish(subject=result_topic, payload=result_payload,
                                     headers={"format": "pickle", "output_scope": output_scope.txt_value()})


            except Exception as e:
                print(e)
                traceback.print_exc()
                break

    # Connect to the NATS server
    nc.connect("nats://localhost:4222")

    # Subscribe to the 'stan_runner' subject
    async def message_handler(msg):
        """
        Handles messages received from the NATS server.
        """
        # Decode the message
        data = json.loads(msg.data.decode())

        # Get the message type
        message_type = data["type"]

        # Get the message data
        message_data = data["data"]

        # Handle the message
        if message_type == "run_stan":
            # Run the Stan model
            run_stan(message_data)

    await nc.subscribe("stan_runner", cb=message_handler)

    # Run the event loop
    try:
        loop = asyncio.get_event_loop()
        loop.run_forever()
    except KeyboardInterrupt:
        loop.close()
