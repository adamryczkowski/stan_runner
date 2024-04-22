import asyncio
import traceback
from base64 import b64encode

import nats
from nats.js import JetStreamContext
from nats.js.api import StorageType, DiscardPolicy, StreamInfo
from nats.js.errors import NotFoundError

STREAM_NAME = "stan_runner"
WORKER_TIMEOUT_SECONDS = 60


async def connect_to_nats(nats_connection: str, user: str, password: str) -> nats.NATS:
    if password is None:
        nc = await nats.connect(nats_connection, token=user, max_reconnect_attempts=1)
    else:
        nc = await nats.connect(nats_connection, user=user, password=password, max_reconnect_attempts=1)
    return nc


async def create_stream(nc: nats.NATS, permanent_storage: bool, stream_name: str,
                        max_bytes=1024 * 1024 * 1024) -> tuple[JetStreamContext, StreamInfo]:
    js: JetStreamContext = nc.jetstream()
    try:
        stream = await js.stream_info(name=stream_name)
    except NotFoundError:
        print(f"Making a new stream {stream_name}")
        stream = await js.add_stream(name=stream_name,
                                     storage=StorageType.FILE if permanent_storage else StorageType.MEMORY,
                                     max_msgs=100000,  # Max 100k unprocessed test cases.
                                     # max_age=str(int(7 * 24 * 60 * 60 * 1_000_000_000)),  # 7 days
                                     no_ack=False,  # We need acks for the task stream
                                     max_consumers=-1,  # Unlimited number of consumers
                                     max_bytes=max_bytes,
                                     max_msg_size=max_bytes,
                                     discard=DiscardPolicy.OLD,
                                     duplicate_window=0,
                                     description="Task queue for scenarios.",
                                     subjects=[f"stan.>"]
                                     )
    return js, stream


# def create_stream(nats_connection: str, user: str, password: str, permanent_storage: bool, stream_name: str,
#                   max_bytes=1024 * 1024 * 1024) -> tuple[JetStreamContext, StreamInfo]:
#     nc = NATS()
#     # nats_connection = f"0.0.0.0:43579"
#     # user = "local"
#     # password = "zFoHRl3fLLniRt03ynfCfnkim33oTtcw"
#     js: JetStreamContext = nc.jetstream()
#
#     async def create_stream_async() -> StreamInfo | None:
#         try:
#             if password is None:
#                 await nc.connect(nats_connection, token=user, max_reconnect_attempts=1)
#             else:
#                 await nc.connect(nats_connection, user=user, password=password, max_reconnect_attempts=1)
#
#             try:
#                 stream = await js.stream_info(name=stream_name)
#             except NotFoundError as e:
#                 print(f"Making a new stream {stream_name}")
#                 stream = await js.add_stream(name=stream_name,
#                                              storage=StorageType.FILE if permanent_storage else StorageType.MEMORY,
#                                              max_msgs=100000,  # Max 100k unprocessed test cases.
#                                              # max_age=str(int(7 * 24 * 60 * 60 * 1_000_000_000)),  # 7 days
#                                              no_ack=False,  # We need acks for the task stream
#                                              max_consumers=-1,  # Unlimited number of consumers
#                                              max_bytes=max_bytes,
#                                              max_msg_size=max_bytes,
#                                              discard=DiscardPolicy.OLD,
#                                              duplicate_window=0,
#                                              description="Task stream for scenario run. Each message is a distinct scenario to be run.",
#                                              subjects=[f"stan.task.>"]
#                                              )
#         except Exception as e:
#             # Prints error and the stack trace:
#             print(e)
#             traceback.print_exc()
#             return None
#         return stream
#
#     stream = asyncio.run(create_stream_async(), debug=True)
#     # loop = asyncio.new_event_loop()
#     # stream = loop.run_until_complete(create_stream_async())
#
#     # async def close():
#     #     loop.stop()
#     #
#     # loop.run_until_complete(close())
#
#     return js, stream


def name_topic_modeldef(model_hash: bytes | str) -> str:
    if isinstance(model_hash, bytes):
        model_hash = b64encode(model_hash).decode()[0:10]
    return f"stan.modeldef.{model_hash}"


def name_topic_datadef(data_hash: bytes | str) -> str:
    if isinstance(data_hash, bytes):
        data_hash = b64encode(data_hash).decode()[0:10]
    return f"stan.datadef.{data_hash}"


def name_topic_run(run_hash: bytes | str, topic: str) -> tuple[str, str]:
    if isinstance(run_hash, bytes):
        run_hash = b64encode(run_hash).decode()[0:10]
    return f"stan.{topic}.{run_hash}", run_hash
