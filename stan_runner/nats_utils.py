from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from base64 import b64encode

import nats
from nats.js import JetStreamContext
from nats.js.api import RawStreamMsg
from nats.js.api import StorageType, DiscardPolicy, StreamInfo
from nats.js.errors import NotFoundError

STREAM_NAME = "stan_runner"
WORKER_TIMEOUT_SECONDS = 60


async def connect_to_nats(nats_connection: str, user: str, password: str | None = None) -> nats.NATS:
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


class SerializableObject(ABC):
    @abstractmethod
    def serialize(self) -> bytes:
        pass


class KeepAliver:
    """Class that sends keep-alive messages to the NATS server."""
    _subject: str  # The subject to send the keep-alive messages to
    _js: JetStreamContext  # The JetStream context to use
    _my_id: str | None  # The ID of this worker. Will be checked against the last message to ensure that only one worker is running on this subject.
    _timeout: float
    _content_object: bytes | None
    _serialized_format: str

    @staticmethod
    async def Create(self, js: JetStreamContext, subject: str, timeout: float = 60., unique_id: str = None,
                     serialized_content: bytes = None, serialized_format: str = "json") -> KeepAliver:
        obj = KeepAliver(js, subject, timeout, unique_id, serialized_content, serialized_format)
        await obj.remove_all_past_messages()

    def __init__(self, js: JetStreamContext, subject: str, timeout: float = 60., unique_id: str = None,
                 serialized_content: bytes = None, serialized_format: str = "json"):
        self._js = js
        self._subject = subject
        self._my_id = unique_id
        self._timeout = timeout
        if serialized_content is None:
            serialized_content = b""
        self._content_object = serialized_content
        self._serialized_format = serialized_format

    async def remove_all_past_messages(self):
        # Remove all the messages that are in the subject self._subject
        #
        # Iterate over all the messages in the stream and delete them one-by-one using self._js.delete_msg(STREAM_NAME...)

        while True:
            try:
                last_message: RawStreamMsg = await self._js.get_msg(stream_name=STREAM_NAME,
                                                                    subject=self._subject, next=True)
            except nats.js.errors.NotFoundError:
                break

            await self._js.delete_msg(STREAM_NAME, last_message.seq)

    async def keep_alive(self):
        """Send a keep-alive message to the NATS server."""
        try:
            while True:
                try:
                    last_message: RawStreamMsg | None = await self._js.get_last_msg(stream_name=STREAM_NAME,
                                                                                    subject=self._subject)
                except nats.js.errors.NotFoundError:
                    last_message = None

                if last_message is not None:
                    time_to_last_message = time.time() - float(last_message.headers["timestamp"])
                    if self._my_id is not None and last_message.headers["id"] != self._my_id:
                        if time_to_last_message < self._timeout:
                            raise Exception(f"Id mismatch: {last_message.headers['id']} != {self._my_id}")
                else:
                    time_to_last_message = float("inf")

                if (time_to_wait := max(0., self._timeout - time_to_last_message)) > 0:
                    await asyncio.sleep(time_to_wait)

                headers = {
                    "format": self._serialized_format,
                    "timestamp": str(float(time.time()))}

                if self._my_id is not None:
                    headers["id"] = self._my_id

                await self._js.publish(self._subject, payload=self._content_object, stream=STREAM_NAME, headers=headers)
                if last_message is not None:
                    await self._js.delete_msg(STREAM_NAME, last_message.seq)
        except asyncio.CancelledError:
            try:
                print("Canceling keep-alive message...")
                last_message: RawStreamMsg | None = await self._js.get_last_msg(stream_name=STREAM_NAME,
                                                                                subject=self._subject)
            except nats.js.errors.NotFoundError:
                return
            else:
                await self._js.delete_msg(STREAM_NAME, last_message.seq)
