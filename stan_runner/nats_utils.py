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
from .nats_ifaces import NetworkDuplicateError

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
    _last_msg_seq: int | None

    @staticmethod
    async def Create(js: JetStreamContext, subject: str, timeout: float = 60., unique_id: str = None,
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
        self._last_msg_seq = None

    async def remove_all_past_messages(self):
        await clear_subject(self._js, STREAM_NAME, self._subject)

    async def shutdown(self):
        if self._last_msg_seq is not None:
            await self._js.delete_msg(STREAM_NAME, self._last_msg_seq)

    async def check_for_network_duplicates(self):
        """Check if there are any other workers on the network with the same ID"""
        try:
            last_message: RawStreamMsg | None = await self._js.get_last_msg(stream_name=STREAM_NAME,
                                                                            subject=self._subject)
        except nats.js.errors.NotFoundError:
            return

        if last_message is not None:
            time_to_last_message = time.time() - float(last_message.headers["timestamp"])
            if self._my_id is not None and last_message.headers["id"] != self._my_id:
                if time_to_last_message < self._timeout:
                    raise NetworkDuplicateError(last_message.headers["id"],
                                                f"Id mismatch: {last_message.headers['id']} != {self._my_id}")

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
                    self._last_msg_seq = last_message.seq
                else:
                    time_to_last_message = float("inf")

                if (time_to_wait := max(0., self._timeout - time_to_last_message)) > 0:
                    await asyncio.sleep(time_to_wait)

                headers = {
                    "format": self._serialized_format,
                    "timestamp": str(float(time.time()))}

                if self._my_id is not None:
                    headers["id"] = self._my_id

                ack = await self._js.publish(self._subject, payload=self._content_object, stream=STREAM_NAME,
                                             headers=headers)
                if self._last_msg_seq is not None:
                    await self._js.delete_msg(STREAM_NAME, self._last_msg_seq)
                    self._last_msg_seq = ack.seq
        except asyncio.CancelledError:
            print("Canceling keep-alive message...")
            await self.shutdown()


async def clear_subject(js: JetStreamContext, stream_name: str, subject: str, msg_id: str = None):
    seq = 0
    while True:
        try:
            if msg_id is None:
                last_message: RawStreamMsg = await js.get_msg(seq=seq, stream_name=stream_name,
                                                              subject=subject, next=True)
            else:
                last_message: RawStreamMsg = await js.get_msg(seq=seq, stream_name=stream_name,
                                                              subject=subject,
                                                              next=True)
        except nats.js.errors.NotFoundError:
            break

        if last_message is None:
            break
        if msg_id is not None and last_message.headers["id"] != msg_id:
            continue
        await js.delete_msg(stream_name, last_message.seq)
        seq = last_message.seq + 1
