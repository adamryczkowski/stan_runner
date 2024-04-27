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
from nats.aio.msg import Msg
from nats.aio.subscription import Subscription
from .nats_ifaces import NetworkDuplicateError, ISerializableObjectInfo, ISerializableObject
from collections import deque

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


class AliveListener:
    """Class that listens to keep-alive messages"""

    _subject: str  # The subject to listen to
    _alive_entities: dict[str, ISerializableObjectInfo]
    _js: JetStreamContext
    _expected_type: type[ISerializableObjectInfo]
    _queue: deque[ISerializableObjectInfo]

    _listen_task: Subscription | None
    _time_to_next_prune: float  # -1 means that the queue is empty and there is no need to wait.
    _prune_event: asyncio.Event
    _prune_task: asyncio.Task | None

    def __init__(self, js: JetStreamContext, subject: str, expected_type: type[ISerializableObjectInfo]):
        self._subject = subject
        self._alive_entities = {}
        self._js = js
        self._expected_type = expected_type
        self._listen_task = None
        self._time_to_next_prune = -1
        self._prune_event = asyncio.Event()
        self._queue = deque()
        self._prune_task = None

    async def handle_keep_alive(self, message: Msg):
        """Handle a keep-alive message."""
        if message.headers["state"] == "shutdown":
            object_id = message.headers["id"]
            if object_id in self._alive_entities:
                del self._alive_entities[object_id]

            # Remove the message
            await self._js.delete_msg(STREAM_NAME, message.seq)
            return

        try:
            entity: ISerializableObject = self._expected_type.CreateFromSerialized(message.data, self._expected_type,
                                                                                   message.headers["format"])
            assert isinstance(entity, ISerializableObjectInfo)
        except Exception as e:
            print(f"Error: {e}")
            return

        entity.update_last_seen(float(message.headers["timestamp"]))  # An important line.
        # The entity has a slot for its timestamp, which is updated here.
        self._alive_entities[entity.object_id] = entity
        self._queue.append(entity)
        if self._time_to_next_prune < 0:
            self._prune_event.set()

    async def the_loop(self):
        assert self._listen_task is None
        self._listen_task = await self._js.subscribe(self._subject, cb=self.handle_keep_alive)

        self._prune_task = asyncio.create_task(self.prune_loop())
        await self._prune_task

    def prune_queue(self):
        """Remove entities from queue that have not advertised themselves in the last WORKER_TIMEOUT_SECONDS seconds.
        Sets the time needed to wait for the next pruning."""
        to_delete = []
        for i, entity in enumerate(self._queue):
            if time.time() - entity.timestamp > WORKER_TIMEOUT_SECONDS:
                to_delete.append((i, entity.object_id, entity.timestamp))
            else:
                break  # We can break here, because the queue is sorted by timestamp

        for i, entity_id, timestamp in reversed(to_delete):
            del self._queue[i]
            if entity_id not in self._alive_entities:
                continue
            if self._alive_entities[entity_id].timestamp == timestamp:
                del self._alive_entities[entity_id]

        if len(self._queue) == 0:
            self._time_to_next_prune = -1
        else:
            first_item = self._queue[0]
            self._time_to_next_prune = first_item.timestamp + WORKER_TIMEOUT_SECONDS - time.time()

    async def prune_loop(self):
        while True:
            # Wait for asyncio.sleep(self._time_to_next_prune) and for the self._event whatever comes first.
            if self._time_to_next_prune < 0:
                await self._prune_event.wait()
            else:
                await asyncio.wait([asyncio.sleep(self._time_to_next_prune), self._prune_event.wait()],
                                   return_when=asyncio.FIRST_COMPLETED)
            self.prune_queue()

    async def shutdown(self):
        if self._listen_task is not None:
            await self._listen_task.unsubscribe()
            self._listen_task = None
        self._prune_event.set()
        self._queue.clear()
        self._alive_entities.clear()
        self._prune_task.cancel()


    def __len__(self):
        return len(self._alive_entities)

    def __iter__(self):
        return iter(self._alive_entities.values())

    def values(self):
        return self._alive_entities.values()

    def __getitem__(self, key):
        return self._alive_entities[key]

    def __contains__(self, key):
        return key in self._alive_entities

    def __delitem__(self, key):
        del self._alive_entities[
            key]  # We simply abandon the object in the queue. In time it will get pruned very efficiently.


class KeepAliver:
    """Class that sends keep-alive messages to the NATS server."""
    _subject: str  # The subject to send the keep-alive messages to
    _js: JetStreamContext  # The JetStream context to use
    _my_id: str | None  # The ID of this worker. Will be checked against the last message to ensure that only one worker is running on this subject.
    _timeout: float
    _content_object: bytes | None
    _serialized_format: str
    _last_msg_seq: int | None
    _is_busy: bool
    _keep_alive_task: asyncio.Task | None

    @staticmethod
    async def Create(js: JetStreamContext, subject: str, timeout: float = 60., unique_id: str = None,
                     serialized_content: bytes = None, serialized_format: str = "json") -> KeepAliver:
        obj = KeepAliver(js, subject, timeout, unique_id, serialized_content, serialized_format)
        await obj.remove_all_past_messages()
        return obj

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
        self._is_busy = False
        self._keep_alive_task = None

    async def remove_all_past_messages(self):
        await clear_subject(self._js, STREAM_NAME, self._subject)

    async def set_as_busy(self):
        if self._is_busy:
            return
        self._is_busy = False
        await self.resend_message()

    async def set_as_ready(self):
        if not self._is_busy:
            return
        self._is_busy = False
        await self.resend_message()

    async def resend_message(self):
        headers = {
            "state": "busy" if self._is_busy else "ready",
            "format": self._serialized_format,
            "timestamp": str(float(time.time()))}

        if self._my_id is not None:
            headers["id"] = self._my_id

        ack = await self._js.publish(self._subject, payload=self._content_object, stream=STREAM_NAME,
                                     headers=headers)
        if self._last_msg_seq is not None:
            await self._js.delete_msg(STREAM_NAME, self._last_msg_seq)
            self._last_msg_seq = ack.seq


    async def shutdown(self):
        if self._last_msg_seq is not None:
            print("Sending shutdown message...")
            await self._js.publish(self._subject, payload=b"shutdown", stream=STREAM_NAME, headers={
                "state": "shutdown",
                "format": self._serialized_format,
                "id": self._my_id,
                "timestamp": str(float(time.time()))
            })

            try:
               await self._js.delete_msg(STREAM_NAME, self._last_msg_seq)
               self._last_msg_seq = None

            except nats.js.errors.NotFoundError:
                print("Cannot delete the last keep-alive message!")
                pass

        if self._keep_alive_task is not None and not self._keep_alive_task.cancelled():
            self._keep_alive_task.cancel()

    async def check_for_network_duplicates(self)->float:
        """Check if there are any other workers on the network with the same ID. Sets last message sequence nr.
        Returns time to time to last message
        """
        try:
            last_message: RawStreamMsg | None = await self._js.get_last_msg(stream_name=STREAM_NAME,
                                                                            subject=self._subject)
        except nats.js.errors.NotFoundError:
            return float("inf")

        assert last_message is not None
        time_to_last_message = time.time() - float(last_message.headers["timestamp"])
        if self._my_id is not None and last_message.headers["id"] != self._my_id and last_message.headers[
            "state"] != "shutdown":
            if time_to_last_message < self._timeout:
                raise NetworkDuplicateError(last_message.headers["id"],
                                            f"Id mismatch: {last_message.headers['id']} != {self._my_id}")
        self._last_msg_seq = last_message.seq
        return time_to_last_message

    async def keep_alive(self):
        """Keep sending keep-alive messages to the NATS server."""

        try:
            while True:
                time_to_last_message = await self.check_for_network_duplicates()

                if (time_to_wait := max(0., self._timeout - time_to_last_message)) > 0:
                    await asyncio.sleep(time_to_wait)

                await self.resend_message()

        except asyncio.CancelledError:
            self._keep_alive_task = None
            print("Canceling keep-alive message...")
            await self.shutdown()

    async def the_loop(self):
        self._keep_alive_task = asyncio.create_task(self.keep_alive())
        await self._keep_alive_task


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
