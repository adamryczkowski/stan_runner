from __future__ import annotations

import asyncio
import datetime
import json
import pickle
import signal
import socket
import time
import uuid
from collections import deque

import humanize
import nats
import netifaces
from adams_text_utils import format_txt_list
from nats.aio.msg import Msg
from nats.js import JetStreamContext
from nats.js.api import RawStreamMsg
from nats.js.errors import NotFoundError

from .ifaces import StanOutputScope, StanResultEngine
from .nats_utils import STREAM_NAME, create_stream, connect_to_nats, WORKER_TIMEOUT_SECONDS
from .worker_capacity_info import WorkerCapacityInfo



# This is a message broker.
# It listens to the NATS messages with "stan.rundef.<run_id>" subject.
# For each messages it checks, if there is a corresponding "stan.result.<run_id>" message.
# If there is none, or the header "status" is "exception" it will try to send the rundef message to the free worker.
#
# It also listens to NATS messages that advertise free workers. Every worker that is free will send a message with
# subject "stan.worker_adv.<worker_id>". The message will contain the worker_id and the worker's capabilities (and in future model cache).
# The worker advertisements are not persistent, so the broker will only know about the workers that are currently
# online.
#
# The broker will maintain a list of workers that are currently free. When a new rundef message arrives, it will
# check if there are any free workers. If there are, it will send the rundef message to the worker. If there are none,
# it will wait for a worker advertisement message.
#
# Broker will send a message "stan.work_task.<worker_id>" that contains the rundef of the task to the appropriate free worker (potentially preferring workers with model already in cache).
# The message will contain the run_id and the run definition.
#
# In future, broker will also learn how to predict the run time and memory requirements of the tasks. It will use this
# information to decide which worker to send the task to.

class BrokerInfo:
    _broker_id: str
    _last_seen: float
    _hostname: str
    _network_addresses: dict[str, list[str]]  # All the network interfaces and their addresses

    @staticmethod
    def CreateFromLocalHost():
        last_seen = time.time()
        hostname = socket.gethostname()
        broker_id = "broker_" + str(uuid.uuid4())[0:8]

        network_addresses = {}
        for iface in netifaces.interfaces():
            iface_addresses = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in iface_addresses:
                for addr in iface_addresses[netifaces.AF_INET]:
                    if addr["addr"] == "127.0.0.1":
                        continue
                    if addr["addr"][0:4] == "172.":
                        continue
                    if iface not in network_addresses:
                        network_addresses[iface] = []
                    network_addresses[iface].append(addr["addr"])

        if "default" in netifaces.gateways():
            default_gateway = netifaces.gateways()["default"]
            if netifaces.AF_INET in default_gateway:
                gateway_ip, iface = default_gateway[netifaces.AF_INET]
                default_entry = network_addresses[iface]
                # Put the default gateway first
                del network_addresses[iface]
                network_addresses = {iface: default_entry, **network_addresses}

        return BrokerInfo(broker_id, last_seen, hostname, network_addresses)

    @staticmethod
    def CreateFromSerialized(serialized: bytes):
        d = json.loads(serialized)
        return BrokerInfo(**d)

    def __init__(self, broker_id: str, last_seen: float, hostname: str, network_addresses: dict[str, list[str]]):
        self._broker_id = broker_id
        self._last_seen = last_seen
        self._hostname = hostname
        self._network_addresses = network_addresses

    @property
    def broker_id(self) -> str:
        return self._broker_id

    @property
    def last_seen(self) -> float:
        return self._last_seen

    def update_last_seen(self):
        self._last_seen = time.time()

    def pretty_print(self):
        ans = f"""Broker {self._broker_id} \"{self._hostname}\", last seen {humanize.naturaltime(datetime.datetime.fromtimestamp(time.time()) - datetime.datetime.fromtimestamp(self._last_seen))}, with network addresses:\n\n"""

        ifaces = []
        for iface, addresses in self._network_addresses.items():
            ifaces.append(f"* {iface}: {format_txt_list(addresses, max_length=5)}")

        return ans + "\n".join(ifaces)

    def serialize(self) -> bytes:
        d = {
            "broker_id": self._broker_id,
            "last_seen": self._last_seen,
            "hostname": self._hostname,
            "network_addresses": self._network_addresses
        }

        return json.dumps(d).encode()

    def __repr__(self):
        return self.pretty_print()


class ModelInfo:
    """Contains lazy methods for accessing the model's data, when required"""
    _model_hash: str

    def __init__(self, model_hash: str):
        self._model_hash = model_hash


class DataInfo:
    """Contains lazy methods for accessing data, when required"""
    _data_hash: str

    def __init__(self, data_hash: str):
        self._data_hash = data_hash


class TaskInfo:
    _task_hash: str
    _model: ModelInfo
    _data: DataInfo
    _scope: StanOutputScope
    _engine: StanResultEngine

    def __init__(self, task_hash: str, model_hash: str, data_hash: str, scope: StanOutputScope,
                 engine: StanResultEngine):
        self._task_hash = task_hash
        self._model = ModelInfo(model_hash)
        self._data = DataInfo(data_hash)
        self._scope = scope
        self._engine = engine


class WorkerInfo:
    _capabilities: WorkerCapacityInfo
    _worker_hash: str
    _name: str
    _models_compiled: set[str]
    _last_seen: float

    def can_handle_task(self, task: TaskInfo) -> bool:
        return True

    def __init__(self, capabilities: dict, worker_hash: str, name: str, models_compiled: set[str] = None):
        self._capabilities = WorkerCapacityInfo(**capabilities)
        self._worker_hash = worker_hash
        self._name = name
        self._models_compiled = models_compiled if models_compiled is not None else set()
        self._last_seen = time.time()

    @property
    def worker_hash(self) -> str:
        return self._worker_hash

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> WorkerCapacityInfo:
        return self._capabilities

    def pretty_print(self):
        return f"""Worker {self._worker_hash} \"{self._name}\", last seen {humanize.naturaltime(datetime.datetime.fromtimestamp(time.time()) - datetime.datetime.fromtimestamp(self._last_seen))}, with capabilities:
{self._capabilities.pretty_print()}"""

    @property
    def last_seen(self) -> float:
        return self._last_seen

    def update_last_seen(self):
        self._last_seen = time.time()

    def __repr__(self):
        return self.pretty_print()


class MessageBroker:
    _server_context: JetStreamContext
    _server_raw: nats.NATS

    _workers: dict[str, WorkerInfo]
    _tasks: dict[str, TaskInfo]
    _task_queue: deque[str]

    _subscriptions: list | None

    _broker_id: BrokerInfo

    @staticmethod
    async def Create(server_url: str, user: str, password: str = None):
        nc = await connect_to_nats(nats_connection=server_url, user=user, password=password)
        js, stream = await create_stream(nc, permanent_storage=True, stream_name=STREAM_NAME)
        return MessageBroker(nc, js)

    def __init__(self, nc: nats.NATS, js: JetStreamContext):
        self._server_raw = nc
        self._server_context = js

        self._workers = {}
        self._tasks = {}
        self._task_queue = deque()
        self._broker_id = BrokerInfo.CreateFromLocalHost()
        print(self._broker_id.pretty_print())

    def prune_workers(self):
        """Removes workers, that have not advertised themselves in the last WORKER_TIMEOUT_SECONDS seconds"""
        to_delete = []
        for worker_id, worker in self._workers.items():
            if time.time() - worker.last_seen > WORKER_TIMEOUT_SECONDS:
                to_delete.append(worker_id)

        for worker_id in to_delete:
            del self._workers[worker_id]

    async def handle_worker_adv(self, message: Msg):
        """Called when a worker advertises itself"""
        worker_id = message.subject.split(".")[-1]
        worker_props = json.loads(message.data)
        worker = WorkerInfo(**worker_props)
        assert worker_id == worker.worker_hash

        if worker.worker_hash not in self._workers:
            print(f"New worker: {worker}")
            self._workers[worker.worker_hash] = worker
        else:
            self._workers[worker.worker_hash].update_last_seen()

        await self.try_sending_tasks_to_workers()

    async def handle_task(self, message: Msg):
        """Called when a new task arrives"""
        task_hash = message.subject.split(".")[-1]
        task_props = pickle.loads(message.data)
        task = TaskInfo(**task_props)
        assert task_hash == task._task_hash

        assert task_hash not in self._tasks
        self._tasks[task_hash] = task
        self._task_queue.append(task_hash)
        await self.try_sending_tasks_to_workers()

    async def try_sending_tasks_to_workers(self):
        """Checks if there is an opportunity to send tasks to workers"""

        # Remove workers that have not advertised themselves in a while
        self.prune_workers()

        # Check if there are any workers
        if len(self._workers) == 0:
            return

        # Check if there are any tasks
        if len(self._task_queue) == 0:
            return

        # Check if there are any workers that can handle the task
        task_hash = self._task_queue.popleft()
        task = self._tasks[task_hash]

        for worker_id, worker in self._workers.items():
            if worker.can_handle_task(task):
                print(f"Sending task {task_hash} to worker {worker_id}")
                await self._server_context.publish(f"stan.work_order.{worker_id}", task_hash.encode())
                return

        # If we are here, there were are no workers that can handle the task
        self._task_queue.appendleft(task_hash)

    async def keep_alive(self):
        while True:
            print(f"Broker {self._broker_id} is alive")
            try:
                last_message: RawStreamMsg | None = await self._server_context.get_last_msg(stream_name=STREAM_NAME,
                                                                                            subject="stan.broker.alive")
            except nats.js.errors.NotFoundError:
                last_message = None

            if last_message is not None:
                time_to_last_message = time.time() - float(last_message.headers["timestamp"])
                if last_message.headers["broker_id"] != self._broker_id.broker_id:
                    if time_to_last_message < WORKER_TIMEOUT_SECONDS:
                        raise Exception(f"Broker id mismatch: {last_message.headers['broker_id']} != {self._broker_id}")
            else:
                time_to_last_message = float("inf")

            if (time_to_wait := max(0, WORKER_TIMEOUT_SECONDS - time_to_last_message)) > 0:
                await asyncio.sleep(time_to_wait)

            broker_bin = self._broker_id.serialize()
            if self._subscriptions is None:
                break
            await self._server_context.publish("stan.broker.alive", payload=broker_bin, stream=STREAM_NAME,
                                               headers={"broker_id": self._broker_id.broker_id,
                                                        "format": "json",
                                                        "timestamp": str(float(time.time()))})
            if last_message is not None:
                await self._server_context.delete_msg(STREAM_NAME, last_message.seq)

    async def the_loop(self):
        # Attach the shutdown coroutine to SIGINT and SIGTERM
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(self.shutdown(loop)))
        loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(self.shutdown(loop)))

        sub1 = await self._server_raw.subscribe("stan.worker_adv.>", cb=self.handle_worker_adv)
        sub2 = await self._server_raw.subscribe("stan.taskdef.>", cb=self.handle_task)

        self._subscriptions = [sub1, sub2]

        await self.keep_alive()

    async def shutdown(self, loop):
        print("Unsubscribing...")
        for s in self._subscriptions:
            await s.unsubscribe()

        self._subscriptions = None

        try:
            last_message: RawStreamMsg | None = await self._server_context.get_last_msg(stream_name=STREAM_NAME,
                                                                                        subject="stan.broker.alive")
        except nats.js.errors.NotFoundError:
            last_message = None

        await asyncio.sleep(0)  # Yield to other tasks to finish

        print("Removing broadcast message...")
        if last_message is not None:
            await self._server_context.delete_msg(STREAM_NAME, last_message.seq)

        print("Received exit signal, shutting down...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        print(f"Cancelling {len(tasks)} tasks")
        [task.cancel() for task in tasks]

        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()
