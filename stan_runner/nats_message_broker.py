from __future__ import annotations

import asyncio
import json
import pickle
import signal
import time
from collections import deque

import nats
from nats.aio.msg import Msg
from nats.js import JetStreamContext

from .nats_DTO_BrokerInfo import BrokerInfo
from .nats_TaskInfo import TaskInfo
from .nats_DTO_WorkerInfo import WorkerInfo
from .nats_utils import STREAM_NAME, create_stream, connect_to_nats, WORKER_TIMEOUT_SECONDS, KeepAliver


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


class MessageBroker:
    _server_context: JetStreamContext
    _server_raw: nats.NATS

    _workers: dict[str, WorkerInfo]
    _tasks: dict[str, TaskInfo]
    _task_queue: deque[str]
    _keep_aliver: KeepAliver
    _keep_aliver_task: asyncio.Task | None

    _subscriptions: list | None

    _broker_id: BrokerInfo

    @staticmethod
    async def Create(server_url: str, user: str, password: str = None):
        nc = await connect_to_nats(nats_connection=server_url, user=user, password=password)
        js, stream = await create_stream(nc, permanent_storage=True, stream_name=STREAM_NAME)
        broker = MessageBroker(nc, js)


    async def check_for_network_duplicates(self):
        await self._keep_aliver.check_for_network_duplicates()

    def __init__(self, nc: nats.NATS, js: JetStreamContext):
        self._server_raw = nc
        self._server_context = js

        self._workers = {}
        self._tasks = {}
        self._task_queue = deque()
        self._broker_id = BrokerInfo.CreateFromLocalHost()
        self._keep_aliver = KeepAliver(self._server_context, "stan.broker.alive", timeout=WORKER_TIMEOUT_SECONDS,
                                       unique_id=self._broker_id.broker_id,
                                       serialized_content=self._broker_id.serialize(),
                                       serialized_format="json")
        self._keep_aliver_task = None

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

    async def the_loop(self):
        # Attach the shutdown coroutine to SIGINT and SIGTERM
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(self.shutdown(loop)))
        loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(self.shutdown(loop)))

        sub1 = await self._server_raw.subscribe("stan.worker_adv.>", cb=self.handle_worker_adv)
        sub2 = await self._server_raw.subscribe("stan.taskdef.>", cb=self.handle_task)

        self._subscriptions = [sub1, sub2]

        self._keep_aliver_task = asyncio.create_task(self._keep_aliver.keep_alive())

    async def shutdown(self, bCloseNATS: bool = False):
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

        if bCloseNATS:
            print("Closing the NATS connection...")
            await self._server_raw.close()
