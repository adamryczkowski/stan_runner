from __future__ import annotations

import datetime
import time

import humanize
from overrides import overrides

from .nats_DTO import SerializableObjectInfo
from .nats_TaskInfo import TaskInfo
from .worker_capacity_info import WorkerCapacityInfo


class WorkerInfo(SerializableObjectInfo):
    _capabilities: WorkerCapacityInfo
    _worker_hash: str
    _name: str
    _models_compiled: set[str]
    _last_seen: float

    def __init__(self, capabilities: dict, worker_hash: str, name: str, models_compiled: set[str] = None,
                 object_id: str = None, timestamp: float = None
                 ):
        super().__init__(object_id, timestamp, object_id_prefix="worker_")
        self._capabilities = WorkerCapacityInfo(**capabilities)
        self._worker_hash = worker_hash
        self._name = name
        self._models_compiled = models_compiled if models_compiled is not None else set()

    @overrides
    def __getstate__(self) -> dict:
        d = super().__get_state__()
        d["capabilities"] = self._capabilities.__getstate__()
        d["worker_hash"] = self._worker_hash
        d["name"] = self._name
        d["models_compiled"] = list(self._models_compiled)
        return d

    @overrides
    def __setstate__(self, state: dict):
        super().__set_state__(state)
        self._capabilities = WorkerCapacityInfo(**state["capabilities"])
        self._worker_hash = state["worker_hash"]
        self._name = state["name"]
        self._models_compiled = set(state["models_compiled"])

    def can_handle_task(self, task: TaskInfo) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> WorkerCapacityInfo:
        return self._capabilities

    def pretty_print(self):
        return f"""Worker {self._worker_hash} \"{self._name}\", last seen {humanize.naturaltime(datetime.datetime.fromtimestamp(time.time()) - datetime.datetime.fromtimestamp(self._last_seen))}, with capabilities:
{self._capabilities.pretty_print()}"""

    def __repr__(self):
        return self.pretty_print()
