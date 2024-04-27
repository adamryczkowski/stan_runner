from __future__ import annotations

import datetime
import time

import humanize
from overrides import overrides

from .nats_DTO import SerializableObjectInfo
from .nats_TaskInfo import TaskInfo
from .worker_capacity_info import WorkerCapacityInfo
from .nats_ModelInfo import ModelInfo


class WorkerInfo(SerializableObjectInfo):
    _capabilities: WorkerCapacityInfo
    _name: str
    _models_compiled: set[str]
    _birth: float

    def __init__(self, capabilities: dict, name: str, birth: float, models_compiled: set[str] = None,
                 object_id: str = None):
        super().__init__(object_id)
        self._capabilities = WorkerCapacityInfo(**capabilities)
        self._name = name
        self._models_compiled = models_compiled if models_compiled is not None else set()
        self._birth = birth

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["capabilities"] = self._capabilities.__getstate__()
        d["name"] = self._name
        d["models_compiled"] = list(self._models_compiled)
        d["birth"] = self._birth
        return d

    @overrides
    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self._capabilities = WorkerCapacityInfo(**state["capabilities"])
        self._name = state["name"]
        self._models_compiled = set(state["models_compiled"])
        self._birth = state["birth"]


    def can_handle_task(self, task: TaskInfo) -> bool:
        return True

    def does_have_model_precompiled(self, model: ModelInfo) -> bool:
        return model.model_hash in self._models_compiled

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> WorkerCapacityInfo:
        return self._capabilities

    def pretty_print(self):
        return f"""Worker {self.object_id} \"{self._name}\", created {humanize.naturaltime(datetime.datetime.fromtimestamp(time.time()) - datetime.datetime.fromtimestamp(self._birth))}, with capabilities:
{self._capabilities.pretty_print()}"""


    def __repr__(self):
        return self.pretty_print()
