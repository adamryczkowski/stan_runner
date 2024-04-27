import json
import pickle
import time
import uuid

from overrides import overrides
from typing import Type
from .nats_ifaces import ISerializableObjectInfo, ISerializableObject
from .utils import make_dict_serializable



class SerializableObject(ISerializableObject):
    def __init__(self):
        super().__init__()

    @staticmethod
    @overrides
    def CreateFromSerialized(serialized: bytes, object_type: Type[ISerializableObject],
                             format: str = "pickle") -> ISerializableObject:
        if format == "json":
            d = json.loads(serialized)
        elif format == "pickle":
            d = pickle.loads(serialized)
        else:
            raise ValueError(f"Unknown format: {format}")

        if isinstance(d, dict):
            # noinspection PyArgumentList
            return object_type(**d)
        else:
            assert isinstance(d, object_type)
        return d

    @overrides
    def serialize(self, format: str = "pickle") -> bytes:
        d = make_dict_serializable(self.__getstate__())
        if format == "json":
            return json.dumps(d).encode("utf-8")
        elif format == "pickle":
            return pickle.dumps(d)

    @overrides
    def __getstate__(self) -> dict:
        return {}

    @overrides
    def __setstate__(self, state: dict):
        pass


class SerializableObjectInfo(SerializableObject, ISerializableObjectInfo):
    _object_id: str
    _timestamp: float | None

    def __init__(self, object_id: str = None):
        super().__init__()
        if object_id is None:
            object_id = str(uuid.uuid4())[0:8]
        self._object_id = object_id
        self._timestamp = None


    @property
    @overrides
    def object_id(self) -> str:
        return self._object_id

    @property
    @overrides
    def timestamp(self) -> float|None:
        return self._timestamp

    @overrides
    def __getstate__(self) -> dict:
        super_state = super().__getstate__()
        super_state["object_id"] = self._object_id
        return super_state

    @overrides
    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self._object_id = state["object_id"]

    @overrides
    def update_last_seen(self, timestamp: float=None):
        if timestamp is not None:
            self._timestamp = timestamp
        else:
            self._timestamp = time.time()

