import json
import pickle
import time
import uuid

from overrides import overrides
from typing import Type
from .nats_ifaces import ISerializableObjectInfo
from .utils import make_dict_serializable


class SerializableObjectInfo(ISerializableObjectInfo):
    _object_id: str
    _timestamp: float

    def __init__(self, object_id: str = None, timestamp: float = None, object_id_prefix: str = ""):
        if object_id is None:
            object_id = object_id_prefix + str(uuid.uuid4())[0:8]
        if timestamp is None:
            timestamp = time.time()
        self._object_id = object_id
        self._timestamp = timestamp

    @staticmethod
    @overrides
    def CreateFromSerialized(serialized: bytes, object_type: Type[ISerializableObjectInfo],
                             format: str = "pickle") -> ISerializableObjectInfo:
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

    @property
    @overrides
    def object_id(self) -> str:
        return self._object_id

    @property
    @overrides
    def timestamp(self) -> float:
        return self._timestamp

    @overrides
    def __getstate__(self) -> dict:
        return {
            "object_id": self._object_id,
            "timestamp": self._timestamp
        }

    @overrides
    def __setstate__(self, state: dict):
        self._object_id = state["object_id"]
        self._timestamp = state["timestamp"]

    @overrides
    def update_last_seen(self):
        self._timestamp = time.time()
