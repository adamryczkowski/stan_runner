from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type


class NetworkDuplicateError(Exception):
    _other_object_id: str

    def __init__(self, other_object_id: str, message: str = None):
        if message is not None:
            super().__init__(message)
        else:
            super().__init__(f"Id mismatch: {other_object_id}")
        self._other_object_id = other_object_id

    @property
    def other_object_id(self):
        return self._other_object_id


class ISerializableObject(ABC):
    @staticmethod
    @abstractmethod
    def CreateFromSerialized(serialized: bytes, object_type: Type[ISerializableObject],
                             format: str = "pickle") -> ISerializableObject:
        ...

    def __init__(self):
        super().__init__()

    @abstractmethod
    def serialize(self, format: str = "pickle") -> bytes:
        ...

    @abstractmethod
    def __getstate__(self) -> dict:
        ...

    @abstractmethod
    def __setstate__(self, state: dict):
        ...


class ISerializableObjectInfo(ISerializableObject):
    """Base class for every object that is transferred by wire"""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def object_id(self) -> str:
        ...

    @abstractmethod
    def pretty_print(self) -> str:
        ...

    @property
    @abstractmethod
    def timestamp(self) -> float|None:
        ...

    def __repr__(self):
        return self.pretty_print()

    @abstractmethod
    def update_last_seen(self, timestamp: float=None):
        ...
