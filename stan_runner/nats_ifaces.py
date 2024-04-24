from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type


class ISerializableObjectInfo(ABC):
    """Base class for every object that is trasfered by wire"""

    @abstractmethod
    @staticmethod
    def CreateFromSerialized(serialized: bytes, object_type: Type[ISerializableObjectInfo],
                             format: str = "pickle") -> ISerializableObjectInfo:
        ...

    @abstractmethod
    def serialize(self, format: str = "pickle") -> bytes:
        ...

    @abstractmethod
    @property
    def object_id(self) -> str:
        ...

    @abstractmethod
    def pretty_print(self) -> str:
        ...

    @abstractmethod
    @property
    def timestamp(self) -> float:
        ...

    @abstractmethod
    def __get_state__(self) -> dict:
        ...

    @abstractmethod
    def __set_state__(self, state: dict):
        ...

    def __repr__(self):
        return self.pretty_print()

    @abstractmethod
    def update_last_seen(self):
        ...
