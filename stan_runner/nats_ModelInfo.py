from .nats_ifaces import ISerializableObject
from .nats_DTO import SerializableObject

class ModelInfo(SerializableObject):
    """Contains lazy methods for accessing the model's data, when required"""
    _model_hash: str

    def __init__(self, model_hash: str):
        super().__init__()
        self._model_hash = model_hash

    @property
    def model_hash(self) -> str:
        return self._model_hash