from .nats_ModelInfo import ModelInfo
from .nats_DataInfo import DataInfo
from .ifaces import StanOutputScope, StanResultEngine


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
