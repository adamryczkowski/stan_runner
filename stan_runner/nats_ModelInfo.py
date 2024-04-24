class ModelInfo:
    """Contains lazy methods for accessing the model's data, when required"""
    _model_hash: str

    def __init__(self, model_hash: str):
        self._model_hash = model_hash
