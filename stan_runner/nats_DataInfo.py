class DataInfo:
    """Contains lazy methods for accessing the data from the NATS stream, when required"""

    def __init__(self, data_hash: str):
        self._data_hash = data_hash
