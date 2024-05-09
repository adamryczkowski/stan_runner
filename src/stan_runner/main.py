from pathlib import Path
from typing import Any

import numpy as np
from overrides import overrides

from . import IStanData, StanOutputScope, StanResultEngine
from .pystan_run import IStanBackend, install_all_dependencies
from .model import StanModel, IStanModel
from .data import StanData, IStanData
from .runner import StanRun, IStanRun

# TODO: Make it into a interface, so I can import actual classes and return them.
class StanBackend(IStanBackend):
    _cpu_cores: int
    _model_cache_parent: Path
    _data_cache_parent: Path
    _run_cache_parent: Path

    def __init__(self, model_cache_parent: Path = Path("model_cache"),
                 data_cache_parent: Path = Path("data_cache"), run_cache_parent: Path = Path("run_cache"),
                 cpu_cores: int = 1 ):
        assert isinstance(model_cache_parent, Path)
        assert isinstance(data_cache_parent, Path)
        assert isinstance(run_cache_parent, Path)
        model_cache_parent.mkdir(exist_ok=True, parents=True)
        data_cache_parent.mkdir(exist_ok=True, parents=True)
        run_cache_parent.mkdir(exist_ok=True, parents=True)

        self._cpu_cores = cpu_cores
        self._model_cache_parent = model_cache_parent
        self._data_cache_parent = data_cache_parent
        self._run_cache_parent = run_cache_parent

    @property
    def model_cache_parent(self) -> Path:
        return self._model_cache_parent

    @property
    def data_cache_parent(self) -> Path:
        return self._data_cache_parent

    @property
    def run_cache_parent(self) -> Path:
        return self._run_cache_parent

    @property
    def cpu_cores(self) -> int:
        return self._cpu_cores

    @overrides
    def install_all_dependencies(self):
        install_all_dependencies(self._cpu_cores)

    @overrides
    def make_model(self, model_name: str, model_code: str, stanc_opts: dict[str, Any] = None,
                   cpp_opts: dict[str, Any] = None) -> IStanModel:
        return StanModel(self._model_cache_parent, model_name, model_code, stanc_opts, cpp_opts)

    @overrides
    def make_data(self, data: dict[str, float | int | np.ndarray], data_opts: dict[str, str] = None) -> IStanData:
        return StanData(self._data_cache_parent, data, data_opts)

    @overrides
    def make_run(self, data: IStanData, model: IStanModel, output_scope: StanOutputScope, run_engine: StanResultEngine,
                 sample_count: int, run_opts: dict[str, Any] = None) -> IStanRun:
        return StanRun(self._run_cache_parent, data, model, output_scope, run_engine, sample_count, run_opts)
