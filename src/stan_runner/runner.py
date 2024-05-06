from __future__ import annotations

from pathlib import Path
from typing import Any

from nats_foundation import IMetaObject
from nats_foundation import calc_hash
from overrides import overrides

from .ifaces import StanResultEngine, StanOutputScope
from .ifaces2 import IStanRunMeta, IStanRun, IMetaObjectBase, MetaObjectBase, IStanModelMeta, IStanDataMeta, IStanData, \
    IStanModel, IResultPromise
from .pystan_run import run


class StanRunMeta(IStanRunMeta, MetaObjectBase):
    _run_opts: dict[str, Any]
    _model: IStanModelMeta
    _data: IStanDataMeta
    _output_scope: StanOutputScope
    _run_engine: StanResultEngine

    def __init__(self, run_hash: int, output_scope: StanOutputScope, run_engine: StanResultEngine,
                 run_opts: dict[str, Any], model: IStanModelMeta, data: IStanDataMeta):
        assert isinstance(model, IStanModelMeta)
        assert isinstance(data, IStanDataMeta)
        assert isinstance(output_scope, StanOutputScope)
        assert isinstance(run_engine, StanResultEngine)
        assert isinstance(run_opts, dict)
        assert "sample_count" in run_opts

        super().__init__(run_hash)
        self._run_opts = {}
        self._model = model
        self._data = data
        self._output_scope = output_scope
        self._run_engine = run_engine

    @property
    @overrides
    def run_opts(self) -> dict[str, Any]:
        return self._run_opts

    @property
    @overrides
    def output_scope(self) -> StanOutputScope:
        return self._output_scope

    @property
    @overrides
    def run_engine(self) -> StanResultEngine:
        return self._run_engine

    @property
    @overrides
    def sample_count(self) -> int:
        return self._run_opts["sample_count"]

    @overrides
    def get_data_meta(self) -> IStanDataMeta:
        return self._data

    @overrides
    def get_model_meta(self) -> IStanModelMeta:
        return self._model

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["run_opts"] = self._run_opts
        d["model"] = self._model.object_hash
        d["data"] = self._data.object_hash
        d["output_scope"] = self._output_scope.txt_value()
        d["run_engine"] = self._run_engine.txt_value()
        return d


class StanRun(IStanRun, IMetaObjectBase):
    _run_folder: Path
    _data: IStanData
    _model: IStanModel
    _run_opts: dict[str, Any]
    _output_scope: StanOutputScope
    _run_engine: StanResultEngine

    def __init__(self, run_folder: Path, data: IStanData, model: IStanModel, output_scope: StanOutputScope,
                 run_engine: StanResultEngine, run_opts: dict[str, Any]):
        assert isinstance(data, IStanData)
        assert isinstance(model, IStanModel)
        assert isinstance(output_scope, StanOutputScope)
        assert isinstance(run_engine, StanResultEngine)
        assert isinstance(run_opts, dict)
        assert "sample_count" in run_opts
        assert isinstance(run_folder, Path)

        super().__init__()

        if not run_folder.exists():
            run_folder.mkdir(parents=True)
        self._run_folder = run_folder
        self._data = data
        self._model = model
        self._run_opts = run_opts
        self._output_scope = output_scope
        self._run_engine = run_engine

    @overrides
    def get_metaobject(self) -> IMetaObject:
        # noinspection PyTypeChecker
        return StanRunMeta(calc_hash(self), self._output_scope, self._run_engine, self._run_opts,
                           self._model.get_metaobject(), self._data.get_metaobject())

    @property
    @overrides
    def run_opts(self) -> dict[str, Any]:
        return self._run_opts

    @property
    @overrides
    def output_scope(self) -> StanOutputScope:
        return self._output_scope

    @property
    @overrides
    def run_engine(self) -> StanResultEngine:
        return self._run_engine

    @property
    @overrides
    def sample_count(self) -> int:
        return self._run_opts["sample_count"]

    @overrides
    def get_data_meta(self) -> IStanDataMeta:
        # noinspection PyTypeChecker
        return self._data

    @overrides
    def get_model_meta(self) -> IStanModelMeta:
        # noinspection PyTypeChecker
        return self._model

    @overrides
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d["run_folder"] = self._run_folder
        d["data"] = self._data.object_hash
        d["model"] = self._model.object_hash
        d["run_opts"] = self._run_opts
        d["output_scope"] = self._output_scope.txt_value()
        d["run_engine"] = self._run_engine.txt_value()
        return d

    @overrides
    def get_object(self) -> IStanRun:
        return self

    @property
    @overrides
    def is_object_available(self) -> bool:
        return True

    @property
    @overrides
    def object_hash(self) -> int:
        return calc_hash(self.__getstate__())

    @overrides
    def run(self) -> IResultPromise:
        return run(self)

    @property
    @overrides
    def run_folder(self) -> Path:
        return self._run_folder


