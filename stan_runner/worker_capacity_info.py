from __future__ import annotations
from pathlib import Path
import psutil
import timeit
import concurrent.futures
import math
import json
from adams_text_utils import human_readable_size
from .nats_ifaces import ISerializableObject
from typing import Type
import datetime
import socket
import time

import humanize
import netifaces
from adams_text_utils import format_txt_list
from overrides import overrides

from .nats_DTO import SerializableObjectInfo



class WorkerCapacityInfo(ISerializableObject):
    _total_mem: int
    _free_mem: int
    _total_cores: int
    _disk_free: int
    _cpu_benchmark_st: float  # Single-threaded
    _cpu_benchmark_at: float  # All threads

    @staticmethod
    def CreateFromSerialized(serialized: bytes, object_type: Type[ISerializableObject],
                             format: str = "pickle") -> ISerializableObject:
        assert format == "json"
        d = json.loads(serialized)
        return WorkerCapacityInfo(**d)

    @staticmethod
    def BenchmarkSelf(model_cache_dir: Path) -> WorkerCapacityInfo:
        total_mem = psutil.virtual_memory().total
        free_mem = psutil.virtual_memory().available
        total_cores = psutil.cpu_count()
        disk_free = psutil.disk_usage(str(model_cache_dir)).free
        cpu_benchmark_st = single_threaded_benchmark()  # Single-threaded
        cpu_benchmark_at = multi_threaded_benchmark()  # All threads

        ans = WorkerCapacityInfo(total_mem, free_mem, total_cores, disk_free, cpu_benchmark_st, cpu_benchmark_at)
        return ans

    def __init__(self, total_mem: int, free_mem: int, total_cores: int, disk_free: int, cpu_benchmark_st: float,
                 cpu_benchmark_at: float):
        self._total_mem = total_mem
        self._free_mem = free_mem
        self._total_cores = total_cores
        self._disk_free = disk_free
        self._cpu_benchmark_st = cpu_benchmark_st
        self._cpu_benchmark_at = cpu_benchmark_at


    @property
    def total_mem(self) -> int:
        return self._total_mem

    @property
    def free_mem(self) -> int:
        return self._free_mem

    @property
    def total_cores(self) -> int:
        return self._total_cores

    @property
    def disk_free(self) -> int:
        return self._disk_free

    @property
    def cpu_benchmark_st(self) -> float:
        return self._cpu_benchmark_st

    @property
    def cpu_benchmark_at(self) -> float:
        return self._cpu_benchmark_at

    def pretty_print(self):
        ans = f"""Total memory: {human_readable_size(self.total_mem)}
Free memory: {human_readable_size(self.free_mem)}
Total cores: {self.total_cores}
Free disk space: {human_readable_size(self.disk_free)}
CPU benchmark (single-threaded): {self.cpu_benchmark_st:.2f} iterations/s
CPU benchmark (all threads): {self.cpu_benchmark_at:.2f} iterations/s"""
        return ans


    def __getstate__(self) -> dict:
        return {
            "total_mem": self.total_mem,
            "free_mem": self.free_mem,
            "total_cores": self.total_cores,
            "disk_free": self.disk_free,
            "cpu_benchmark_st": self.cpu_benchmark_st,
            "cpu_benchmark_at": self.cpu_benchmark_at
        }

    def __setstate__(self, state:dict):
        self._total_mem = state["total_mem"]
        self._free_mem = state["free_mem"]
        self._total_cores = state["total_cores"]
        self._disk_free = state["disk_free"]
        self._cpu_benchmark_st = state["cpu_benchmark_st"]
        self._cpu_benchmark_at = state["cpu_benchmark_at"]

    def serialize(self)->bytes:
        d = self.__getstate__()
        return json.dumps(d).encode()

    def __repr__(self):
        return self.pretty_print()


def single_threaded_benchmark(duration=1.0):
    start_time = timeit.default_timer()
    end_time = start_time + duration
    iterations = 0
    while timeit.default_timer() < end_time:
        math.sqrt(9999)
        iterations += 1
    return iterations / duration

def multi_threaded_benchmark(core_count:int=None, duration=1.0):
    if core_count is None:
        core_count = psutil.cpu_count()
    start_time = timeit.default_timer()
    end_time = start_time + duration
    iterations = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=core_count) as executor:
        while timeit.default_timer() < end_time:
            futures = [executor.submit(math.sqrt, 9999) for _ in range(100)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                iterations += 1
    return iterations / duration
