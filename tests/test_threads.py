"""Tests of pictologics/threads.py: the default number of threads and the way to change it."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import numba
import pytest

from pictologics import threads


class _Sysctl:
    """A C library with sysctlbyname: it writes `count` and returns `found` (0: the key is
    there)."""

    def __init__(self, count: int, found: int) -> None:
        self.count = count
        self.found = found
        self.name = b""

    def sysctlbyname(self, name: bytes, value: Any, size: Any, new: Any, new_size: Any) -> int:
        self.name = name
        value._obj.value = self.count  # value is ctypes.byref of a c_int
        return self.found


@pytest.fixture
def numba_threads(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """The numba threads of this thread as a dict ({"n": 14}, a pool of 14), and the
    module state of threads.py, both restored after the test."""
    state = {"n": 14}
    monkeypatch.setattr(numba, "get_num_threads", lambda: state["n"])
    monkeypatch.setattr(numba, "set_num_threads", lambda n: state.update(n=n))
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 14)
    monkeypatch.setattr(threads, "_default", threads._default)
    monkeypatch.setattr(threads, "_cores", threads._cores)
    with patch.dict(os.environ):
        os.environ.pop("PICTOLOGICS_NUM_THREADS", None)
        os.environ.pop("NUMBA_NUM_THREADS", None)
        yield state


def test_performance_cores_reads_the_sysctl_key() -> None:
    # The count of the key hw.perflevel0.logicalcpu, or 0 when the system has no such key
    for count, found, expected in ((10, 0, 10), (10, -1, 0)):
        library = _Sysctl(count, found)
        with patch.object(ctypes, "CDLL", return_value=library):
            assert threads._performance_cores() == expected
        assert library.name == b"hw.perflevel0.logicalcpu"


def test_fast_cores_of_a_mac_are_its_performance_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(threads, "_performance_cores", lambda: 10)
    assert threads._fast_cores() == 10
    # No performance-core key (an old macOS): all logical CPUs
    monkeypatch.setattr(threads, "_performance_cores", lambda: 0)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(threads, "_words", lambda path: [])
    assert threads._fast_cores() == 8


_CAPACITY = [f"cpu{c}" for c in (0, 2, 4, 6, 8, 10)]  # the CPUs of the process below


@pytest.mark.parametrize(
    ("machine", "files", "expected"),
    [
        ("x86_64", {}, 6),  # all CPUs of the process
        ("x86_64", {"cpu.max": ["max", "100000"]}, 6),  # a container without a limit
        ("x86_64", {"cpu.max": ["150000", "100000"]}, 2),  # 1.5 CPUs, rounded up
        ("x86_64", {"cpu.max": ["9000000", "100000"]}, 6),  # a limit above the CPUs
        ("x86_64", {"quota": ["200000"], "period": ["100000"]}, 2),  # cgroup v1
        ("x86_64", {"quota": ["-1"], "period": ["100000"]}, 6),  # cgroup v1 without a limit
        ("x86_64", {"cpu0": ["1024"], "cpu1": ["1024"], "cpu2": ["400"]}, 6),  # not ARM
        ("aarch64", {}, 6),  # no cpu_capacity files
        ("aarch64", dict.fromkeys(_CAPACITY, ["1024"]), 6),  # one kind of core
        # Two kinds: 3 x 1024 = 3072 > 6 x 400 = 2400, so the 3 fast cores
        ("aarch64", dict(zip(_CAPACITY, [["1024"]] * 3 + [["400"]] * 3, strict=True)), 3),
        # Three kinds: 1 x 1024 < 4 x 800 = 3200 > 6 x 300, so the prime and middle cores
        (
            "aarch64",
            dict(zip(_CAPACITY, [["1024"]] + [["800"]] * 3 + [["300"]] * 2, strict=True)),
            4,
        ),
        ("aarch64", {"cpu0": ["1024"], "cpu.max": ["100000", "100000"]}, 1),  # and a limit
    ],
)
def test_fast_cores_on_linux(
    monkeypatch: pytest.MonkeyPatch, machine: str, files: dict[str, list[str]], expected: int
) -> None:
    # The CPUs of the process (six here), the fastest ones of an ARM chip with more kinds
    # of cores, and at most the CPU limit of a container
    paths = {
        "/sys/fs/cgroup/cpu.max": files.get("cpu.max", []),
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": files.get("quota", []),
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": files.get("period", []),
    }
    for cpu in range(12):
        paths[f"/sys/devices/system/cpu/cpu{cpu}/cpu_capacity"] = files.get(f"cpu{cpu}", [])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {10, 8, 6, 4, 2, 0}, raising=False)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(threads, "_words", paths.__getitem__)
    assert threads._fast_cores() == expected


def test_words_reads_a_small_file(tmp_path: Path) -> None:
    path = tmp_path / "cpu.max"
    path.write_text("150000 100000\n")
    assert threads._words(str(path)) == ["150000", "100000"]
    assert threads._words(str(tmp_path / "missing")) == []


def test_configure_sets_the_variable_of_numba_before_its_import(
    monkeypatch: pytest.MonkeyPatch, numba_threads: dict[str, int]
) -> None:
    # Before the import of numba, the default goes into NUMBA_NUM_THREADS: numba then
    # starts every thread with it. PICTOLOGICS_NUM_THREADS comes first.
    monkeypatch.delitem(sys.modules, "numba")
    monkeypatch.setattr(threads, "_fast_cores", lambda: 10)
    threads._configure()
    assert os.environ["NUMBA_NUM_THREADS"] == "10" and threads._default == 10
    assert threads._cores == 10
    del os.environ["NUMBA_NUM_THREADS"]
    os.environ["PICTOLOGICS_NUM_THREADS"] = "12"
    threads._configure()
    assert os.environ["NUMBA_NUM_THREADS"] == "12" and threads._default == 12
    assert threads._cores == 10
    assert numba_threads["n"] == 14  # numba itself is not touched


@pytest.mark.parametrize(
    ("pictologics_variable", "numba_variable", "threads_before", "expected"),
    [
        (None, None, 14, 10),  # numba was imported first: the fast cores in this thread
        (None, None, 4, 4),  # a smaller number set before the import stays
        (None, "14", 14, 14),  # the NUMBA_NUM_THREADS of the user stays
        ("6", None, 14, 6),  # PICTOLOGICS_NUM_THREADS comes first
        ("6", "14", 14, 6),
        ("20", None, 14, 14),  # at most the threads that numba started
    ],
)
def test_configure_after_the_import_of_numba(
    numba_threads: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
    pictologics_variable: Optional[str],
    numba_variable: Optional[str],
    threads_before: int,
    expected: int,
) -> None:
    # After the import of numba, its variable must stay as it is: only the importing
    # thread gets the default.
    monkeypatch.setattr(threads, "_fast_cores", lambda: 10)
    numba_threads["n"] = threads_before
    for name, value in (
        ("PICTOLOGICS_NUM_THREADS", pictologics_variable),
        ("NUMBA_NUM_THREADS", numba_variable),
    ):
        if value is not None:
            os.environ[name] = value
    threads._configure()
    assert numba_threads["n"] == expected and threads._default == expected
    assert os.environ.get("NUMBA_NUM_THREADS") == numba_variable


@pytest.mark.parametrize("text", ["0", "-2", "1.5", "four", ""])
def test_a_bad_pictologics_variable_stops_the_import(
    numba_threads: dict[str, int], text: str
) -> None:
    os.environ["PICTOLOGICS_NUM_THREADS"] = text
    with pytest.raises(ValueError, match="PICTOLOGICS_NUM_THREADS must be a whole number"):
        threads._configure()


def test_set_and_get_the_number_of_threads(numba_threads: dict[str, int]) -> None:
    threads._default = 10
    threads.set_num_threads(3)
    assert threads.get_num_threads() == 3
    threads.set_num_threads()  # the default again
    assert threads.get_num_threads() == 10
    threads.set_num_threads(14)
    for n in (0, 15):
        with pytest.raises(ValueError, match="from 1 to 14, not .*PICTOLOGICS_NUM_THREADS"):
            threads.set_num_threads(n)
    assert threads.get_num_threads() == 14


@pytest.mark.parametrize(
    ("cores", "before", "after", "returned"),
    [
        (10, 10, 9, 10),  # the threads fill the fast cores: one less
        (10, 14, 13, 14),  # more threads than fast cores: one less
        (10, 4, 4, 0),  # a fast core is free: no change
        (1, 1, 1, 0),  # one thread: no change
        (1, 2, 1, 2),
    ],
)
def test_make_room_frees_a_core_when_the_threads_fill_them(
    numba_threads: dict[str, int], cores: int, before: int, after: int, returned: int
) -> None:
    threads._cores = cores
    numba_threads["n"] = before
    assert threads._make_room() == returned
    assert numba_threads["n"] == after


def test_the_package_exports_the_thread_functions() -> None:
    import pictologics

    assert pictologics.get_num_threads is threads.get_num_threads
    assert pictologics.set_num_threads is threads.set_num_threads
    assert {"get_num_threads", "set_num_threads"} <= set(pictologics.__all__)
