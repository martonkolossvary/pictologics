"""The number of threads of pictologics.

Pictologics runs its parallel parts on the threads of numba: the numba kernels, the FFT
filters and the thread pools of the filters. numba cuts a parallel loop into equal parts,
one part for each thread, so the slowest core sets the time of the loop. Thus, the default
number of threads depends on the cores of the computer:

- A Mac with Apple silicon: the performance cores (for example, 10 of the 14 cores of an
  M4 Pro).
- Linux: the CPUs of the process, at most the CPU limit of a container. On an ARM chip with
  more kinds of cores, only the fast cores.
- Windows and other computers: all logical CPUs, as numba does. The slow cores of an
  Intel chip are many and not much slower than a second thread of a fast core, so they help.

For another number, set the environment variable PICTOLOGICS_NUM_THREADS before you import
pictologics, or call `set_num_threads` in your script. A NUMBA_NUM_THREADS of your own also
sets the number. The results do not depend on the number of threads, with one exception:
from SciPy 1.18, the FFT of SciPy splits its work by thread. So the Riesz and Simoncelli
filters, the LoG of float64 images of 131,072 voxels or more, and Moran's I and Geary's C of
large ROIs can change in their last digits with the number of threads (Moran's I by about
1e-16, relative, in a test).

This module also gives the memory budget of the large steps: half of the memory of the
computer (at most the limit of a Linux container), shared by the workers of run_batch.
"""

from __future__ import annotations

import functools
import math
import os
import platform
import sys
from typing import Optional

# This module does not import numba at its top: pictologics/__init__.py calls _configure
# before numba starts.
_default = 0  # the number of threads that set_num_threads() sets (see _configure)
_cores = 0  # the fast cores of this computer (see _fast_cores), found at the import
_memory_shares = 1  # the processes that share the memory budget (see _share_memory)


def get_num_threads() -> int:
    """The number of threads that pictologics uses in the calling thread.

    Returns:
        The number of numba threads of the calling thread.
    """
    import numba

    return int(numba.get_num_threads())


def set_num_threads(n: Optional[int] = None) -> None:
    """Set the number of threads that pictologics uses in the calling thread.

    As with `numba.set_num_threads`, the number applies only to the thread that calls this
    function. It must be from 1 to the number of threads that numba started at the import.
    For more threads, set the environment variable PICTOLOGICS_NUM_THREADS before you
    import pictologics.

    Args:
        n: The number of threads. None (the default) sets the default of this computer
            again, as the import chose it.

    Raises:
        ValueError: If n is less than 1 or more than the threads that numba started.
    """
    import numba

    if n is None:
        n = _default
    limit = numba.config.NUMBA_NUM_THREADS
    if not 1 <= n <= limit:
        raise ValueError(
            f"The number of threads must be from 1 to {limit}, not {n}. For more threads, set "
            "the environment variable PICTOLOGICS_NUM_THREADS before you import pictologics."
        )
    numba.set_num_threads(n)


def _configure() -> None:
    """Choose the default number of threads at the import of pictologics.

    PICTOLOGICS_NUM_THREADS comes first, then a NUMBA_NUM_THREADS of the user, then the fast
    cores. numba starts each new thread with the number of NUMBA_NUM_THREADS. Thus, when
    numba is not imported yet, this function sets that variable, and every thread starts
    with the default. Else, only the importing thread gets it. The variable must then stay:
    numba reads it again at each compile, and stops at a change.
    """
    global _default, _cores
    _cores = _fast_cores()
    text = os.environ.get("PICTOLOGICS_NUM_THREADS")
    numba_variable = "NUMBA_NUM_THREADS" in os.environ
    wanted = _parse(text) if text is not None else 0 if numba_variable else _cores
    if not numba_variable and "numba" not in sys.modules:
        os.environ["NUMBA_NUM_THREADS"] = str(wanted)
        _default = wanted
        return
    import numba

    _default = numba.get_num_threads()
    if 0 < wanted < _default:
        numba.set_num_threads(wanted)
        _default = wanted


def _make_room() -> int:
    """One thread less in the calling thread when its threads fill the fast cores, so that a
    serial task in another thread has a core of its own.

    Returns:
        The number of threads to set again after the task, or 0 for no change.
    """
    import numba

    threads = int(numba.get_num_threads())
    if threads < max(_cores, 2):
        return 0
    numba.set_num_threads(threads - 1)
    return threads


def _parse(text: str) -> int:
    """The number of PICTOLOGICS_NUM_THREADS."""
    if not text.strip().isdigit() or int(text) < 1:
        raise ValueError(
            f"PICTOLOGICS_NUM_THREADS must be a whole number of 1 or more, not {text!r}."
        )
    return int(text)


def _fast_cores() -> int:
    """The default number of threads: the fast cores that this process can use.

    - A Mac: the performance cores (sysctl hw.perflevel0.logicalcpu), for example 10 of the
      14 cores of an M4 Pro. There, 10 threads are as fast as 14 or faster.
    - An ARM chip with more kinds of cores, on Linux: the fastest cores (see `_fastest`).
    - Windows and other computers: all logical CPUs of the process. An Intel chip has many slow
      cores, each about as fast as a second thread of a fast core, so all of them help (not
      measured). Windows gives only a rank for each kind of core, not its speed.
    - A Linux container: at most its CPU limit (see `_cpu_limit`).
    """
    if sys.platform == "darwin" and (cores := _performance_cores()) > 0:
        return cores
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
    else:
        cpus = list(range(os.cpu_count() or 1))
    if platform.machine().startswith(("aarch64", "arm")):
        cpus = _fastest(cpus)
    limit = _cpu_limit()
    return max(1, min(len(cpus), limit)) if limit else len(cpus)


def _performance_cores() -> int:
    """The number of performance cores of a Mac (sysctl hw.perflevel0.logicalcpu), or 0
    when the system has no such key."""
    import ctypes

    count = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(count))
    found = ctypes.CDLL(None).sysctlbyname(
        b"hw.perflevel0.logicalcpu",
        ctypes.byref(count),
        ctypes.byref(size),
        None,
        ctypes.c_size_t(0),
    )
    return count.value if found == 0 else 0


def _fastest(cpus: list[int]) -> list[int]:
    """The k fastest of `cpus` with the largest k * speed, by the cpu_capacity of Linux.

    With equal parts, the slowest of the k cores sets the time, so k * speed is the work
    in a unit of time. All `cpus` stay when a CPU has no cpu_capacity file.
    """
    speeds: dict[int, int] = {}
    for cpu in cpus:
        words = _words(f"/sys/devices/system/cpu/cpu{cpu}/cpu_capacity")
        if not words:
            return cpus
        speeds[cpu] = int(words[0])
    ranked = sorted(speeds.values(), reverse=True)
    k = max(range(1, len(ranked) + 1), key=lambda i: i * ranked[i - 1])
    return [cpu for cpu in cpus if speeds[cpu] >= ranked[k - 1]]


def _cpu_limit() -> int:
    """The CPU limit of a Linux container, rounded up (cgroup v2, then v1), or 0 for none."""
    quota = _words("/sys/fs/cgroup/cpu.max") or (
        _words("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        + _words("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    )
    if len(quota) != 2 or quota[0] in ("max", "-1"):
        return 0
    return math.ceil(int(quota[0]) / int(quota[1]))


@functools.cache
def _physical_memory() -> int:
    """The bytes of memory of this computer: its physical memory, at most the memory limit
    of a Linux container (cgroup v2, then v1). 8 GiB when the system gives no valid number.
    """
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):  # MEMORYSTATUSEX: 2 DWORD, 7 DWORDLONG
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64)
                for name in (
                    "total",
                    "free",
                    "page",
                    "free_page",
                    "virtual",
                    "free_virtual",
                    "extended",
                )
            ]

        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        found = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        total = int(status.total) if found else 0
    else:
        try:
            page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            page = pages = 0
        total = page * pages if page > 0 and pages > 0 else 0
    known = [n for n in (total, _memory_limit()) if n > 0]
    return min(known) if known else 8 << 30


def _memory_limit() -> int:
    """The memory limit of a Linux container in bytes (cgroup v2, then v1), or 0 for none."""
    words = _words("/sys/fs/cgroup/memory.max") or _words(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    return int(words[0]) if words and words[0].isdigit() else 0


def _memory_budget() -> int:
    """The bytes that one large step may use: half of the memory of this computer (see
    _physical_memory), shared by the worker processes that run at the same time."""
    return _physical_memory() // 2 // _memory_shares


def _share_memory(processes: int) -> None:
    """Give this process its share of the memory budget: one of `processes` worker processes
    that run at the same time (the workers of run_batch)."""
    global _memory_shares
    _memory_shares = processes


def _words(path: str) -> list[str]:
    """The words of a small system file, or [] when the file cannot be read."""
    try:
        with open(path) as file:
            return file.read().split()
    except OSError:
        return []
