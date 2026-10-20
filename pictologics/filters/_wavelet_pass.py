# pictologics/filters/_wavelet_pass.py
"""scipy.ndimage.convolve1d of a float32 3-D row-order image as numba kernels with the
arithmetic of scipy (each line in double, the last tap first, then the taps in order as
fused multiply-adds, one rounding to float32) and the line extension of scipy. Every wavelet
kernel has an even length, so scipy takes its general branch. The zero taps of an a trous
kernel (level 2 and above) are skipped: a NaN or inf voxel then spreads over the non-zero
taps only (scipy adds x * 0 for a zero tap).

- axis 0: no strided gather. The pass runs in blocks of rows of axis 1; a ring of the last
  n_taps extended planes (doubles) feeds a plane accumulator, so each input plane is read
  once, in memory order, and the ring stays in the cache.
- axis 1: a ring of rows for each plane.
- axis 2: the line in a double buffer; only the edge values take the extension index.
- The extension indices come from one table (`_src_table`).
- In place (out is x) works on every axis: the extended values of the tail (which read
  planes or rows already written) are copied before the loop.
"""

from typing import Any, Optional

import numpy as np
from numba import get_num_threads, jit
from numpy import typing as npt
from scipy.ndimage import convolve1d

from ._passes import _source_line, fma
from .base import _PAD_CODES, _PAD_MODES, _SLAB_MIN_SIZE, _slab_pass, _slab_pool

_BLOCK_ROWS = 32  # rows of axis 1 in one block of the axis-0 pass (ring in the cache)


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _src_table(ext: int, before: int, n: int, code: int) -> npt.NDArray[np.int64]:
    """The line of the axis (or -1 for the constant) that each of the `ext` extended
    lines reads."""
    src = np.empty(ext, dtype=np.int64)
    for t in range(ext):
        src[t] = _source_line(t - before, n, code)
    return src


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _pass_axis0(x: Any, out: Any, fw: Any, before: int, code: int, first: int, last: int) -> None:
    """out[:, first:last, :] = the pass along axis 0 of x[:, first:last, :] (out can be x)."""
    n0, n1, n2 = x.shape
    n_taps = fw.shape[0]
    ext = n0 + n_taps - 1
    src = _src_table(ext, before, n0, code)
    tail0 = n0 + before  # extended planes from here on can read planes already written
    n_tail = ext - tail0
    end = fw[n_taps - 1]
    for o0 in range(first, last, _BLOCK_ROWS):
        o1 = min(o0 + _BLOCK_ROWS, last)
        nb = o1 - o0
        ring = np.empty((n_taps, nb, n2), dtype=np.float64)
        acc = np.empty((nb, n2), dtype=np.float64)
        tail = np.empty((max(n_tail, 0), nb, n2), dtype=np.float64)
        for t in range(tail0, ext):
            s = src[t]
            for j in range(nb):
                for k in range(n2):
                    tail[t - tail0, j, k] = 0.0 if s < 0 else np.float64(x[s, o0 + j, k])
        # the ring holds the extended planes t .. t + n_taps - 1 of output ll = t
        for t in range(n_taps - 1):
            s = src[t]
            r = ring[t]
            for j in range(nb):
                for k in range(n2):
                    r[j, k] = 0.0 if s < 0 else np.float64(x[s, o0 + j, k])
        for ll in range(n0):
            t = ll + n_taps - 1
            r = ring[t % n_taps]
            if t >= tail0:
                tp = tail[t - tail0]
                for j in range(nb):
                    for k in range(n2):
                        r[j, k] = tp[j, k]
            else:
                s = src[t]
                for j in range(nb):
                    for k in range(n2):
                        r[j, k] = 0.0 if s < 0 else np.float64(x[s, o0 + j, k])
            # scipy: tmp = line[size2] * fw[size2], then jj = 0 .. n_taps - 2 in order
            for j in range(nb):
                for k in range(n2):
                    acc[j, k] = r[j, k] * end
            for m in range(n_taps - 1):
                w = fw[m]
                if w != 0.0:  # a zero tap (a trous) adds x * 0 to the sum: skipped
                    rm = ring[(ll + m) % n_taps]
                    for j in range(nb):
                        for k in range(n2):
                            acc[j, k] = fma(rm[j, k], w, acc[j, k])
            for j in range(nb):
                for k in range(n2):
                    out[ll, o0 + j, k] = acc[j, k]


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _pass_axis1(x: Any, out: Any, fw: Any, before: int, code: int, first: int, last: int) -> None:
    """out[first:last] = the pass along axis 1 of x[first:last] (out can be x)."""
    n0, n1, n2 = x.shape
    n_taps = fw.shape[0]
    ext = n1 + n_taps - 1
    src = _src_table(ext, before, n1, code)
    tail0 = n1 + before
    n_tail = ext - tail0
    end = fw[n_taps - 1]
    ring = np.empty((n_taps, n2), dtype=np.float64)
    acc = np.empty(n2, dtype=np.float64)
    tail = np.empty((max(n_tail, 0), n2), dtype=np.float64)
    for o in range(first, last):
        for t in range(tail0, ext):
            s = src[t]
            for k in range(n2):
                tail[t - tail0, k] = 0.0 if s < 0 else np.float64(x[o, s, k])
        for t in range(n_taps - 1):
            s = src[t]
            r = ring[t]
            for k in range(n2):
                r[k] = 0.0 if s < 0 else np.float64(x[o, s, k])
        for ll in range(n1):
            t = ll + n_taps - 1
            r = ring[t % n_taps]
            if t >= tail0:
                tp = tail[t - tail0]
                for k in range(n2):
                    r[k] = tp[k]
            else:
                s = src[t]
                for k in range(n2):
                    r[k] = 0.0 if s < 0 else np.float64(x[o, s, k])
            for k in range(n2):
                acc[k] = r[k] * end
            for m in range(n_taps - 1):
                w = fw[m]
                if w != 0.0:
                    rm = ring[(ll + m) % n_taps]
                    for k in range(n2):
                        acc[k] = fma(rm[k], w, acc[k])
            for k in range(n2):
                out[o, ll, k] = acc[k]


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _pass_axis2(x: Any, out: Any, fw: Any, before: int, code: int, first: int, last: int) -> None:
    """out[first:last] = the pass along axis 2 of x[first:last] (out can be x)."""
    n0, n1, n2 = x.shape
    n_taps = fw.shape[0]
    ext = n2 + n_taps - 1
    src = _src_table(ext, before, n2, code)
    after = n_taps - 1 - before
    end = fw[n_taps - 1]
    buf = np.empty(ext, dtype=np.float64)
    acc = np.empty(n2, dtype=np.float64)
    for i in range(first, last):
        for j in range(n1):
            line = x[i, j]
            for t in range(before):
                s = src[t]
                buf[t] = 0.0 if s < 0 else np.float64(line[s])
            for t in range(n2):
                buf[before + t] = np.float64(line[t])
            for t in range(after):
                s = src[before + n2 + t]
                buf[before + n2 + t] = 0.0 if s < 0 else np.float64(line[s])
            for ll in range(n2):
                acc[ll] = buf[ll + n_taps - 1] * end
            for m in range(n_taps - 1):
                w = fw[m]
                if w != 0.0:
                    for ll in range(n2):
                        acc[ll] = fma(buf[ll + m], w, acc[ll])
            for ll in range(n2):
                out[i, j, ll] = acc[ll]


def wavelet_pass(
    image: npt.NDArray[Any],
    axis: int,
    output: Optional[npt.NDArray[Any]],
    threads: Optional[int],
    weights: npt.NDArray[Any],
    mode: str,
    origin: int,
) -> npt.NDArray[Any]:
    """convolve1d(image, weights, axis, output, mode, origin) of a float32 3-D row-order
    image on slabs in the slab threads, in the numba kernels (the bits of scipy, each line
    summed in double); the scipy pass of _slab_pass for other arrays, a line shorter than
    the kernel or a mode without a code."""
    n = len(weights)
    if (
        image.ndim != 3
        or image.dtype != np.float32
        or not image.flags.c_contiguous
        or (output is not None and not output.flags.c_contiguous)
        or image.shape[axis] < n
        or mode not in _PAD_MODES
    ):
        return _slab_pass(
            convolve1d, image, axis, output, threads, weights=weights, mode=mode, origin=origin
        )
    if output is None:
        output = np.empty(image.shape, dtype=image.dtype)
    fw = np.ascontiguousarray(np.asarray(weights, dtype=np.float64)[::-1])
    # scipy: size1 + origin_c, with origin_c = -origin (one less for an even length)
    before = n // 2 - origin - (0 if n & 1 else 1)
    code = _PAD_CODES[_PAD_MODES[mode]]
    kernel = (_pass_axis0, _pass_axis1, _pass_axis2)[axis]
    split_len = image.shape[1] if axis == 0 else image.shape[0]
    threads = get_num_threads() if threads is None else threads
    parts = min(threads, split_len)
    if image.size < _SLAB_MIN_SIZE or parts < 2:
        kernel(image, output, fw, before, code, 0, split_len)
        return output
    edges = np.linspace(0, split_len, parts + 1).astype(int)

    def run(k: int) -> None:
        kernel(image, output, fw, before, code, int(edges[k]), int(edges[k + 1]))

    list(_slab_pool().map(run, range(parts)))
    return output
