# pictologics/filters/_passes.py
"""numba kernels of the 1-D passes: twins of correlate1d of scipy that keep its arithmetic
(a float64 line, plane or row buffer, the fused multiply-add where the arm64 build of scipy
contracts, the symmetric, antisymmetric and general tap orders, the line extension of
scipy), and the uniform (mean) passes, which sum each window on its own (see _direct). The
output can be the image (in place). Mode codes as base._PAD_CODES (np.pad names): constant
0, symmetric 1 (scipy reflect), reflect 2 (scipy mirror), edge 3 (scipy nearest), wrap 4.
`cval` is the constant of mode 0."""

import math
from typing import Any

import numpy as np
from llvmlite import ir
from numba import config as numba_config
from numba import jit, types
from numba.extending import intrinsic
from numpy import typing as npt


@intrinsic  # type: ignore
def _fma_intrinsic(typingctx: Any, a: Any, b: Any, c: Any) -> Any:
    """a * b + c of three float64 values with one rounding (llvm.fma), for the kernels."""
    sig = types.float64(types.float64, types.float64, types.float64)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        fma = builder.module.declare_intrinsic("llvm.fma", [ir.DoubleType()] * 3)
        return builder.call(fma, args)

    return sig, codegen


def _fma_python(a: float, b: float, c: float) -> float:
    """The fma of the kernels with the compiler off: the exact rational a * b + c, divided
    once (int / int rounds correctly). A zero or a non-finite result takes the plain form,
    which gives the same value and the same sign."""
    if math.isfinite(a) and math.isfinite(b) and math.isfinite(c):
        an, ad = float(a).as_integer_ratio()
        bn, bd = float(b).as_integer_ratio()
        cn, cd = float(c).as_integer_ratio()
        try:
            r = (an * bn * cd + cn * ad * bd) / (ad * bd * cd)
        except OverflowError:
            r = 0.0
        if r != 0.0:
            return r
    return a * b + c


fma = _fma_python if numba_config.DISABLE_JIT else _fma_intrinsic


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _source_line(t: int, n: int, mode: int) -> int:
    """The line of an axis of n lines that line t of the padded axis reads, as np.pad does
    (`mode`: see base._PAD_CODES); -1 for the constant."""
    if 0 <= t < n:
        return t
    if mode == 0:
        return -1
    if mode == 3:
        return 0 if t < 0 else n - 1
    if mode == 4:
        return t % n
    if mode == 1:
        period = 2 * n
        t = t % period
        return t if t < n else period - 1 - t
    if n == 1:
        return 0
    period = 2 * n - 2
    t = t % period
    return t if t < n else period - t


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _plain_pairs(r: int) -> int:
    """The pairs of a symmetric kernel of r pairs that the compiled loop of scipy adds as
    plain products (its vectorized body of 8 pairs, no contraction): the first (r // 8) * 8
    for r >= 10, none below. The other pairs are fused multiply-adds."""
    return (r // 8) * 8 if r >= 10 else 0


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _taps(
    buf: Any,
    ext: int,
    n: int,
    m: int,
    out_row: Any,
    w: Any,
    sym: int,
    mode: int,
    cval: float,
    acc: Any,
    pair: Any,
) -> None:
    """out_row[k] = the correlation of the extended lines buf[., k] at line `ext` in the
    order of scipy (the center tap first, then the pairs of a symmetric kernel from the
    outside in; the last tap first for a general kernel). acc and pair are float64 work
    rows of m."""
    nw = w.shape[0]
    r = nw // 2
    size2 = nw - r - 1
    if sym != 0:
        wc = w[r]
        for k in range(m):
            acc[k] = buf[ext, k] * wc
        plain = _plain_pairs(r)
        for jj in range(-r, 0):
            sl = _source_line(ext + jj, n, mode)
            sr = _source_line(ext - jj, n, mode)
            wj = w[r + jj]
            if sl < 0 and sr < 0:
                v = cval + cval if sym > 0 else cval - cval
                for k in range(m):
                    pair[k] = v
            elif sl < 0:
                if sym > 0:
                    for k in range(m):
                        pair[k] = cval + buf[sr, k]
                else:
                    for k in range(m):
                        pair[k] = cval - buf[sr, k]
            elif sr < 0:
                if sym > 0:
                    for k in range(m):
                        pair[k] = buf[sl, k] + cval
                else:
                    for k in range(m):
                        pair[k] = buf[sl, k] - cval
            else:
                if sym > 0:
                    for k in range(m):
                        pair[k] = buf[sl, k] + buf[sr, k]
                else:
                    for k in range(m):
                        pair[k] = buf[sl, k] - buf[sr, k]
            if jj + r < plain:
                for k in range(m):
                    acc[k] = acc[k] + pair[k] * wj
            else:
                for k in range(m):
                    acc[k] = fma(pair[k], wj, acc[k])
    else:
        s = _source_line(ext + size2, n, mode)
        wl = w[r + size2]
        if s < 0:
            for k in range(m):
                acc[k] = cval * wl
        else:
            for k in range(m):
                acc[k] = buf[s, k] * wl
        for jj in range(-r, size2):
            s = _source_line(ext + jj, n, mode)
            wj = w[r + jj]
            if s < 0:
                for k in range(m):
                    acc[k] = fma(cval, wj, acc[k])
            else:
                for k in range(m):
                    acc[k] = fma(buf[s, k], wj, acc[k])
    for k in range(m):
        out_row[k] = acc[k]


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def corr_axis0(
    image: Any, out: Any, w: Any, mode: int, cval: float, sym: int, a: int, b: int
) -> None:
    """out[:, a:b, :] = correlate1d(image, w, axis=0, mode, cval)[:, a:b, :] (out can be
    image): one (n0 x n2) float64 buffer per row j."""
    n0, n1, n2 = image.shape
    buf = np.empty((n0, n2), dtype=np.float64)
    acc = np.empty(n2, dtype=np.float64)
    pair = np.empty(n2, dtype=np.float64)
    for j in range(a, b):
        for i in range(n0):
            for k in range(n2):
                buf[i, k] = image[i, j, k]
        for i in range(n0):
            _taps(buf, i, n0, n2, out[i, j], w, sym, mode, cval, acc, pair)


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def corr_axis1(
    image: Any, out: Any, w: Any, mode: int, cval: float, sym: int, first: int, last: int
) -> None:
    """out[first:last] = correlate1d(image[first:last], w, axis=1, mode, cval) (out can be
    image): one (n1 x n2) float64 buffer per plane i."""
    n0, n1, n2 = image.shape
    buf = np.empty((n1, n2), dtype=np.float64)
    acc = np.empty(n2, dtype=np.float64)
    pair = np.empty(n2, dtype=np.float64)
    for i in range(first, last):
        for j in range(n1):
            for k in range(n2):
                buf[j, k] = image[i, j, k]
        for j in range(n1):
            _taps(buf, j, n1, n2, out[i, j], w, sym, mode, cval, acc, pair)


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def corr_axis2(
    image: Any, out: Any, w: Any, mode: int, cval: float, sym: int, first: int, last: int
) -> None:
    """out[first:last] = correlate1d(image[first:last], w, axis=2, mode, cval) (out can be
    image). Each line is extended into a buffer (index r + x), then the taps run over x."""
    n0, n1, n2 = image.shape
    nw = w.shape[0]
    r = nw // 2
    size2 = nw - r - 1
    line = np.empty(n2 + nw, dtype=np.float64)
    acc = np.empty(n2, dtype=np.float64)
    for i in range(first, last):
        for j in range(n1):
            for x in range(-r, n2 + size2):
                s = _source_line(x, n2, mode)
                line[x + r] = cval if s < 0 else image[i, j, s]
            if sym != 0:
                wc = w[r]
                for x in range(n2):
                    acc[x] = line[x + r] * wc
                plain = _plain_pairs(r)
                for jj in range(-r, 0):
                    wj = w[r + jj]
                    if jj + r < plain:
                        if sym > 0:
                            for x in range(n2):
                                acc[x] = acc[x] + (line[x + r + jj] + line[x + r - jj]) * wj
                        else:
                            for x in range(n2):
                                acc[x] = acc[x] + (line[x + r + jj] - line[x + r - jj]) * wj
                    elif sym > 0:
                        for x in range(n2):
                            acc[x] = fma(line[x + r + jj] + line[x + r - jj], wj, acc[x])
                    else:
                        for x in range(n2):
                            acc[x] = fma(line[x + r + jj] - line[x + r - jj], wj, acc[x])
            else:
                wl = w[r + size2]
                for x in range(n2):
                    acc[x] = line[x + r + size2] * wl
                for jj in range(-r, size2):
                    wj = w[r + jj]
                    for x in range(n2):
                        acc[x] = fma(line[x + r + jj], wj, acc[x])
            for x in range(n2):
                out[i, j, x] = acc[x]


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _direct(
    buf: Any, n: int, m: int, size: int, mode: int, cval: float, out_rows: Any, tmp: Any
) -> None:
    """out_rows[t, k] = the mean of the extended window around t of the lines buf[., k],
    each window summed on its own from its first value. So a value does not depend on where
    the line starts: a crop of the image gives the same values. For float32 values the
    window sum is exact in double, so the mean is the exact mean rounded once."""
    r = size // 2
    fs = float(size)
    for t in range(n):
        for k in range(m):
            tmp[k] = 0.0
        for jj in range(size):
            s = _source_line(t + jj - r, n, mode)
            if s < 0:
                for k in range(m):
                    tmp[k] += cval
            else:
                for k in range(m):
                    tmp[k] += buf[s, k]
        for k in range(m):
            out_rows[t, k] = tmp[k] / fs


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _block_of(size: int) -> int:
    """The block length of the two-level window sum: 5 or 3 when the size is a multiple
    from 10 (9) on, else 0 (the plain sum of _direct)."""
    if size >= 10 and size % 5 == 0:
        return 5
    if size >= 9 and size % 3 == 0:
        return 3
    return 0


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _two_level(
    buf: Any,
    n: int,
    m: int,
    size: int,
    mode: int,
    cval: float,
    out_rows: Any,
    tmp: Any,
    blocks: Any,
    block: int,
) -> None:
    """out_rows[t, k] = the mean of the extended window around t as the sum of its
    size // block block sums. Each block sum is summed on its own (blocks[q, k] = the sum of
    the extended lines q - r .. q - r + block - 1), so a value does not depend on where the
    line starts (as _direct, with fewer adds)."""
    r = size // 2
    fs = float(size)
    nb = size // block
    for q in range(n + size - block):
        for k in range(m):
            tmp[k] = 0.0
        for jj in range(block):
            s = _source_line(q + jj - r, n, mode)
            if s < 0:
                for k in range(m):
                    tmp[k] += cval
            else:
                for k in range(m):
                    tmp[k] += buf[s, k]
        for k in range(m):
            blocks[q, k] = tmp[k]
    for t in range(n):
        for k in range(m):
            tmp[k] = blocks[t, k]
        for b in range(1, nb):
            for k in range(m):
                tmp[k] += blocks[t + b * block, k]
        for k in range(m):
            out_rows[t, k] = tmp[k] / fs


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def unif_axis0(image: Any, out: Any, size: int, mode: int, cval: float, a: int, b: int) -> None:
    """out[:, a:b, :] = the mean of each window of `size` along axis 0 (uniform_filter1d with
    the window sums of _direct and _two_level) for the rows a:b (out can be image)."""
    n0, n1, n2 = image.shape
    buf = np.empty((n0, n2), dtype=np.float64)
    rows = np.empty((n0, n2), dtype=np.float64)
    tmp = np.empty(n2, dtype=np.float64)
    block = _block_of(size)
    blocks = np.empty((n0 + size, n2), dtype=np.float64)
    for j in range(a, b):
        for i in range(n0):
            for k in range(n2):
                buf[i, k] = image[i, j, k]
        if block > 0:
            _two_level(buf, n0, n2, size, mode, cval, rows, tmp, blocks, block)
        else:
            _direct(buf, n0, n2, size, mode, cval, rows, tmp)
        for i in range(n0):
            for k in range(n2):
                out[i, j, k] = rows[i, k]


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def unif_axis1(
    image: Any, out: Any, size: int, mode: int, cval: float, first: int, last: int
) -> None:
    """As unif_axis0, along axis 1 for the planes first:last."""
    n0, n1, n2 = image.shape
    buf = np.empty((n1, n2), dtype=np.float64)
    rows = np.empty((n1, n2), dtype=np.float64)
    tmp = np.empty(n2, dtype=np.float64)
    block = _block_of(size)
    blocks = np.empty((n1 + size, n2), dtype=np.float64)
    for i in range(first, last):
        for j in range(n1):
            for k in range(n2):
                buf[j, k] = image[i, j, k]
        if block > 0:
            _two_level(buf, n1, n2, size, mode, cval, rows, tmp, blocks, block)
        else:
            _direct(buf, n1, n2, size, mode, cval, rows, tmp)
        for j in range(n1):
            for k in range(n2):
                out[i, j, k] = rows[j, k]


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def unif_axis2(
    image: Any, out: Any, size: int, mode: int, cval: float, first: int, last: int
) -> None:
    """As unif_axis0, along axis 2 for the planes first:last: each line is extended into a
    buffer, then summed in windows (or in blocks, then in windows of blocks)."""
    n0, n1, n2 = image.shape
    r = size // 2
    size2 = size - r - 1
    line = np.empty(n2 + size, dtype=np.float64)
    acc = np.empty(n2, dtype=np.float64)
    bsum = np.empty(n2 + size, dtype=np.float64)
    block = _block_of(size)
    nb = size // block if block > 0 else 0
    fs = float(size)
    for i in range(first, last):
        for j in range(n1):
            for x in range(-r, n2 + size2):
                s = _source_line(x, n2, mode)
                line[x + r] = cval if s < 0 else image[i, j, s]
            if block == 0:
                for x in range(n2):
                    acc[x] = 0.0
                for t in range(size):
                    for x in range(n2):
                        acc[x] += line[x + t]
                for x in range(n2):
                    out[i, j, x] = acc[x] / fs
            else:
                for q in range(n2 + size - block):
                    v = line[q]
                    for jj in range(1, block):
                        v += line[q + jj]
                    bsum[q] = v
                for x in range(n2):
                    v = bsum[x]
                    for b in range(1, nb):
                        v += bsum[x + b * block]
                    out[i, j, x] = v / fs


def symmetry(w: npt.NDArray[np.float64]) -> int:
    """1 for a symmetric kernel of odd length, -1 for an antisymmetric one, else 0 (the
    general tap order): the test of scipy, within one float64 unit."""
    n = w.size
    if n % 2 == 0:
        return 0
    r = n // 2
    eps = np.finfo(np.float64).eps
    if np.all(np.abs(w[r + 1 :] - w[:r][::-1]) <= eps):
        return 1
    if np.all(np.abs(w[r + 1 :] + w[:r][::-1]) <= eps):
        return -1
    return 0
