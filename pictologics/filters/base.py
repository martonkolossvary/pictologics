# pictologics/filters/base.py
"""Base classes and utilities for IBSI 2 filter implementations."""

import os
import threading
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import scipy.fft
from numba import config as numba_config
from numba import get_num_threads, jit
from numpy import typing as npt
from scipy.ndimage import convolve1d, gaussian_filter1d, uniform_filter1d


class BoundaryCondition(Enum):
    """
    IBSI 2 boundary conditions for image padding (GBYQ).

    Maps to scipy.ndimage mode parameter values.

    Example:
        ```python
        from pictologics.filters import BoundaryCondition

        boundary = BoundaryCondition.MIRROR
        print(boundary.value)
        # "reflect"

        # By name (the filters take the names, in any case)
        boundary = BoundaryCondition["MIRROR"]
        # By the scipy mode string of get_scipy_mode
        boundary = BoundaryCondition("reflect")
        ```
    """

    ZERO = "constant"  # Constant value padding (Z3VE): 0, or a filter's padding_value
    NEAREST = "nearest"  # Nearest value padding (SIJG)
    PERIODIC = "wrap"  # Periodic/wrap padding (Z7YO)
    MIRROR = "reflect"  # Mirror/symmetric padding (ZDTV)
    CONSTANT = "constant"  # Another name of ZERO


@dataclass
class FilterResult:
    """Container for filter response maps and metadata.

    Example:
        ```python
        import numpy as np
        from pictologics.filters.base import FilterResult

        result = FilterResult(
            response_map=np.zeros((4, 4, 4), dtype=np.float32),
            filter_name="mean",
            filter_params={"support": 3},
        )
        print(result.shape, result.dtype)
        # (4, 4, 4) float32
        ```
    """

    response_map: npt.NDArray[np.floating[Any]]
    filter_name: str
    filter_params: Dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the response map."""
        return self.response_map.shape  # type: ignore[no-any-return]

    @property
    def dtype(self) -> np.dtype[Any]:
        """Data type of the response map."""
        return self.response_map.dtype  # type: ignore[no-any-return]


_Array = TypeVar("_Array", bound=npt.NDArray[Any])

# Byte limit of each transfer-function cache: large enough for the few shapes of one
# image (rotations, orders, levels), small enough that a cohort of images with
# different shapes does not keep a full volume for each of them.
_TRANSFER_CACHE_BYTES = 2 << 30


def _whole_number(value: Any) -> bool:
    """An int, or a float with a whole value (32.0 from a YAML or JSON file)."""
    return (isinstance(value, (int, np.integer)) and not isinstance(value, bool)) or (
        isinstance(value, (float, np.floating)) and float(value).is_integer()
    )


def cache_by_bytes(
    max_bytes: int,
) -> Callable[[Callable[..., _Array]], Callable[..., _Array]]:
    """Least-recently-used cache of array results, bounded by their total bytes.

    The newest result always stays cached, even when it alone is larger than
    `max_bytes`. Arguments must be hashable. `cache_clear()` drops every result. A lock
    makes the cache safe for threads: a thread waits while another builds a result.
    """

    def decorate(func: Callable[..., _Array]) -> Callable[..., _Array]:
        cache: OrderedDict[Any, _Array] = OrderedDict()
        total = 0
        lock = threading.Lock()

        @wraps(func)
        def cached(*args: Any) -> _Array:
            nonlocal total
            with lock:
                if args in cache:
                    cache.move_to_end(args)
                    return cache[args]
                value = func(*args)
                cache[args] = value
                total += value.nbytes
                while total > max_bytes and len(cache) > 1:
                    total -= cache.popitem(last=False)[1].nbytes
                return value

        def cache_clear() -> None:
            nonlocal total
            with lock:
                cache.clear()
                total = 0

        cached.cache_clear = cache_clear  # type: ignore[attr-defined]
        return cached

    return decorate


def _slabs(
    shape: Tuple[int, ...], elements: int = 1 << 21, minimum: int = 1 << 23
) -> list[Tuple[int, int]]:
    """(start, stop) ranges along the first axis with about `elements` values each, for
    an array of more than `minimum` values; one range for a smaller array."""
    if int(np.prod(shape)) <= minimum:
        return [(0, shape[0])]
    step = max(1, elements // max(1, int(np.prod(shape[1:]))))
    return [(start, min(start + step, shape[0])) for start in range(0, shape[0], step)]


_PRODUCT_PART = 1 << 17

# Transfer tables of at least this many cells (16 MB for Riesz) keep half of their rows
# (see _times_mirrored); a smaller table stays whole, as the split product costs more.
_HALF_TABLE_MIN = 1 << 20


def _kept_rows(rows: int, half: int, cells: int) -> int:
    """The rows on axis 0 that a transfer table of `cells` cells keeps: `half` of its
    `rows` for a large table, else all."""
    return half if cells >= _HALF_TABLE_MIN else rows


def _times_mirrored(
    spectrum: npt.NDArray[np.complexfloating[Any, Any]],
    half: npt.NDArray[Any],
    offset: int,
    sign: int,
) -> npt.NDArray[np.complexfloating[Any, Any]]:
    """`spectrum` times a transfer table that mirrors on axis 0, from `half`, the rows
    that the table keeps: row k >= len(half) of the full table equals row offset - k
    times `sign`. The products are those of the full table, as the mirrored rows are
    exact copies (or exact negatives).

    The product is written into `spectrum` when that keeps its type: a float64 image has
    a complex128 spectrum. A complex64 spectrum (float32 image) times a float64 table is
    complex128, so it gets a new array, as before. Large spectra run on slabs in threads.
    """
    dtype = np.result_type(spectrum, half)
    out = spectrum if dtype == spectrum.dtype else np.empty(spectrum.shape, dtype)
    n, h = spectrum.shape[0], half.shape[0]
    # Rows h..n-1 read rows offset - h down to offset - n + 1 (none for a whole table)
    mirror = half[offset - n + 1 : offset - h + 1][::-1]
    if spectrum.size < _SLAB_MIN_SIZE:  # small: the two products in this thread
        np.multiply(spectrum[:h], half, out=out[:h])
        np.multiply(spectrum[h:], mirror, out=out[h:])
        if sign < 0:
            np.negative(out[h:], out=out[h:])
        return out
    threads = get_num_threads()
    tasks = []
    for rows, table, row_sign in ((slice(0, h), half, 1), (slice(h, n), mirror, sign)):
        source, target = spectrum[rows], out[rows]
        # Parts of at least _PRODUCT_PART values: smaller ones cost more to hand out
        n_parts = min(threads, target.shape[0], max(1, target.size // _PRODUCT_PART))
        edges = np.linspace(0, target.shape[0], n_parts + 1).astype(int)
        tasks += [
            (source[a:b], table[a:b], target[a:b], row_sign)
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]

    def run(task: Tuple[Any, Any, Any, int]) -> None:
        source, table, target, row_sign = task
        np.multiply(source, table, out=target)
        if row_sign < 0:
            np.negative(target, out=target)

    list(_slab_pool().map(run, tasks))
    return out


def ensure_float32(
    image: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Ensure image is at least 32-bit floating point precision.

    Per IBSI 2: "The phantom data need to be converted from an integer
    data type to at least 32 bit floating point precision, prior to filtering."

    The two rules of the filters: the FFT filters (LoG, Simoncelli, Riesz), Gabor and
    the normalized convolution of a source mask take this function, so a float64 image
    stays float64. The separable filters (mean, Gaussian, Laws, wavelets) take
    `as_float32`, so their passes run in float32.

    Args:
        image: Input image array

    Returns:
        Image as float32 (or higher precision if already float64)
    """
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float32) if image.dtype == np.float16 else image
    return image.astype(np.float32)


def as_float32(image: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """The image as float32: the image itself when it is float32, else a new float32 copy
    (on slabs in threads for a large array). The separable filters run their passes on
    it: a float32 pass moves half the bytes of a float64 pass, and scipy still sums each
    line in double. A float64 image thus changes by the float32 rounding of the passes
    (see `ensure_float32` for the other rule)."""
    if image.dtype == np.float32:
        return image
    return _float32_cut(image, None)


def get_scipy_mode(boundary: BoundaryCondition) -> str:
    """Convert BoundaryCondition to scipy.ndimage mode string."""
    return boundary.value


def resolve_boundary(boundary: Union[BoundaryCondition, str]) -> BoundaryCondition:
    """
    Resolve a boundary condition from a `BoundaryCondition` instance or a
    case-insensitive member name.

    Args:
        boundary: Either a `BoundaryCondition` member, or one of the
            (case-insensitive) names "zero", "constant", "nearest", "periodic", "mirror".

    Returns:
        The resolved `BoundaryCondition`.

    Raises:
        ValueError: If `boundary` is a string that does not match any
            `BoundaryCondition` member name.
    """
    if isinstance(boundary, BoundaryCondition):
        return boundary
    try:
        return BoundaryCondition[str(boundary).upper()]
    except KeyError:
        valid = ", ".join(name.lower() for name in BoundaryCondition.__members__)
        raise ValueError(
            f"Unknown boundary condition: {boundary!r}. Valid values: {valid}"
        ) from None


# Boundary conditions handled by manual np.pad, keyed by scipy-style pad mode name.
# PERIODIC is deliberately excluded: it is handled by the fast path in
# `_apply_with_boundary_padding` (calling `func` directly, with no padding at all),
# so it never needs a numpy pad-mode lookup.
_NUMPY_PAD_MODE_MAP = {
    BoundaryCondition.ZERO: "constant",
    BoundaryCondition.NEAREST: "edge",
    BoundaryCondition.MIRROR: "symmetric",
}


def _padding_value_problem(boundary: BoundaryCondition, padding_value: float) -> Optional[str]:
    """Why a padding value does not fit the boundary (a value other than 0 needs constant
    value padding), or None."""
    if padding_value and boundary is not BoundaryCondition.ZERO:
        return (
            f"padding_value {padding_value!r} needs the constant (zero) boundary, not "
            f"'{boundary.name.lower()}'"
        )
    return None


# A filter response, or (response, output valid mask) with a source mask
_FilterOutput = Union[
    npt.NDArray[np.floating[Any]], Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]
]


def _constant_padded(
    func: Callable[..., Any],
    image: npt.NDArray[np.floating[Any]],
    source_mask: Optional[npt.NDArray[np.bool_]],
    value: float,
    reach: Union[int, Tuple[int, ...]],
    **kwargs: Any,
) -> _FilterOutput:
    """The response of the filter `func` to `image` padded with the constant `value`
    (IBSI 2 constant value padding, Z3VE): the image is padded by `reach` voxels on each
    side of every axis (valid voxels in a source mask), filtered, and cut back. The filter
    reads at most `reach` voxels on each side, so its own boundary does not reach the
    image. A (response, valid mask) result is cut in both parts."""
    widths = (reach,) * image.ndim if isinstance(reach, int) else tuple(reach)
    pads = [(w, w) for w in widths]
    padded = _padded(image, pads, "constant", value)
    if source_mask is not None:
        kwargs["source_mask"] = np.pad(source_mask, pads, mode="constant", constant_values=True)
    result = func(padded, **kwargs)
    cut = tuple(slice(w, w + s) for w, s in zip(widths, image.shape, strict=True))
    if isinstance(result, tuple):
        return np.ascontiguousarray(result[0][cut]), np.ascontiguousarray(result[1][cut])
    return cast(npt.NDArray[np.floating[Any]], np.ascontiguousarray(result[cut]))


def _apply_with_boundary_padding(
    func: Callable[..., npt.NDArray[np.floating[Any]]],
    image: npt.NDArray[np.floating[Any]],
    boundary: BoundaryCondition,
    pad_width: Union[int, Tuple[int, ...]],
    padding_value: float = 0.0,
    **kwargs: Any,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply an FFT-based filter honouring a boundary condition via pad-filter-crop.

    FFT-based filters (Simoncelli, Riesz) transform the whole array at once and
    are therefore inherently periodic. To approximate any other boundary
    condition, the image is padded using the requested condition, `func` is run
    on the padded (still periodically-transformed) array, and the result is
    cropped back to the original shape so only the region away from the
    artificial periodic wrap of the padded array is returned.

    Args:
        func: FFT-based filter to apply; called as `func(array, crop, **kwargs)`. It
            returns its response on `array` cut to `crop` (None: the whole array), so
            that the cut is part of its last copy and the padded response is freed
            (see `_float32_cut`).
        image: Input image array.
        boundary: Requested boundary condition. For `BoundaryCondition.PERIODIC`,
            `func` is called directly on `image` (crop None) with no padding at all —
            this is byte-identical to calling `func` without any boundary handling,
            i.e. today's (pre-boundary-contract) behaviour.
        pad_width: Padding added on both sides of every axis: either a single int
            (same width for every axis) or a tuple with one width per axis of
            `image`. Each axis's width is capped at that axis's own length, so
            padding never blows up the array by more than 3x even for small inputs.
        padding_value: The constant of the ZERO (constant value) boundary.
        **kwargs: Additional keyword arguments forwarded to `func`.

    Returns:
        The output of `func`, cut (if padded) to match `image`'s shape.
    """
    if boundary is BoundaryCondition.PERIODIC:
        return func(image, None, **kwargs)

    pad_mode = _NUMPY_PAD_MODE_MAP[boundary]
    widths = (pad_width,) * image.ndim if isinstance(pad_width, int) else tuple(pad_width)
    widths = tuple(min(w, s) for w, s in zip(widths, image.shape, strict=True))

    padded = _padded(image, [(w, w) for w in widths], pad_mode, padding_value)
    crop = tuple(slice(w, w + s) for w, s in zip(widths, image.shape, strict=True))
    return func(padded, crop, **kwargs)


# From this many values on, `_float32_cut` copies on slabs in threads. Measured: a
# copy of 262,144 values is faster in one thread than with the start of the threads.
_PARALLEL_COPY_MIN = 1 << 21


def _float32_cut(
    response: npt.NDArray[Any], crop: Optional[Tuple[slice, ...]]
) -> npt.NDArray[np.float32]:
    """`response` cut to `crop` (None: the whole array) as a new C-ordered float32 array,
    so a padded response is not kept. Large arrays copy on slabs in threads."""
    part = response if crop is None else response[crop]
    if part.size < _PARALLEL_COPY_MIN:
        return np.array(part, dtype=np.float32, order="C")
    output = np.empty(part.shape, dtype=np.float32)
    _slab_pass(_copy_pass, part, part.ndim - 1, output)
    return output


def _copy_pass(image: npt.NDArray[Any], axis: int, output: npt.NDArray[Any]) -> None:
    """`output[...] = image` (with a cast) in the form of a `_slab_pass` filter."""
    np.copyto(output, image, casting="same_kind")


# ==============================================================================
# Threaded scipy filters
# ==============================================================================

# From this many voxels on, the 1-D passes of the scipy filters run on slabs in threads
# (scipy frees the GIL in its filter loops). Each line of a pass lies in one slab, so the
# values are those of one scipy call.
_SLAB_MIN_SIZE = 1 << 18

# The threads of _slab_pass, made on first use (a new pool for each pass costs about
# 0.3 ms). A forked process makes its own: the threads of the parent are not in it.
_SLAB_POOL: list[ThreadPoolExecutor] = []
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_SLAB_POOL.clear)


def _slab_pool() -> ThreadPoolExecutor:
    """The shared thread pool of `_slab_pass`, with numba's largest thread count."""
    if not _SLAB_POOL:
        _SLAB_POOL.append(ThreadPoolExecutor(max_workers=numba_config.NUMBA_NUM_THREADS))
    return _SLAB_POOL[0]


_Result = TypeVar("_Result")


def _ordered_map(
    function: Callable[[Any], _Result], items: Iterable[Any], workers: int
) -> Iterator[_Result]:
    """`map(function, items)` with at most `workers` calls at a time in a thread pool. The
    results come in item order, each as soon as it and the ones before it are ready."""
    if workers < 2:
        yield from map(function, items)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[Future[_Result]] = deque()
        for item in items:
            pending.append(executor.submit(function, item))
            if len(pending) == workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _slab_pass(
    function: Callable[..., Any],
    image: npt.NDArray[Any],
    axis: int,
    output: Optional[npt.NDArray[Any]],
    threads: Optional[int] = None,
    **kwargs: Any,
) -> npt.NDArray[Any]:
    """`function(image, axis=axis, output=output, **kwargs)`, a 1-D scipy filter along `axis`, on
    slabs of the longest other axis in `threads` threads (default: numba's thread count;
    1 inside a pool that runs several filters at once). A None `output` is a new array of
    the image type, as scipy makes it; `output` may be `image` itself (scipy copies each
    line). The filter writes every value, so a new output needs no zero fill (a zero
    fill of reused memory runs in one thread)."""
    if output is None:
        output = np.empty(image.shape, dtype=image.dtype)
    if image.size < _SLAB_MIN_SIZE:  # checked first: small images pay no set-up
        function(image, axis=axis, output=output, **kwargs)
        return output
    threads = get_num_threads() if threads is None else threads
    others = [a for a in range(image.ndim) if a != axis]
    if threads < 2 or not others:
        function(image, axis=axis, output=output, **kwargs)
        return output
    split = max(others, key=lambda a: image.shape[a])
    edges = np.linspace(0, image.shape[split], min(threads, image.shape[split]) + 1).astype(int)

    def run(k: int) -> None:
        part = tuple(
            slice(edges[k], edges[k + 1]) if a == split else slice(None) for a in range(image.ndim)
        )
        function(image[part], axis=axis, output=output[part], **kwargs)

    list(_slab_pool().map(run, range(edges.size - 1)))
    return output


def _slab_ufunc(ufunc: Any, inputs: Tuple[Any, ...], out: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """`ufunc(*inputs, out=out)` with the "same_kind" casting (each input an array of the
    shape of `out`, or a number), on slabs of the first axis in the slab threads for a
    large array. Each value comes from the same operation as in one call."""
    threads = get_num_threads()
    if out.size < _SLAB_MIN_SIZE or threads < 2:
        return cast(npt.NDArray[Any], ufunc(*inputs, out=out, casting="same_kind"))
    edges = np.linspace(0, out.shape[0], min(threads, out.shape[0]) + 1).astype(int)

    def run(k: int) -> None:
        part = slice(edges[k], edges[k + 1])
        parts = (x[part] if isinstance(x, np.ndarray) else x for x in inputs)
        ufunc(*parts, out=out[part], casting="same_kind")

    list(_slab_pool().map(run, range(edges.size - 1)))
    return out


# _add_times adds in chunks of this many values, so a chunk stays in the cache for its adds
_ADD_CHUNK = 1 << 15


def _add_times(
    total: npt.NDArray[np.float64], values: npt.NDArray[Any], times: int, first: bool
) -> None:
    """`times` float64 adds of `values` into `total`; with `first`, `total` starts at
    `values` and takes `times` - 1 adds: the bits of `times` numpy adds. In one thread,
    each chunk of values is read (and cast) once and stays in the cache for its adds. A
    large array with threads takes one ufunc call for each slab and add instead (a chunk
    loop in threads waits for the GIL). Both arrays are C-ordered, with one shape."""
    adds = times - 1 if first else times
    if total.size >= _SLAB_MIN_SIZE and get_num_threads() > 1:
        if first:
            _slab_pass(_copy_pass, values, values.ndim - 1, total)
        for _ in range(adds):
            _slab_ufunc(np.add, (total, values), total)
        return
    flat_total, flat_values = total.reshape(-1), values.reshape(-1)
    for a in range(0, total.size, _ADD_CHUNK):
        part = flat_total[a : a + _ADD_CHUNK]
        value = flat_values[a : a + _ADD_CHUNK].astype(np.float64, copy=False)
        if first:
            part[...] = value
        for _ in range(adds):
            np.add(part, value, out=part)


def _per_axis(value: Any, ndim: int) -> tuple[Any, ...]:
    """A scalar or a sequence as one value per axis (scipy's _normalize_sequence)."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(value)
    return (value,) * ndim


def _gaussian_filter(
    image: npt.NDArray[Any],
    sigma: Any,
    mode: str,
    truncate: float = 4.0,
    order: Any = 0,
    output: Optional[npt.NDArray[Any]] = None,
) -> npt.NDArray[Any]:
    """`scipy.ndimage.gaussian_filter` with the same steps (one 1-D pass per axis with a
    sigma above 1e-15, in axis order), each in `_slab_pass`: the same values."""
    if output is None:  # the passes (or the copy) write every value
        output = np.empty(image.shape, dtype=image.dtype)
    sigmas = _per_axis(sigma, image.ndim)
    orders = _per_axis(order, image.ndim)
    source = image
    for axis in range(image.ndim):
        if sigmas[axis] > 1e-15:
            one = partial(
                gaussian_filter1d,
                sigma=sigmas[axis],
                order=orders[axis],
                mode=mode,
                truncate=truncate,
            )
            _slab_pass(one, source, axis, output)
            source = output
    if source is image:
        output[...] = image
    return output


def _gaussian_laplace(
    image: npt.NDArray[Any], sigma: Any, mode: str, truncate: float = 4.0
) -> npt.NDArray[Any]:
    """`scipy.ndimage.gaussian_laplace` with the same steps (scipy's generic_laplace: the
    Gaussian second derivative along each axis, added in axis order), each 1-D pass in
    `_slab_pass`: the same values. The terms after the first all start with the same plain
    pass along axis 0 of the image, so this pass runs once (8 passes, not 9, in 3-D). Its
    result and one more temporary hold these terms (the passes write all of it).

    From `_LOG_FFT_MIN` voxels on, a float64 3-D image (with no zero sigma, as scipy skips
    such an axis) takes `_fft_gaussian_laplace`: values within about 1e-15 of the largest
    value of the passes, not the same bits."""
    ndim = image.ndim
    sigmas = _per_axis(sigma, ndim)
    if (
        image.dtype == np.float64
        and image.size >= _LOG_FFT_MIN
        and ndim == 3
        and min(sigmas) > 1e-15
    ):
        return _fft_gaussian_laplace(image, sigmas, mode, truncate)
    output = _gaussian_filter(image, sigmas, mode, truncate, order=_second_on(0, ndim))
    smoothed = _gaussian_filter(image, (sigmas[0],) + (0.0,) * (ndim - 1), mode, truncate)
    rest = (0.0, *sigmas[1:])
    term = np.empty_like(output)
    for axis in range(1, ndim):
        order = _second_on(axis, ndim)
        output += _gaussian_filter(smoothed, rest, mode, truncate, order=order, output=term)
    return output


def _second_on(axis: int, ndim: int) -> tuple[int, ...]:
    """Derivative orders with 2 on `axis` and 0 on the other axes."""
    return tuple(2 if a == axis else 0 for a in range(ndim))


# From this many voxels on, the LoG of a float64 image is one FFT convolution. Measured: it
# is faster than the 8 passes from 64^3 on, for sigma 1 to 4 voxels, at 1 and 10 threads.
_LOG_FFT_MIN = 1 << 18

# The np.pad modes that extend an image as the scipy modes extend each line
_PAD_MODES = {
    "constant": "constant",
    "nearest": "edge",
    "wrap": "wrap",
    "reflect": "symmetric",
    "mirror": "reflect",
}
# The code of each np.pad mode in _source_line
_PAD_CODES = {"constant": 0, "symmetric": 1, "reflect": 2, "edge": 3, "wrap": 4}


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _source_line(t: int, n: int, mode: int) -> int:
    """The line of an axis of n lines that line t of the padded axis reads, as np.pad does
    (`mode`: see _PAD_CODES); -1 for the constant."""
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


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _pad_planes_numba(
    image: npt.NDArray[np.floating[Any]],
    out: npt.NDArray[np.floating[Any]],
    w0: int,
    w1: int,
    w2: int,
    mode: int,
    value: float,
    first: int,
    last: int,
) -> None:
    """Planes first to last of `out` = np.pad of `image` with the widths w0, w1 and w2 before
    the axes (the shape of `out` gives the widths after them), np.pad mode `mode` (see
    _PAD_CODES) and the constant `value`."""
    n0, n1, n2 = image.shape
    for i in range(first, last):
        si = _source_line(i - w0, n0, mode)
        for j in range(out.shape[1]):
            sj = _source_line(j - w1, n1, mode)
            if si < 0 or sj < 0:
                for k in range(out.shape[2]):
                    out[i, j, k] = value
                continue
            for k in range(w2):
                sk = _source_line(k - w2, n2, mode)
                out[i, j, k] = value if sk < 0 else image[si, sj, sk]
            for k in range(n2):
                out[i, j, w2 + k] = image[si, sj, k]
            for k in range(w2 + n2, out.shape[2]):
                sk = _source_line(k - w2, n2, mode)
                out[i, j, k] = value if sk < 0 else image[si, sj, sk]


def _padded(
    image: npt.NDArray[Any], widths: Any, mode: str, value: float = 0.0
) -> npt.NDArray[Any]:
    """np.pad(image, widths, mode) with the constant `value` (mode "constant"), in the same
    memory order. A 3-D image is filled by a numba kernel, on slabs of planes in the slab
    threads from _SLAB_MIN_SIZE voxels on."""
    if image.ndim != 3:
        extra = {"constant_values": value} if mode == "constant" else {}
        return np.pad(image, widths, mode=mode, **extra)  # type: ignore[call-overload,no-any-return]
    if image.flags.fnc:  # column order: pad the row-order view of its transpose
        return _padded(image.T, [tuple(w) for w in widths][::-1], mode, value).T
    shape = tuple(n + a + b for n, (a, b) in zip(image.shape, widths, strict=True))
    out = np.empty(shape, dtype=image.dtype)
    args = (image, out, widths[0][0], widths[1][0], widths[2][0], _PAD_CODES[mode], float(value))
    threads = get_num_threads()
    if out.size < _SLAB_MIN_SIZE or threads < 2:
        _pad_planes_numba(*args, 0, shape[0])
        return out
    edges = np.linspace(0, shape[0], min(threads, shape[0]) + 1).astype(int)
    list(
        _slab_pool().map(
            lambda k: _pad_planes_numba(*args, int(edges[k]), int(edges[k + 1])),
            range(edges.size - 1),
        )
    )
    return out


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _times_log_transfer(
    spectrum: npt.NDArray[np.complex128],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
) -> None:
    """spectrum[i, j, k] *= a[i, j] * g[k] + b[i, j] * d[k], in place: the LoG transfer from
    the Gaussian (g) and the second derivative (d) of the last axis and the tables a and b of
    the first two axes."""
    for i in range(spectrum.shape[0]):
        for j in range(spectrum.shape[1]):
            a_ij = a[i, j]
            b_ij = b[i, j]
            for k in range(spectrum.shape[2]):
                spectrum[i, j, k] *= a_ij * g[k] + b_ij * d[k]


def _kernel_spectra(
    sigma: float, radius: int, size: int, truncate: float, half: bool
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The DFTs (rfft for `half`) on `size` points of the order 0 and order 2 kernels of
    gaussian_filter1d, centred on point 0. The kernels are symmetric: the DFTs are real."""
    impulse = np.zeros(2 * radius + 1)
    impulse[radius] = 1.0
    spectra = []
    for order in (0, 2):
        kernel = np.zeros(size)
        kernel[np.arange(-radius, radius + 1) % size] = gaussian_filter1d(
            impulse, sigma, order=order, mode="constant", truncate=truncate
        )
        spectra.append((np.fft.rfft(kernel) if half else np.fft.fft(kernel)).real.copy())
    return spectra[0], spectra[1]


def _fft_gaussian_laplace(
    image: npt.NDArray[np.float64], sigmas: Tuple[Any, ...], mode: str, truncate: float
) -> npt.NDArray[np.float64]:
    """The LoG of the passes of `_gaussian_laplace` as one FFT convolution with the same
    truncated kernels. The image is extended with the boundary mode (as each pass extends
    its lines) by the kernel radius and on to a fast FFT size. Its spectrum is multiplied by
    the transfer: on each axis the second derivative, times the Gaussian on the other axes,
    summed over the axes. The result is cut back to the image."""
    radii = [int(truncate * s + 0.5) for s in sigmas]
    sizes = [
        scipy.fft.next_fast_len(n + 2 * r, real=True)
        for n, r in zip(image.shape, radii, strict=True)
    ]
    widths = [(r, size - n - r) for n, r, size in zip(image.shape, radii, sizes, strict=True)]
    padded = _padded(image, widths, _PAD_MODES[mode])
    (g0, d0), (g1, d1), (g2, d2) = (
        _kernel_spectra(s, r, size, truncate, axis == 2)
        for axis, (s, r, size) in enumerate(zip(sigmas, radii, sizes, strict=True))
    )
    workers = get_num_threads()
    spectrum = scipy.fft.rfftn(padded, workers=workers)
    del padded
    first_two = np.multiply.outer(d0, g1) + np.multiply.outer(g0, d1)
    _times_log_transfer(spectrum, first_two, np.multiply.outer(g0, g1), g2, d2)
    response = scipy.fft.irfftn(spectrum, s=sizes, workers=workers, overwrite_x=True)
    return cast(
        npt.NDArray[np.float64],
        response[tuple(slice(r, r + n) for r, n in zip(radii, image.shape, strict=True))],
    )


def _uniform_filter(
    image: npt.NDArray[Any],
    size: int,
    mode: str,
    output: Optional[npt.NDArray[Any]] = None,
) -> npt.NDArray[Any]:
    """`scipy.ndimage.uniform_filter` with the same steps (one running-sum pass per axis,
    in axis order), each in `_slab_pass`: the same values. `output` may be `image`."""
    if output is None:  # the passes (or the copy) write every value
        output = np.empty(image.shape, dtype=image.dtype)
    if size <= 1:
        output[...] = image
        return output
    source = image
    for axis in range(image.ndim):
        _slab_pass(partial(uniform_filter1d, size=int(size), mode=mode), source, axis, output)
        source = output
    return output


def _convolve_axes(
    image: npt.NDArray[Any],
    kernels: Any,
    mode: str,
    threads: Optional[int] = None,
    output: Optional[npt.NDArray[Any]] = None,
    last_dtype: Any = None,
) -> npt.NDArray[Any]:
    """1-D convolutions along the axes in order (kernels[axis], one per axis), the first
    into `output` (default: a new array of the image type; may be the image) and the
    others in place, as the filters make them with convolve1d; each pass in
    `_slab_pass`. With `last_dtype` of another type than the passes, the last pass writes a
    new array of that type: scipy computes in double, so the values are those of a cast
    after the pass."""
    first = partial(convolve1d, weights=kernels[0], mode=mode)
    result = _slab_pass(first, image, 0, output, threads)
    for axis in range(1, len(kernels)):
        one = partial(convolve1d, weights=kernels[axis], mode=mode)
        last = axis == len(kernels) - 1 and last_dtype not in (None, result.dtype)
        target = np.empty(result.shape, dtype=last_dtype) if last else result
        result = _slab_pass(one, result, axis, target, threads)
    return result


# ==============================================================================
# Normalized Convolution Utilities for Source Mask Support
# ==============================================================================


def _normalized_uniform_filter(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    size: int,
    mode: str = "constant",
    weight_threshold: float = 0.01,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Apply uniform (mean) filter with normalized convolution for source masking.

    Excludes invalid (sentinel) voxels from the mean computation by using
    normalized convolution: output = sum(valid_values) / sum(valid_weights).

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        size: Filter support (kernel size)
        mode: Boundary mode for scipy.ndimage
        weight_threshold: Minimum weight to consider output valid.
            Default 0.01 means at least 1% contribution from valid voxels.

    Returns:
        Tuple of (filtered_image, output_valid_mask)
    """
    # Sum of the valid values (invalid = 0) and of the valid weights, each in place
    weighted_sum = _zero_filled(image, source_mask)
    _uniform_filter(weighted_sum, size, mode, output=weighted_sum)
    weight_sum = source_mask.astype(np.float64)
    _uniform_filter(weight_sum, size, mode, output=weight_sum)
    return _normalize(weighted_sum, weight_sum, weight_threshold)


def _normalized_gaussian_laplace(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    sigma: Union[float, Tuple[float, ...]],
    mode: str = "constant",
    truncate: float = 4.0,
    weight_threshold: float = 0.01,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Apply Laplacian of Gaussian with normalized convolution for source masking.

    Note: LoG with normalized convolution is approximated by computing LoG on
    the valid-zeroed image and normalizing by the Gaussian-smoothed mask weights.
    This is an approximation since LoG is a second derivative.

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        sigma: Standard deviation in voxels (scalar or per-axis tuple)
        mode: Boundary mode for scipy.ndimage
        truncate: Filter size cutoff in sigma units
        weight_threshold: Minimum weight to consider output valid

    Returns:
        Tuple of (filtered_image, output_valid_mask)
    """
    # LoG of the zeroed image; the weight sum uses the Gaussian (the LoG sums to 0)
    log_response = _gaussian_laplace(_zero_filled(image, source_mask), sigma, mode, truncate)
    weight_sum = source_mask.astype(np.float64)
    _gaussian_filter(weight_sum, sigma, mode, truncate, output=weight_sum)
    return _normalize(log_response, weight_sum, weight_threshold)


def _normalized_gaussian(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    sigma: Union[float, Tuple[float, ...]],
    mode: str = "constant",
    truncate: float = 4.0,
    weight_threshold: float = 0.01,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """Gaussian smoothing by normalized convolution: G * (f m) / G * m, where the weight
    G * m reaches `weight_threshold` (the output valid mask), else 0."""
    smoothed = _gaussian_filter(_zero_filled(image, source_mask), sigma, mode, truncate)
    weight_sum = source_mask.astype(np.float64)
    _gaussian_filter(weight_sum, sigma, mode, truncate, output=weight_sum)
    return _normalize(smoothed, weight_sum, weight_threshold)


def _normalized_convolve1d(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    kernel: npt.NDArray[np.floating[Any]],
    axis: int,
    mode: str = "constant",
    weight_threshold: float = 0.01,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Apply 1D convolution with normalized convolution for source masking.

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        kernel: 1D convolution kernel
        axis: Axis along which to convolve
        mode: Boundary mode for scipy.ndimage
        weight_threshold: Minimum weight to consider output valid

    Returns:
        Tuple of (filtered_image, output_valid_mask)
    """
    # Zero out invalid voxels
    valid_image = np.where(source_mask, image, 0.0).astype(np.float64)

    # Apply convolution to zeroed image
    response = convolve1d(valid_image, kernel, axis=axis, mode=mode)

    # Weight of the valid voxels, as a fraction of the full kernel weight (1 where every
    # voxel is valid)
    abs_kernel = np.abs(kernel)
    weight_sum = convolve1d(source_mask.astype(np.float64), abs_kernel, axis=axis, mode=mode)
    weight_sum /= float(abs_kernel.sum())

    # Normalize
    valid_output = weight_sum >= weight_threshold
    result = np.zeros_like(response)
    result[valid_output] = response[valid_output] / weight_sum[valid_output]

    return result.astype(np.float32), valid_output


def _normalized_separable_convolve_3d(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    g1: npt.NDArray[np.floating[Any]],
    g2: npt.NDArray[np.floating[Any]],
    g3: npt.NDArray[np.floating[Any]],
    mode: str = "constant",
    weight_threshold: float = 0.01,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Apply separable 3D convolution with normalized convolution for source masking.

    Uses three 1D kernels applied sequentially along each axis.

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        g1, g2, g3: 1D kernels for axes 0, 1, 2
        mode: Boundary mode for scipy.ndimage
        weight_threshold: Minimum weight to consider output valid

    Returns:
        Tuple of (filtered_image, output_valid_mask)
    """
    # Separable convolution of the zeroed image, in place
    valid_image = _zero_filled(image, source_mask)
    result = _convolve_axes(valid_image, (g1, g2, g3), mode, output=valid_image)

    # Weight of the valid voxels, as a fraction of the full kernel weight: 1 where every
    # voxel is valid, so the response there is the plain convolution
    abs_g1, abs_g2, abs_g3 = np.abs(g1), np.abs(g2), np.abs(g3)
    weight = source_mask.astype(np.float64)
    _convolve_axes(weight, (abs_g1, abs_g2, abs_g3), mode, output=weight)
    weight /= float(abs_g1.sum()) * float(abs_g2.sum()) * float(abs_g3.sum())
    return _normalize(result, weight, weight_threshold)


def _zero_filled(
    image: npt.NDArray[np.floating[Any]], source_mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """The image as float64, with 0.0 at the invalid voxels."""
    result = np.zeros(image.shape, dtype=np.float64)
    np.copyto(result, image, where=source_mask)
    return result


def _normalize(
    response: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
    weight_threshold: float,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """(response / weight where the weight reaches `weight_threshold`, else 0) as float32,
    and where it does (the output valid mask). The division writes into `response`."""
    valid_output = weight >= weight_threshold
    np.divide(response, weight, out=response, where=valid_output)
    response[~valid_output] = 0.0
    return response.astype(np.float32), valid_output


def _prepare_masked_image(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    fill_value: float = 0.0,
) -> npt.NDArray[np.floating[Any]]:
    """
    Prepare image for FFT-based filtering by zeroing out invalid voxels.

    For FFT-based filters (Simoncelli, Riesz), we cannot use normalized
    convolution directly. Instead, we zero out invalid voxels which acts
    as a first-order approximation.

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        fill_value: Value to use for invalid voxels (default 0.0)

    Returns:
        Image with invalid voxels set to fill_value
    """
    result = image.copy()
    result[~source_mask] = fill_value
    return result
