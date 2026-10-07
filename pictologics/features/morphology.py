"""
Morphology Feature Extraction Module
====================================

This module provides functions for calculating Morphological (Shape and Size)
features from medical images. It implements the Image Biomarker Standardisation
Initiative (IBSI) compliant algorithms.

Key Features:
-------------
- **Voxel-based**: Volume (voxel counting).
- **Mesh-based**: Surface Area, Volume (mesh), Compactness, Sphericity.
- **PCA-based**: Major/Minor/Least Axis Length, Elongation, Flatness.
- **Convex Hull**: Volume, Area, Max 3D Diameter.
- **Bounding Box**: Oriented (OMBB) and Axis-Aligned (AABB) Bounding Boxes.
- **Minimum Volume Enclosing Ellipsoid (MVEE)**: Volume, Area.
- **Intensity-Weighted**: Center of Mass Shift, Integrated Intensity.

Optimization:
-------------
Uses `numba` kernels for the marching cubes mesh, the moments, the exact convex hull and
its candidates, the oriented bounding box and the Khachiyan algorithm of the MVEE.

Example:
    Calculate morphology features from a mask:

    ```python
    import numpy as np
    from pictologics.loader import Image
    from pictologics.features.morphology import calculate_morphology_features

    # Create dummy mask
    mask_arr = np.zeros((50, 50, 50), dtype=np.uint8)
    mask_arr[10:40, 10:40, 10:40] = 1
    mask = Image(mask_arr, spacing=(1.0, 1.0, 1.0), origin=(0,0,0))

    # Calculate features
    features = calculate_morphology_features(mask)
    print(features["volume_voxel_counting_YEKZ"])
    ```
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple, Optional, cast

import numpy as np
from numba import get_num_threads, jit, prange
from numpy import typing as npt
from scipy.spatial import ConvexHull
from scipy.special import eval_legendre

from ..loader import Image
from ._mc_tables import EDGE_TABLE, TRIANGLE_COUNT, TRIANGLE_TABLE
from ._utils import PRANGE_ONLY, compute_nonzero_bbox, serial_twin, sized


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _accumulate_moments_from_mask_numba(
    mask: npt.NDArray[np.floating[Any]],
) -> tuple[int, float, float, float, float, float, float, float, float, float]:
    """Accumulate first/second moments of voxel indices for mask != 0.

    Each slice has its own sums, and the slice sums add in order, so the result does
    not depend on the number of threads. The sums are whole numbers, so they are exact
    below 2^53."""
    d0, d1, d2 = mask.shape
    counts = np.empty(d0, dtype=np.int64)
    sums = np.empty((d0, 5), dtype=np.float64)  # sums of j, k, j * j, k * k and j * k
    for i in prange(d0):
        count = 0
        sj = 0.0
        sk = 0.0
        sjj = 0.0
        skk = 0.0
        sjk = 0.0
        for j in range(d1):
            for k in range(d2):
                if mask[i, j, k] != 0:
                    fj = float(j)
                    fk = float(k)
                    count += 1
                    sj += fj
                    sk += fk
                    sjj += fj * fj
                    skk += fk * fk
                    sjk += fj * fk
        counts[i] = count
        sums[i, 0] = sj
        sums[i, 1] = sk
        sums[i, 2] = sjj
        sums[i, 3] = skk
        sums[i, 4] = sjk

    n = 0
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s00 = 0.0
    s11 = 0.0
    s22 = 0.0
    s01 = 0.0
    s02 = 0.0
    s12 = 0.0
    for i in range(d0):
        fi = float(i)
        n += counts[i]
        s0 += fi * counts[i]
        s00 += fi * fi * counts[i]
        s1 += sums[i, 0]
        s2 += sums[i, 1]
        s11 += sums[i, 2]
        s22 += sums[i, 3]
        s12 += sums[i, 4]
        s01 += fi * sums[i, 0]
        s02 += fi * sums[i, 1]
    return n, s0, s1, s2, s00, s11, s22, s01, s02, s12


# Serial twins of the parallel kernels of a small ROI (see serial_twin). Measured on 10
# threads: below these sizes, starting the threads costs more than the work.
_accumulate_moments_from_mask_numba_serial = serial_twin(_accumulate_moments_from_mask_numba)
# Mask box voxels from which the two moment kernels run in parallel (crossover about 130k)
_MOMENTS_PARALLEL_MIN = 1 << 17


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _accumulate_intensity_weighted_moments_numba(
    mask: npt.NDArray[np.floating[Any]], image: npt.NDArray[np.floating[Any]]
) -> tuple[int, float, float, float, float]:
    """Accumulate intensity-weighted index sums over mask != 0.

    Each slice has its own sums, and the slice sums add in order, so the result does
    not depend on the number of threads."""
    d0, d1, d2 = mask.shape
    counts = np.empty(d0, dtype=np.int64)
    sums = np.empty((d0, 4), dtype=np.float64)
    for i in prange(d0):
        count = 0
        sum_w = 0.0
        sum_i1_w = 0.0
        sum_i2_w = 0.0
        for j in range(d1):
            for k in range(d2):
                if mask[i, j, k] != 0:
                    w = float(image[i, j, k])
                    count += 1
                    sum_w += w
                    sum_i1_w += float(j) * w
                    sum_i2_w += float(k) * w
        counts[i] = count
        sums[i, 0] = sum_w
        sums[i, 1] = float(i) * sum_w
        sums[i, 2] = sum_i1_w
        sums[i, 3] = sum_i2_w

    count = 0
    sum_w = 0.0
    sum_i0_w = 0.0
    sum_i1_w = 0.0
    sum_i2_w = 0.0
    for i in range(d0):
        count += counts[i]
        sum_w += sums[i, 0]
        sum_i0_w += sums[i, 1]
        sum_i1_w += sums[i, 2]
        sum_i2_w += sums[i, 3]
    return count, sum_w, sum_i0_w, sum_i1_w, sum_i2_w


_accumulate_intensity_weighted_moments_numba_serial = serial_twin(
    _accumulate_intensity_weighted_moments_numba
)


@jit(nopython=True, cache=True)  # type: ignore
def _column_stats_numba(
    points: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Minimum, maximum and mean of each column of an (n, 3) array in one pass. The sums
    run row by row, as numpy reduces axis 0, so the mean is that of np.mean."""
    lo = points[0].copy()
    hi = points[0].copy()
    total = np.zeros(3, dtype=np.float64)
    for i in range(points.shape[0]):
        for a in range(3):
            v = points[i, a]
            if v < lo[a]:
                lo[a] = v
            if v > hi[a]:
                hi[a] = v
            total[a] += v
    return lo, hi, total / points.shape[0]


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _ombb_extents_numba(
    verts: npt.NDArray[np.floating[Any]],
    center: npt.NDArray[np.floating[Any]],
    evecs: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Compute min/max extents of vertices projected onto PCA axes (for OMBB)."""
    min_rot = np.empty(3, dtype=np.float64)
    max_rot = np.empty(3, dtype=np.float64)

    n = verts.shape[0]

    # Reductions for min/max
    min_r0 = np.inf
    min_r1 = np.inf
    min_r2 = np.inf
    max_r0 = -np.inf
    max_r1 = -np.inf
    max_r2 = -np.inf

    for idx in prange(n):
        dx0 = verts[idx, 0] - center[0]
        dx1 = verts[idx, 1] - center[1]
        dx2 = verts[idx, 2] - center[2]

        r0 = dx0 * evecs[0, 0] + dx1 * evecs[1, 0] + dx2 * evecs[2, 0]
        r1 = dx0 * evecs[0, 1] + dx1 * evecs[1, 1] + dx2 * evecs[2, 1]
        r2 = dx0 * evecs[0, 2] + dx1 * evecs[1, 2] + dx2 * evecs[2, 2]

        min_r0 = min(min_r0, r0)
        min_r1 = min(min_r1, r1)
        min_r2 = min(min_r2, r2)
        max_r0 = max(max_r0, r0)
        max_r1 = max(max_r1, r1)
        max_r2 = max(max_r2, r2)

    min_rot[0] = min_r0
    min_rot[1] = min_r1
    min_rot[2] = min_r2
    max_rot[0] = max_r0
    max_rot[1] = max_r1
    max_rot[2] = max_r2

    return min_rot, max_rot


_ombb_extents_numba_serial = serial_twin(_ombb_extents_numba)
# Mesh vertices from which the OMBB extents run in parallel (crossover about 120,000)
_OMBB_PARALLEL_MIN = 120_000


@jit(nopython=True, parallel=PRANGE_ONLY, fastmath=True, cache=True)  # type: ignore
def _max_pairwise_distance_numba(points: npt.NDArray[np.floating[Any]]) -> float:
    """Compute the maximum pairwise Euclidean distance."""
    n = points.shape[0]
    if n < 2:
        return 0.0

    # Store max distance squared found by each outer iteration
    # Since we parallelize the outer loop, each iteration 'i' is independent.
    max_d2_arr = np.zeros(n - 1, dtype=np.float64)

    for i in prange(n - 1):
        x0 = points[i, 0]
        y0 = points[i, 1]
        z0 = points[i, 2]

        local_max = 0.0

        for j in range(i + 1, n):
            dx = points[j, 0] - x0
            dy = points[j, 1] - y0
            dz = points[j, 2] - z0
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > local_max:
                local_max = d2

        max_d2_arr[i] = local_max

    # Global max
    return float(math.sqrt(np.max(max_d2_arr)))


@jit(nopython=True, nogil=True, fastmath=True, cache=True)  # type: ignore
def _max_pairwise_distance_serial_numba(points: npt.NDArray[np.floating[Any]]) -> float:
    """`_max_pairwise_distance_numba` in one thread, with the same arithmetic: the maximum
    of each row, then the maximum of the rows, so the same result. It has no parallel
    region, so it can run in a thread next to the parallel kernels."""
    n = points.shape[0]
    if n < 2:
        return 0.0

    max_d2_arr = np.zeros(n - 1, dtype=np.float64)

    for i in range(n - 1):
        x0 = points[i, 0]
        y0 = points[i, 1]
        z0 = points[i, 2]

        local_max = 0.0

        for j in range(i + 1, n):
            dx = points[j, 0] - x0
            dy = points[j, 1] - y0
            dz = points[j, 2] - z0
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > local_max:
                local_max = d2

        max_d2_arr[i] = local_max

    return float(math.sqrt(np.max(max_d2_arr)))


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _line_end_candidates_numba(
    verts: npt.NDArray[np.floating[Any]], spacing: npt.NDArray[np.floating[Any]]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """Indices, in order, of the mesh vertices that are the first or the last vertex on each
    of their three grid lines (a hull vertex is extreme on each line through it), and the
    grid coordinates of these vertices. Marching cubes on a binary mask puts the vertices on
    a half-voxel grid, so `round(2 * verts / spacing)` gives exact grid coordinates. The
    tables hold the low and the high end of each line side by side, as int32."""
    n = verts.shape[0]
    lowest = np.empty(3, dtype=np.int64)
    for a in range(3):
        lowest[a] = int(np.rint(2.0 * verts[0, a] / spacing[a]))
    for i in range(n):
        for a in range(3):
            g = int(np.rint(2.0 * verts[i, a] / spacing[a]))
            if g < lowest[a]:
                lowest[a] = g
    grid = np.empty((n, 3), dtype=np.int32)
    size = np.zeros(3, dtype=np.int64)
    for i in range(n):
        for a in range(3):
            g = int(np.rint(2.0 * verts[i, a] / spacing[a])) - lowest[a]
            grid[i, a] = g
            if g + 1 > size[a]:
                size[a] = g + 1
    nx, ny, nz = size[0], size[1], size[2]
    ends0 = np.empty((ny, nz, 2), dtype=np.int32)  # along axis 0
    ends1 = np.empty((nx, nz, 2), dtype=np.int32)
    ends2 = np.empty((nx, ny, 2), dtype=np.int32)
    ends0[:, :, 0] = nx
    ends0[:, :, 1] = -1
    ends1[:, :, 0] = ny
    ends1[:, :, 1] = -1
    ends2[:, :, 0] = nz
    ends2[:, :, 1] = -1
    for i in range(n):
        x, y, z = grid[i, 0], grid[i, 1], grid[i, 2]
        ends0[y, z, 0] = min(ends0[y, z, 0], x)
        ends0[y, z, 1] = max(ends0[y, z, 1], x)
        ends1[x, z, 0] = min(ends1[x, z, 0], y)
        ends1[x, z, 1] = max(ends1[x, z, 1], y)
        ends2[x, y, 0] = min(ends2[x, y, 0], z)
        ends2[x, y, 1] = max(ends2[x, y, 1], z)

    keep = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        x, y, z = grid[i, 0], grid[i, 1], grid[i, 2]
        if (
            (x == ends0[y, z, 0] or x == ends0[y, z, 1])
            and (y == ends1[x, z, 0] or y == ends1[x, z, 1])
            and (z == ends2[x, y, 0] or z == ends2[x, y, 1])
        ):
            keep[m] = i
            m += 1
    return keep[:m], grid[keep[:m]]


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _counting_order(
    keys: npt.NDArray[np.int32], order: npt.NDArray[np.int64], size: int
) -> npt.NDArray[np.int64]:
    """`order`, stably sorted by `keys[order]` (whole numbers from 0 to size - 1)."""
    counts = np.zeros(size + 1, dtype=np.int64)
    for i in order:
        counts[keys[i] + 1] += 1
    for v in range(size):
        counts[v + 1] += counts[v]
    out = np.empty_like(order)
    for i in order:
        out[counts[keys[i]]] = i
        counts[keys[i]] += 1
    return out


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _mark_plane_hull(
    us: npt.NDArray[np.int64],
    vs: npt.NDArray[np.int64],
    ids: npt.NDArray[np.int64],
    mark: npt.NDArray[np.bool_],
    stack: npt.NDArray[np.int64],
) -> None:
    """Mark the vertices of the 2-D convex hull of the points (us, vs), sorted by u, then v
    (Andrew's monotone chain, exact integer cross products). Points on a hull edge are no
    vertices. `stack` is a work array of three columns."""
    n = us.shape[0]
    if n <= 2:
        for k in range(n):
            mark[ids[k]] = True
        return
    for direction in range(2):  # the lower chain, then the upper chain
        top = 0
        for t in range(n):
            k = t if direction == 0 else n - 1 - t
            while top >= 2:
                du = stack[top - 1, 0] - stack[top - 2, 0]
                dv = stack[top - 1, 1] - stack[top - 2, 1]
                if du * (vs[k] - stack[top - 2, 1]) - dv * (us[k] - stack[top - 2, 0]) > 0:
                    break
                top -= 1
            stack[top, 0] = us[k]
            stack[top, 1] = vs[k]
            stack[top, 2] = ids[k]
            top += 1
        for t in range(top - 1):
            mark[stack[t, 2]] = True


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _plane_hull_candidates_numba(
    first: npt.NDArray[np.int64], grid: npt.NDArray[np.int32]
) -> npt.NDArray[np.int64]:
    """The indices of `first` (line-end candidates, with their grid coordinates `grid`) that
    are vertices of the 2-D convex hull of the candidates of each of their three axis planes.
    A 3-D hull vertex is extreme in every plane through it, so no hull vertex is lost. Each
    axis orders the candidates by plane, row and column with three counting sorts."""
    m = first.shape[0]  # a mesh has at least one line end
    marks = np.zeros((3, m), dtype=np.bool_)
    coords = np.empty((3, m), dtype=np.int32)
    size = np.empty(3, dtype=np.int64)
    for a in range(3):
        for t in range(m):
            coords[a, t] = grid[t, a]
        size[a] = coords[a].max() + 1
    us = np.empty(m, dtype=np.int64)
    vs = np.empty(m, dtype=np.int64)
    ids = np.empty(m, dtype=np.int64)
    stack = np.empty((m, 3), dtype=np.int64)
    for a in range(3):  # the plane axis; the rows run along axis b, the columns along c
        b = 1 if a == 0 else 0
        c = 1 if a == 2 else 2
        order = np.arange(m)
        order = _counting_order(coords[c], order, size[c])
        order = _counting_order(coords[b], order, size[b])
        order = _counting_order(coords[a], order, size[a])
        start = 0
        while start < m:
            plane = coords[a, order[start]]
            n = 0
            while start + n < m and coords[a, order[start + n]] == plane:
                t = order[start + n]
                us[n] = coords[b, t]
                vs[n] = coords[c, t]
                ids[n] = t
                n += 1
            _mark_plane_hull(us[:n], vs[:n], ids[:n], marks[a], stack)
            start += n
    keep = np.empty(m, dtype=np.int64)
    k = 0
    for t in range(m):
        if marks[0, t] and marks[1, t] and marks[2, t]:
            keep[k] = first[t]
            k += 1
    return keep[:k]


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _hull_candidates_numba(
    verts: npt.NDArray[np.floating[Any]], spacing: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.int64]:
    """Indices, in order, of the mesh vertices that can be convex hull vertices: the line
    ends (`_line_end_candidates_numba`) that are 2-D hull vertices in their three axis
    planes (`_plane_hull_candidates_numba`). On the CT lesion, 856 of 30,186 vertices stay
    (4,950 line ends) for its 632 hull vertices."""
    first, grid = _line_end_candidates_numba(verts, spacing)
    keep: npt.NDArray[np.int64] = _plane_hull_candidates_numba(first, grid)
    return keep


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _gcd_numba(a: int, b: int) -> int:
    """The greatest common divisor of two whole numbers (0 for two zeros)."""
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _exact_hull_numba(
    lattice: npt.NDArray[np.int64],
) -> tuple[bool, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The convex hull of distinct lattice points (quickhull with exact int64 orientation
    tests): whether it exists (four points, not in one plane), the indices of its vertices
    in order, and its triangles (outward). A triangle vertex is a hull vertex when its
    triangles lie in three or more planes; a point inside a face or an edge is none, as in
    Qhull. Dead triangles give their slots to new ones, so at most 4n + 16 slots are used:
    a hull of v vertices has 2v - 4 triangles, and a step adds at most v."""
    n = lattice.shape[0]
    no_vertices = np.zeros(0, dtype=np.int64)
    no_triangles = np.zeros((0, 3), dtype=np.int64)
    if n < 4:
        return False, no_vertices, no_triangles
    g = lattice
    # 1. The start: the lowest x, the farthest point from it, the farthest point from their
    # line, and the farthest point from their plane
    a = 0
    for i in range(n):
        if g[i, 0] < g[a, 0]:
            a = i
    b = a
    best = 0
    for i in range(n):
        d = (g[i, 0] - g[a, 0]) ** 2 + (g[i, 1] - g[a, 1]) ** 2 + (g[i, 2] - g[a, 2]) ** 2
        if d > best:
            best = d
            b = i
    ux, uy, uz = g[b, 0] - g[a, 0], g[b, 1] - g[a, 1], g[b, 2] - g[a, 2]
    c = -1
    best = 0
    for i in range(n):
        vx, vy, vz = g[i, 0] - g[a, 0], g[i, 1] - g[a, 1], g[i, 2] - g[a, 2]
        cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        d = cx * cx + cy * cy + cz * cz
        if d > best:
            best = d
            c = i
    if c < 0:  # all points on one line
        return False, no_vertices, no_triangles
    vx, vy, vz = g[c, 0] - g[a, 0], g[c, 1] - g[a, 1], g[c, 2] - g[a, 2]
    px, py, pz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    top = -1
    best = 0
    for i in range(n):
        o = px * (g[i, 0] - g[a, 0]) + py * (g[i, 1] - g[a, 1]) + pz * (g[i, 2] - g[a, 2])
        if abs(o) > best:
            best = abs(o)
            top = i
    if top < 0:  # all points in one plane
        return False, no_vertices, no_triangles
    if px * (g[top, 0] - g[a, 0]) + py * (g[top, 1] - g[a, 1]) + pz * (g[top, 2] - g[a, 2]) > 0:
        b, c = c, b  # the triangle (a, b, c) faces away from the fourth point

    # 2. The triangles: vertices, the neighbours across (v0 v1), (v1 v2), (v2 v0), the
    # outward normal with its offset, and the outside points (linked lists)
    cap = 4 * n + 16
    tri = np.empty((cap, 3), dtype=np.int64)
    nbr = np.empty((cap, 3), dtype=np.int64)
    normal = np.empty((cap, 3), dtype=np.int64)
    offset = np.empty(cap, dtype=np.int64)
    alive = np.zeros(cap, dtype=np.bool_)
    queued = np.zeros(cap, dtype=np.bool_)
    head = np.full(cap, -1, dtype=np.int64)
    after = np.full(n, -1, dtype=np.int64)
    free = np.empty(cap, dtype=np.int64)
    n_free = 0
    used = 0
    start = np.array([[a, b, c], [a, top, b], [b, top, c], [c, top, a]], dtype=np.int64)
    for f in range(4):
        tri[f] = start[f]
    used = 4
    new = np.empty(cap, dtype=np.int64)
    visible = np.zeros(cap, dtype=np.bool_)
    seen = np.empty(cap, dtype=np.int64)
    edge_a = np.empty(cap, dtype=np.int64)
    edge_b = np.empty(cap, dtype=np.int64)
    edge_n = np.empty(cap, dtype=np.int64)
    by_start = np.full(n, -1, dtype=np.int64)
    by_end = np.full(n, -1, dtype=np.int64)
    stack = np.empty(cap, dtype=np.int64)
    depth = 0
    for f in range(4):
        new[f] = f
    n_new = 4
    point = -1  # the point of the step (none for the start)
    n_seen = 0  # the triangles that see the point
    while True:
        # Normals and neighbours of the new triangles
        for t in range(n_new):
            f = new[t]
            p0, p1, p2 = tri[f, 0], tri[f, 1], tri[f, 2]
            e1x, e1y, e1z = g[p1, 0] - g[p0, 0], g[p1, 1] - g[p0, 1], g[p1, 2] - g[p0, 2]
            e2x, e2y, e2z = g[p2, 0] - g[p0, 0], g[p2, 1] - g[p0, 1], g[p2, 2] - g[p0, 2]
            nx = e1y * e2z - e1z * e2y
            ny = e1z * e2x - e1x * e2z
            nz = e1x * e2y - e1y * e2x
            normal[f, 0], normal[f, 1], normal[f, 2] = nx, ny, nz
            offset[f] = nx * g[p0, 0] + ny * g[p0, 1] + nz * g[p0, 2]
            alive[f] = True
            head[f] = -1
        if point < 0:  # the start: the neighbours by shared edges
            for f in range(4):
                for e in range(3):
                    u, v = tri[f, e], tri[f, (e + 1) % 3]
                    for h in range(4):
                        for e2 in range(3):
                            if tri[h, e2] == v and tri[h, (e2 + 1) % 3] == u:
                                nbr[f, e] = h
            for i in range(n):
                if i != a and i != b and i != c and i != top:
                    for f in range(4):
                        if (
                            normal[f, 0] * g[i, 0] + normal[f, 1] * g[i, 1] + normal[f, 2] * g[i, 2]
                            > offset[f]
                        ):
                            after[i] = head[f]
                            head[f] = i
                            break
        else:  # a step: the outside points of the visible triangles go to the new ones
            for t in range(n_new):
                f = new[t]
                nbr[f, 1] = by_start[tri[f, 1]]  # across (b, point): the one that starts at b
                nbr[f, 2] = by_end[tri[f, 0]]  # across (point, a): the one that ends at a
            for t in range(n_new):
                by_start[edge_a[t]] = -1
                by_end[edge_b[t]] = -1
            for t in range(n_seen):
                f = seen[t]
                i = head[f]
                while i >= 0:
                    following = after[i]
                    if i != point:
                        for u in range(n_new):
                            h = new[u]
                            if (
                                normal[h, 0] * g[i, 0]
                                + normal[h, 1] * g[i, 1]
                                + normal[h, 2] * g[i, 2]
                                > offset[h]
                            ):
                                after[i] = head[h]
                                head[h] = i
                                break
                    i = following
                head[f] = -1
                alive[f] = False
                visible[f] = False
                free[n_free] = f
                n_free += 1
        for t in range(n_new):
            f = new[t]
            if head[f] >= 0 and not queued[f]:
                queued[f] = True
                stack[depth] = f
                depth += 1
        # 3. The next triangle with outside points, and its farthest point
        f0 = -1
        while depth > 0:
            depth -= 1
            f = stack[depth]
            queued[f] = False
            if alive[f] and head[f] >= 0:
                f0 = f
                break
        if f0 < 0:
            break
        length = np.sqrt(float(normal[f0, 0] ** 2 + normal[f0, 1] ** 2 + normal[f0, 2] ** 2))
        point = -1
        far = -1.0
        i = head[f0]
        while i >= 0:
            d = (
                normal[f0, 0] * g[i, 0]
                + normal[f0, 1] * g[i, 1]
                + normal[f0, 2] * g[i, 2]
                - offset[f0]
            ) / length
            if d > far:
                far = d
                point = i
            i = after[i]
        # 4. The triangles that see the point, and the edges of their border
        n_seen = 0
        seen[n_seen] = f0
        n_seen += 1
        visible[f0] = True
        q = 0
        while q < n_seen:
            f = seen[q]
            q += 1
            for e in range(3):
                h = nbr[f, e]
                if not visible[h] and (
                    normal[h, 0] * g[point, 0]
                    + normal[h, 1] * g[point, 1]
                    + normal[h, 2] * g[point, 2]
                    > offset[h]
                ):
                    visible[h] = True
                    seen[n_seen] = h
                    n_seen += 1
        n_new = 0
        for t in range(n_seen):
            f = seen[t]
            for e in range(3):
                h = nbr[f, e]
                if not visible[h]:
                    edge_a[n_new] = tri[f, e]
                    edge_b[n_new] = tri[f, (e + 1) % 3]
                    edge_n[n_new] = h
                    n_new += 1
        # 5. A new triangle (a, b, point) on each border edge
        for t in range(n_new):
            if n_free > 0:
                n_free -= 1
                f = free[n_free]
            else:
                f = used
                used += 1
            ea, eb, h = edge_a[t], edge_b[t], edge_n[t]
            tri[f, 0], tri[f, 1], tri[f, 2] = ea, eb, point
            nbr[f, 0] = h
            for e in range(3):  # the old neighbour faces the new triangle now
                if tri[h, e] == eb and tri[h, (e + 1) % 3] == ea:
                    nbr[h, e] = f
            by_start[ea] = f
            by_end[eb] = f
            new[t] = f

    # 6. The triangles and the vertices whose triangles lie in three or more planes
    planes = np.zeros((n, 3, 3), dtype=np.int64)
    count = np.zeros(n, dtype=np.int64)
    m = 0
    for f in range(used):
        if alive[f]:
            m += 1
            k = _gcd_numba(_gcd_numba(normal[f, 0], normal[f, 1]), normal[f, 2])
            w0, w1, w2 = normal[f, 0] // k, normal[f, 1] // k, normal[f, 2] // k
            for e in range(3):
                v = tri[f, e]
                fresh = count[v] < 3
                for j in range(count[v]):
                    if planes[v, j, 0] == w0 and planes[v, j, 1] == w1 and planes[v, j, 2] == w2:
                        fresh = False
                if fresh:
                    planes[v, count[v], 0] = w0
                    planes[v, count[v], 1] = w1
                    planes[v, count[v], 2] = w2
                    count[v] += 1
    triangles = np.empty((m, 3), dtype=np.int64)
    k = 0
    for f in range(used):
        if alive[f]:
            triangles[k] = tri[f]
            k += 1
    vertices = np.flatnonzero(count >= 3)
    return True, vertices, triangles


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _hull_area_volume_numba(
    points: npt.NDArray[np.float64], triangles: npt.NDArray[np.int64]
) -> tuple[float, float]:
    """The area and the volume of a closed triangle mesh with outward triangles."""
    area = 0.0
    vol6 = 0.0
    for t in range(triangles.shape[0]):
        a, b, c = triangles[t, 0], triangles[t, 1], triangles[t, 2]
        e1x, e1y, e1z = (
            points[b, 0] - points[a, 0],
            points[b, 1] - points[a, 1],
            points[b, 2] - points[a, 2],
        )
        e2x, e2y, e2z = (
            points[c, 0] - points[a, 0],
            points[c, 1] - points[a, 1],
            points[c, 2] - points[a, 2],
        )
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        area += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        vol6 += (
            points[a, 0] * (points[b, 1] * points[c, 2] - points[b, 2] * points[c, 1])
            + points[a, 1] * (points[b, 2] * points[c, 0] - points[b, 0] * points[c, 2])
            + points[a, 2] * (points[b, 0] * points[c, 1] - points[b, 1] * points[c, 0])
        )
    return area, abs(vol6) / 6.0


@jit(nopython=True, cache=True)  # type: ignore
def _mc_corners(vol: npt.NDArray[np.uint8], i: int, j: int, k: int) -> int:
    """Bits of the four cube corners in the plane z = k that are 0, in PyMCubes' order."""
    c = 0
    if vol[i, j, k] == 0:
        c |= 1
    if vol[i + 1, j, k] == 0:
        c |= 2
    if vol[i + 1, j + 1, k] == 0:
        c |= 4
    if vol[i, j + 1, k] == 0:
        c |= 8
    return c


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _mc_counts_numba(
    vol: npt.NDArray[np.uint8],
    edge_table: npt.NDArray[np.int32],
    tri_count: npt.NDArray[np.int32],
    n_verts: npt.NDArray[np.int64],
    n_faces: npt.NDArray[np.int64],
) -> None:
    """The vertices and the face corners that each x plane of cubes of
    `_marching_cubes_numba` makes (whole numbers, so the plane order does not matter)."""
    nx, ny, nz = vol.shape[0] - 1, vol.shape[1] - 1, vol.shape[2] - 1
    for i in prange(nx):
        n_v = 0
        n_f = 0
        for j in range(ny):
            c = _mc_corners(vol, i, j, 0) << 4
            for k in range(nz):
                c = (c >> 4) | (_mc_corners(vol, i, j, k + 1) << 4)
                e = edge_table[c]
                n_v += ((e >> 5) & 1) + ((e >> 6) & 1) + ((e >> 10) & 1)
                n_f += tri_count[c]
        n_verts[i] = n_v
        n_faces[i] = n_f


_mc_counts_numba_serial = serial_twin(_mc_counts_numba)
# Padded volume voxels from which the cube counts run in parallel (crossover about 110,000)
_MC_COUNTS_PARALLEL_MIN = 100_000


@jit(nopython=True, cache=True)  # type: ignore
def _marching_cubes_numba(
    vol: npt.NDArray[np.uint8],
    edge_table: npt.NDArray[np.int32],
    tri_table: npt.NDArray[np.int8],
    tri_count: npt.NDArray[np.int32],
    n_v: int,
    n_f: int,
    offset: npt.NDArray[np.float64],
    spacing: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Marching cubes of a 0/1 volume with a zero border, at the isovalue 0.5.

    Gives the mesh of mc::marching_cubes in PyMCubes 0.1.6 (marchingcubes.h; BSD 3-Clause,
    Copyright (c) 2012-2015, P. M. Neila; see NOTICE): the same vertices and faces, in the
    same order. It walks the cubes in the same order and keeps PyMCubes' vertex numbers of
    the edges on the last two x planes. The zero border makes two steps simpler: no
    surface crosses an edge on the first x, y or z face, so only edges 5, 6 and 10 make new
    vertices; and each vertex is the midpoint of its 0/1 edge, which is exactly the value
    of PyMCubes' interpolation. `_mc_counts_numba` counts the vertices (n_v) and the face
    corners (n_f) first, so the arrays get their final size at once. A vertex v (padded
    voxel units) is stored as (v - 1 + offset) * spacing, the physical position.
    """
    nx, ny, nz = vol.shape[0] - 1, vol.shape[1] - 1, vol.shape[2] - 1
    o0, o1, o2 = offset[0], offset[1], offset[2]
    s0, s1, s2 = spacing[0], spacing[1], spacing[2]
    verts = np.empty((n_v, 3), dtype=np.float64)
    faces = np.empty(n_f, dtype=np.int64)
    # Vertex numbers of the edges on the two latest x planes (index: i % 2, j, k).
    sx = np.zeros((2, ny + 1, nz + 1), dtype=np.int64)
    sy = np.zeros((2, ny + 1, nz + 1), dtype=np.int64)
    sz = np.zeros((2, ny + 1, nz + 1), dtype=np.int64)
    idx = np.zeros(12, dtype=np.int64)
    nv = 0
    nf = 0
    for i in range(nx):
        p = i % 2
        q = 1 - p
        for j in range(ny):
            c = _mc_corners(vol, i, j, 0) << 4
            for k in range(nz):
                c = (c >> 4) | (_mc_corners(vol, i, j, k + 1) << 4)
                e = edge_table[c]
                if e == 0:
                    continue
                if e & 0x040:
                    verts[nv, 0] = (i + 0.5 - 1.0 + o0) * s0
                    verts[nv, 1] = (j + 1.0 - 1.0 + o1) * s1
                    verts[nv, 2] = (k + 1.0 - 1.0 + o2) * s2
                    idx[6] = nv
                    sx[q, j + 1, k + 1] = nv
                    nv += 1
                if e & 0x020:
                    verts[nv, 0] = (i + 1.0 - 1.0 + o0) * s0
                    verts[nv, 1] = (j + 0.5 - 1.0 + o1) * s1
                    verts[nv, 2] = (k + 1.0 - 1.0 + o2) * s2
                    idx[5] = nv
                    sy[q, j + 1, k + 1] = nv
                    nv += 1
                if e & 0x400:
                    verts[nv, 0] = (i + 1.0 - 1.0 + o0) * s0
                    verts[nv, 1] = (j + 1.0 - 1.0 + o1) * s1
                    verts[nv, 2] = (k + 0.5 - 1.0 + o2) * s2
                    idx[10] = nv
                    sz[q, j + 1, k + 1] = nv
                    nv += 1
                # The other edges were numbered by an earlier cube; the triangle
                # table reads only the crossed ones.
                idx[0] = sx[q, j, k]
                idx[1] = sy[q, j + 1, k]
                idx[2] = sx[q, j + 1, k]
                idx[3] = sy[p, j + 1, k]
                idx[4] = sx[q, j, k + 1]
                idx[7] = sy[p, j + 1, k + 1]
                idx[8] = sz[p, j, k + 1]
                idx[9] = sz[q, j, k + 1]
                idx[11] = sz[p, j + 1, k + 1]
                for m in range(tri_count[c]):
                    faces[nf] = idx[tri_table[c, m]]
                    nf += 1
    return verts, faces.reshape(-1, 3)


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _marching_cubes_parallel_numba(
    vol: npt.NDArray[np.uint8],
    edge_table: npt.NDArray[np.int32],
    tri_table: npt.NDArray[np.int8],
    tri_count: npt.NDArray[np.int32],
    v_start: npt.NDArray[np.int64],
    f_start: npt.NDArray[np.int64],
    offset: npt.NDArray[np.float64],
    spacing: npt.NDArray[np.float64],
    verts: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int64],
) -> None:
    """`_marching_cubes_numba` with the x planes of cubes in parallel: the same vertices
    and faces in the same order. Plane i writes its vertices from v_start[i] and its face
    corners from f_start[i] (the counts of `_mc_counts_numba` before plane i). It counts
    plane i - 1 again for the numbers of the vertices on the edges of their common face."""
    nx, ny, nz = vol.shape[0] - 1, vol.shape[1] - 1, vol.shape[2] - 1
    o0, o1, o2 = offset[0], offset[1], offset[2]
    s0, s1, s2 = spacing[0], spacing[1], spacing[2]
    for i in prange(nx):
        # The vertex numbers of the y and z edges at x = i, made by plane i - 1
        sy_prev = np.zeros((ny + 1, nz + 1), dtype=np.int64)
        sz_prev = np.zeros((ny + 1, nz + 1), dtype=np.int64)
        if i > 0:
            nv = v_start[i - 1]
            for j in range(ny):
                c = _mc_corners(vol, i - 1, j, 0) << 4
                for k in range(nz):
                    c = (c >> 4) | (_mc_corners(vol, i - 1, j, k + 1) << 4)
                    e = edge_table[c]
                    if e & 0x040:
                        nv += 1
                    if e & 0x020:
                        sy_prev[j + 1, k + 1] = nv
                        nv += 1
                    if e & 0x400:
                        sz_prev[j + 1, k + 1] = nv
                        nv += 1
        sx = np.zeros((ny + 1, nz + 1), dtype=np.int64)
        sy = np.zeros((ny + 1, nz + 1), dtype=np.int64)
        sz = np.zeros((ny + 1, nz + 1), dtype=np.int64)
        idx = np.zeros(12, dtype=np.int64)
        nv = v_start[i]
        nf = f_start[i]
        for j in range(ny):
            c = _mc_corners(vol, i, j, 0) << 4
            for k in range(nz):
                c = (c >> 4) | (_mc_corners(vol, i, j, k + 1) << 4)
                e = edge_table[c]
                if e == 0:
                    continue
                if e & 0x040:
                    verts[nv, 0] = (i + 0.5 - 1.0 + o0) * s0
                    verts[nv, 1] = (j + 1.0 - 1.0 + o1) * s1
                    verts[nv, 2] = (k + 1.0 - 1.0 + o2) * s2
                    idx[6] = nv
                    sx[j + 1, k + 1] = nv
                    nv += 1
                if e & 0x020:
                    verts[nv, 0] = (i + 1.0 - 1.0 + o0) * s0
                    verts[nv, 1] = (j + 0.5 - 1.0 + o1) * s1
                    verts[nv, 2] = (k + 1.0 - 1.0 + o2) * s2
                    idx[5] = nv
                    sy[j + 1, k + 1] = nv
                    nv += 1
                if e & 0x400:
                    verts[nv, 0] = (i + 1.0 - 1.0 + o0) * s0
                    verts[nv, 1] = (j + 1.0 - 1.0 + o1) * s1
                    verts[nv, 2] = (k + 0.5 - 1.0 + o2) * s2
                    idx[10] = nv
                    sz[j + 1, k + 1] = nv
                    nv += 1
                idx[0] = sx[j, k]
                idx[1] = sy[j + 1, k]
                idx[2] = sx[j + 1, k]
                idx[3] = sy_prev[j + 1, k]
                idx[4] = sx[j, k + 1]
                idx[7] = sy_prev[j + 1, k + 1]
                idx[8] = sz_prev[j, k + 1]
                idx[9] = sz[j, k + 1]
                idx[11] = sz_prev[j + 1, k + 1]
                for m in range(tri_count[c]):
                    faces[nf] = idx[tri_table[c, m]]
                    nf += 1


# From this many voxels of the padded volume on (and with more than one thread), the x
# planes of the marching cubes run in parallel. Measured on 10 threads: 0.57 times the
# serial time on a 52^3 volume, 0.30 on 152^3; slower on 27^3 and with one thread.
_MESH_PARALLEL_MIN = 1 << 17


def _mesh(
    padded: npt.NDArray[np.uint8],
    offset: npt.NDArray[np.float64],
    spacing: npt.NDArray[np.float64],
    parallel: Optional[bool] = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """The marching cubes mesh of a 0/1 volume with a zero border: a parallel count pass,
    then `_marching_cubes_numba` (vertices as (v - 1 + offset) * spacing). `parallel` picks
    its parallel form (default: from _MESH_PARALLEL_MIN voxels on, with more than one
    thread; the warm-up asks for it on a small volume)."""
    n_verts = np.empty(padded.shape[0] - 1, dtype=np.int64)
    n_faces = np.empty(padded.shape[0] - 1, dtype=np.int64)
    counts = sized(_mc_counts_numba, _mc_counts_numba_serial, padded.size, _MC_COUNTS_PARALLEL_MIN)
    counts(padded, EDGE_TABLE, TRIANGLE_COUNT, n_verts, n_faces)
    if parallel is None:
        parallel = padded.size >= _MESH_PARALLEL_MIN and get_num_threads() > 1
    if parallel:
        verts = np.empty((int(n_verts.sum()), 3), dtype=np.float64)
        faces = np.empty(int(n_faces.sum()), dtype=np.int64)
        _marching_cubes_parallel_numba(
            padded,
            EDGE_TABLE,
            TRIANGLE_TABLE,
            TRIANGLE_COUNT,
            np.cumsum(n_verts) - n_verts,
            np.cumsum(n_faces) - n_faces,
            offset,
            spacing,
            verts,
            faces,
        )
        return verts, faces.reshape(-1, 3)
    return cast(
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]],
        _marching_cubes_numba(
            padded,
            EDGE_TABLE,
            TRIANGLE_TABLE,
            TRIANGLE_COUNT,
            int(n_verts.sum()),
            int(n_faces.sum()),
            offset,
            spacing,
        ),
    )


# Faces per partial sum of the mesh area and volume. A fixed block size, not one block
# per thread, so the sums do not depend on the number of threads.
_MESH_SUM_BLOCK = 4096


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _mesh_area_volume_numba(
    verts: npt.NDArray[np.floating[Any]], faces: npt.NDArray[np.floating[Any]]
) -> tuple[float, float]:
    """Compute mesh surface area and absolute volume in one pass. The faces add in
    blocks of _MESH_SUM_BLOCK and the block sums in order, so the result is the same
    for every number of threads."""
    n_faces = faces.shape[0]
    n_blocks = (n_faces + _MESH_SUM_BLOCK - 1) // _MESH_SUM_BLOCK
    block_area = np.empty(n_blocks, dtype=np.float64)
    block_vol6 = np.empty(n_blocks, dtype=np.float64)
    for b in prange(n_blocks):
        block_area[b], block_vol6[b] = _mesh_block_sums(
            verts, faces, b * _MESH_SUM_BLOCK, min(n_faces, (b + 1) * _MESH_SUM_BLOCK)
        )

    area = 0.0
    vol6 = 0.0
    for b in range(n_blocks):
        area += block_area[b]
        vol6 += block_vol6[b]
    vol = vol6 / 6.0
    if vol < 0.0:
        vol = -vol
    return float(area), float(vol)


_mesh_area_volume_numba_serial = serial_twin(_mesh_area_volume_numba)
# Mesh faces from which the area and volume sums run in parallel (crossover about 50,000)
_MESH_SUM_PARALLEL_MIN = 12 * _MESH_SUM_BLOCK


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _mesh_block_sums(
    verts: npt.NDArray[np.floating[Any]],
    faces: npt.NDArray[np.floating[Any]],
    start: int,
    stop: int,
) -> tuple[float, float]:
    """Surface area and six times the signed volume of the faces start to stop."""
    area = 0.0
    vol6 = 0.0
    for f in range(start, stop):
        i0 = faces[f, 0]
        i1 = faces[f, 1]
        i2 = faces[f, 2]

        v0x = verts[i0, 0]
        v0y = verts[i0, 1]
        v0z = verts[i0, 2]
        v1x = verts[i1, 0]
        v1y = verts[i1, 1]
        v1z = verts[i1, 2]
        v2x = verts[i2, 0]
        v2y = verts[i2, 1]
        v2z = verts[i2, 2]

        # Surface area: 0.5 * ||(v1 - v0) x (v2 - v0)||
        e1x = v1x - v0x
        e1y = v1y - v0y
        e1z = v1z - v0z
        e2x = v2x - v0x
        e2y = v2y - v0y
        e2z = v2z - v0z

        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        area += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)

        # Volume via divergence theorem
        c1x = v1y * v2z - v1z * v2y
        c1y = v1z * v2x - v1x * v2z
        c1z = v1x * v2y - v1y * v2x
        vol6 += v0x * c1x + v0y * c1y + v0z * c1z
    return area, vol6


@jit(nopython=True, nogil=True, fastmath=True, cache=True)  # type: ignore
def _mvee_khachiyan_numba(
    points: npt.NDArray[np.floating[Any]], tol: float = 0.001
) -> tuple[Optional[npt.NDArray[np.floating[Any]]], Optional[npt.NDArray[np.floating[Any]]]]:
    """
    Find Minimum Volume Enclosing Ellipsoid (MVEE) using the Khachiyan algorithm.

    Optimized with Numba for performance:
    1. Transposed Q layout (N, d+1) for cache locality.
    2. Rank-1 updates (Sherman-Morrison) for matrix inversion.
    3. Periodic full recomputation for numerical stability.
    4. Pre-allocated working arrays to minimize memory churn.
    5. The products q_r q_c (r <= c) of each point are kept, so the quadratic form of a
       point is one dot product with the matching weights of invX (10 products, not 20).

    Args:
        points: Array of points (N, d).
        tol: Tolerance for convergence.

    Returns:
        Tuple containing:
            - A: The shape matrix (d, d).
            - c: The center vector (d,).
            Returns (None, None) if calculation fails.
    """
    N, d = points.shape
    d1 = d + 1

    # 1. Optimize Memory Layout: Q as (N, d+1)
    # Contiguous memory for points access in the inner loop
    Q = np.empty((N, d1), dtype=np.float64)
    for k in range(N):
        for i in range(d):
            Q[k, i] = points[k, i]
        Q[k, d] = 1.0
    # The products q_r q_c (r <= c) of each point, and their weights in q^T invX q
    n_prod = d1 * (d1 + 1) // 2
    P = np.empty((N, n_prod), dtype=np.float64)
    for k in range(N):
        m = 0
        for r in range(d1):
            for c_idx in range(r, d1):
                P[k, m] = Q[k, r] * Q[k, c_idx]
                m += 1
    weights = np.empty(n_prod, dtype=np.float64)

    # Initialize weights u
    u = np.ones(N, dtype=np.float64) / N

    # Initial X = (1/N) * Q.T @ Q
    # Compute X explicitly
    X = np.zeros((d1, d1), dtype=np.float64)
    for k in range(N):
        for r in range(d1):
            val_r = Q[k, r]
            for c_idx in range(d1):
                X[r, c_idx] += val_r * Q[k, c_idx]

    X /= N

    try:
        invX = np.linalg.inv(X)
    except Exception:
        return None, None

    err = 1.0
    count = 0

    # Pre-allocate work arrays
    tmp_vec = np.zeros(d1, dtype=np.float64)

    while err > tol and count < 1000:
        # Find point with max Mahalanobis distance
        # M_k = Q[k] @ invX @ Q[k].T
        max_val = -1.0
        j = -1

        # Bottleneck loop: O(N * d^2). The quadratic form q_k^T * invX * q_k is the dot
        # product of the kept products of q_k with the weights of invX.
        m = 0
        for r in range(d1):
            for c_idx in range(r, d1):
                weights[m] = invX[r, r] if r == c_idx else invX[r, c_idx] + invX[c_idx, r]
                m += 1
        for k in range(N):
            val = 0.0
            for m in range(n_prod):
                val += P[k, m] * weights[m]

            if val > max_val:
                max_val = val
                j = k

        step_size = (max_val - d1) / (d1 * (max_val - 1))

        # Update u
        sum_u_sq = np.sum(u**2)
        err_sq = step_size**2 * (sum_u_sq - u[j] ** 2 + (1 - u[j]) ** 2)
        err = np.sqrt(err_sq)

        new_u_j = u[j] * (1 - step_size) + step_size
        u *= 1 - step_size
        u[j] = new_u_j

        # Rank-1 Update of invX (Sherman-Morrison)
        # Recompute fully every 50 iterations to prevent numerical drift
        if count % 50 == 0 and count > 0:
            X.fill(0.0)
            for k in range(N):
                uk = u[k]
                for r in range(d1):
                    val_r = Q[k, r]
                    for c_idx in range(d1):
                        X[r, c_idx] += uk * val_r * Q[k, c_idx]
            try:
                invX = np.linalg.inv(X)
            except Exception:
                return None, None
        else:
            # Fast update
            alpha = step_size / (1.0 - step_size)

            # Compute v = invX @ Q[j]
            for r in range(d1):
                tmp_vec[r] = 0.0
                for c_idx in range(d1):
                    tmp_vec[r] += invX[r, c_idx] * Q[j, c_idx]

            # Denominator
            denom = 1.0 + alpha * max_val

            # Update invX
            factor = alpha / denom
            for r in range(d1):
                val_r = tmp_vec[r]
                for c_idx in range(d1):
                    invX[r, c_idx] -= factor * val_r * tmp_vec[c_idx]

            # Scale by 1/(1-step)
            invX *= 1.0 / (1.0 - step_size)

        count += 1

    # Calculate center c
    c = np.zeros(d, dtype=np.float64)
    for i in range(d):
        sum_val = 0.0
        for k in range(N):
            sum_val += points[k, i] * u[k]
        c[i] = sum_val

    # Calculate A matrix
    Cov = np.zeros((d, d), dtype=np.float64)
    for k in range(N):
        uk = u[k]
        for r in range(d):
            val_r = points[k, r]
            for c_idx in range(d):
                Cov[r, c_idx] += uk * val_r * points[k, c_idx]

    for r in range(d):
        for c_idx in range(d):
            Cov[r, c_idx] -= c[r] * c[c_idx]

    try:
        A = (1.0 / d) * np.linalg.inv(Cov)
    except Exception:
        return None, None

    return A, c


def _calculate_ellipsoid_surface_area(a: float, b: float, c: float) -> float:
    """
    Approximate surface area of an ellipsoid using Legendre polynomials series.
    Based on IBSI RDD2 definition.

    Args:
        a, b, c: Semi-axis lengths.

    Returns:
        Approximated surface area.
    """
    # Sort axes a >= b >= c
    axes = np.sort([a, b, c])[::-1]
    a, b, c = float(axes[0]), float(axes[1]), float(axes[2])

    if a == 0 or b == 0 or c == 0:
        return 0.0

    if a == c:  # Sphere
        return float(4 * np.pi * a**2)

    alpha = np.sqrt(1 - (b / a) ** 2)
    beta = np.sqrt(1 - (c / a) ** 2)

    # Handle special cases where alpha or beta is 0 (spheroids)
    if alpha == 0:  # a = b (oblate spheroid)
        e = np.sqrt(1 - (c / a) ** 2)
        # Note: e cannot be 0 here because a == c case is handled above
        return float(2 * np.pi * a**2 + np.pi * (c**2 / e) * np.log((1 + e) / (1 - e)))

    # General case approximation
    total_sum = 0.0
    x = (alpha**2 + beta**2) / (2 * alpha * beta)
    for v in range(21):  # 0 to 20
        pv = eval_legendre(v, x)
        term = ((alpha * beta) ** v / (1 - 4 * v**2)) * pv
        total_sum += term

    area = 4 * np.pi * a * b * total_sum
    return float(area)


def _uint8_roi(mask: npt.NDArray[Any], keep: bool = False) -> npt.NDArray[np.uint8]:
    """The ROI (`mask != 0`) as a row-order uint8 array, so the kernels compile for one
    mask type and layout. With `keep`, a uint8 mask (the common case) is used as it is."""
    if keep and mask.dtype == np.uint8:
        return cast(npt.NDArray[np.uint8], mask)
    return cast(npt.NDArray[np.uint8], np.not_equal(mask, 0, order="C").view(np.uint8))


def _get_mesh_features(
    mask: Image,
    roi_bbox: Optional[tuple[slice, slice, slice]] = None,
    grid_offset: Optional[tuple[int, int, int]] = None,
) -> tuple[
    dict[str, float],
    Optional[npt.NDArray[np.floating[Any]]],
    Optional[npt.NDArray[np.floating[Any]]],
]:
    """
    Calculate mesh-based features (Surface Area, Volume) and return mesh data.

    The marching cubes kernel gives the PyMCubes mesh (the same vertices and faces, in
    the same order), which produces IBSI-compliant results for the digital phantom.

    Optimization: Crops mask to bounding box before mesh generation for large sparse ROIs.
    `roi_bbox` may pass a precomputed nonzero bounding box to skip the scan.
    """
    features: dict[str, float] = {}

    # Scan the original mask for the bbox, then binarise only the cropped region:
    # this avoids the two full-volume temporaries of binarising up front.
    bbox = roi_bbox if roi_bbox is not None else compute_nonzero_bbox(mask.array)
    if bbox is None:
        return {}, None, None

    mask_cropped = _uint8_roi(mask.array[bbox])
    origin_offset = np.array([bbox[0].start, bbox[1].start, bbox[2].start], dtype=np.float64)
    if grid_offset is not None:  # the mask is a region of a larger grid
        origin_offset += np.array(grid_offset, dtype=np.float64)

    # The vertices come in physical units: padding (-1), bbox offset, spacing
    verts, faces = _mesh(
        np.pad(mask_cropped, 1), origin_offset, np.asarray(mask.spacing, dtype=np.float64)
    )
    if len(faces) == 0:  # a bbox with no ROI voxel
        return {}, None, None

    area_volume = sized(
        _mesh_area_volume_numba,
        _mesh_area_volume_numba_serial,
        len(faces),
        _MESH_SUM_PARALLEL_MIN,
    )
    surface_area, mesh_volume = area_volume(verts, faces)
    features["surface_area_C0JK"] = float(surface_area)
    features["volume_RNU0"] = float(mesh_volume)

    return features, verts, faces  # type: ignore[return-value]


def _get_shape_features(surface_area: float, mesh_volume: float) -> dict[str, float]:
    """Calculate shape features based on mesh volume and area."""
    features: dict[str, float] = {}
    if mesh_volume <= 0 or surface_area <= 0:
        return features

    features["surface_to_volume_ratio_2PR5"] = surface_area / mesh_volume
    features["compactness_1_SKGS"] = mesh_volume / (np.sqrt(np.pi) * (surface_area**1.5))
    features["compactness_2_BQWJ"] = (36 * np.pi * (mesh_volume**2)) / (surface_area**3)
    features["spherical_disproportion_KRCK"] = surface_area / (
        (36 * np.pi * (mesh_volume**2)) ** (1 / 3)
    )
    features["sphericity_QCFX"] = ((36 * np.pi * (mesh_volume**2)) ** (1 / 3)) / surface_area
    features["asphericity_25C7"] = ((1 / (36 * np.pi)) * (surface_area**3) / (mesh_volume**2)) ** (
        1 / 3
    ) - 1

    return features


def _get_pca_features(
    mask: Image,
    mesh_volume: float,
    surface_area: float,
    mask_moments: Optional[
        tuple[float, float, float, float, float, float, float, float, float, float]
    ] = None,
) -> tuple[
    dict[str, float],
    Optional[npt.NDArray[np.floating[Any]]],
    Optional[npt.NDArray[np.floating[Any]]],
]:
    """Calculate PCA-based features and return eigenvalues/vectors."""
    features: dict[str, float] = {}

    if mask_moments is not None:
        # The moments may come from a bbox-cropped scan; only the covariance is
        # used below, which is invariant to that index translation.
        n, s0, s1, s2, s00, s11, s22, s01, s02, s12 = mask_moments
    else:
        moments = sized(
            _accumulate_moments_from_mask_numba,
            _accumulate_moments_from_mask_numba_serial,
            mask.array.size,
            _MOMENTS_PARALLEL_MIN,
        )
        n, s0, s1, s2, s00, s11, s22, s01, s02, s12 = moments(mask.array)
    if n <= 3:
        return features, None, None

    # Compute sample covariance of physical coordinates without materializing (N,3) arrays.
    mean0 = s0 / n
    mean1 = s1 / n
    mean2 = s2 / n

    denom = float(n - 1)
    c00 = (s00 - n * mean0 * mean0) / denom
    c11 = (s11 - n * mean1 * mean1) / denom
    c22 = (s22 - n * mean2 * mean2) / denom
    c01 = (s01 - n * mean0 * mean1) / denom
    c02 = (s02 - n * mean0 * mean2) / denom
    c12 = (s12 - n * mean1 * mean2) / denom

    sp = np.asarray(mask.spacing, dtype=np.float64)
    cov = np.array(
        [
            [c00 * sp[0] * sp[0], c01 * sp[0] * sp[1], c02 * sp[0] * sp[2]],
            [c01 * sp[1] * sp[0], c11 * sp[1] * sp[1], c12 * sp[1] * sp[2]],
            [c02 * sp[2] * sp[0], c12 * sp[2] * sp[1], c22 * sp[2] * sp[2]],
        ],
        dtype=np.float64,
    )
    evals, evecs = np.linalg.eigh(cov)

    # Sort descending
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    evals[evals < 0] = 0

    lambda_major, lambda_minor, lambda_least = evals[0], evals[1], evals[2]

    features["major_axis_length_TDIC"] = 4 * np.sqrt(lambda_major)
    features["minor_axis_length_P9VJ"] = 4 * np.sqrt(lambda_minor)
    features["least_axis_length_7J51"] = 4 * np.sqrt(lambda_least)

    if lambda_major > 0:
        features["elongation_Q3CK"] = np.sqrt(lambda_minor / lambda_major)
        features["flatness_N17B"] = np.sqrt(lambda_least / lambda_major)

    # Ellipsoid Features
    a, b, c = (
        2 * np.sqrt(lambda_major),
        2 * np.sqrt(lambda_minor),
        2 * np.sqrt(lambda_least),
    )
    vol_aee = (4 * np.pi / 3) * a * b * c
    if vol_aee > 0:
        features["volume_density_aee_6BDE"] = mesh_volume / vol_aee

    area_aee = _calculate_ellipsoid_surface_area(a, b, c)
    if area_aee > 0:
        features["area_density_aee_RDD2"] = surface_area / area_aee

    return features, evals, evecs


def _get_convex_hull_features(
    verts: npt.NDArray[np.floating[Any]],
    mesh_volume: float,
    surface_area: float,
    spacing: tuple[float, float, float],
    serial: bool = False,
) -> tuple[dict[str, float], Optional[npt.NDArray[np.float64]]]:
    """Calculate Convex Hull features.

    `verts` are the marching cubes vertices of `_get_mesh_features`. The hull is that of
    the vertices that can be hull vertices (see `_hull_candidates_numba`), exact on their
    lattice coordinates (`_exact_hull_numba`; Qhull when it finds no hull). It has the hull
    vertices of Qhull in the same order, and the same volume and area to about 1e-15. It
    returns the features and the hull vertices (for the MVEE). With
    `serial`, the maximum diameter comes from the serial kernel (the same value).
    """
    features: dict[str, float] = {}
    if len(verts) <= 3:
        return features, None

    try:
        grid_spacing = np.asarray(spacing, dtype=np.float64)
        points = np.asarray(verts[_hull_candidates_numba(verts, grid_spacing)], dtype=np.float64)
        found, vertices, triangles = _exact_hull_numba(
            np.rint(2.0 * points / grid_spacing).astype(np.int64)
        )
        if found:
            area_convex, vol_convex = _hull_area_volume_numba(points, triangles)
            hull_points = points[vertices]
        else:  # fewer than four points, or all in one plane: Qhull raises there
            hull = ConvexHull(points)
            vol_convex, area_convex = hull.volume, hull.area
            hull_points = np.asarray(hull.points[hull.vertices], dtype=np.float64)

        if vol_convex > 0:
            features["volume_density_convex_hull_R3ER"] = mesh_volume / vol_convex
        if area_convex > 0:
            features["area_density_convex_hull_7T7F"] = surface_area / area_convex

        # Max 3D Diameter
        if hull_points.shape[0] > 1:
            diameter = (
                _max_pairwise_distance_serial_numba if serial else _max_pairwise_distance_numba
            )
            features["maximum_3d_diameter_L0JK"] = float(diameter(hull_points))

        return features, hull_points
    except Exception:
        return features, None


def _get_bounding_box_features(
    verts: npt.NDArray[np.floating[Any]],
    evecs: Optional[npt.NDArray[np.floating[Any]]],
    mesh_volume: float,
    surface_area: float,
) -> dict[str, float]:
    """Calculate AABB and OMBB features."""
    features: dict[str, float] = {}
    if len(verts) == 0:
        return features

    # AABB, and the vertex mean (the OMBB centre), from one pass
    min_bound, max_bound, center = _column_stats_numba(np.asarray(verts, dtype=np.float64))
    dims = max_bound - min_bound
    vol_aabb = np.prod(dims)
    area_aabb = 2 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[2] * dims[0])

    if vol_aabb > 0:
        features["volume_density_aabb_PBX1"] = mesh_volume / vol_aabb
    if area_aabb > 0:
        features["area_density_aabb_R59B"] = surface_area / area_aabb

    # OMBB
    if evecs is not None:
        # Deterministic streaming extents in Numba (avoids allocating rotated_verts and Python loop overhead)
        extents = sized(
            _ombb_extents_numba, _ombb_extents_numba_serial, len(verts), _OMBB_PARALLEL_MIN
        )
        min_rot, max_rot = extents(
            np.asarray(verts, dtype=np.float64),
            np.asarray(center, dtype=np.float64),
            np.asarray(evecs, dtype=np.float64),
        )

        dims_rot = max_rot - min_rot
        vol_ombb = float(np.prod(dims_rot))
        area_ombb = float(
            2 * (dims_rot[0] * dims_rot[1] + dims_rot[1] * dims_rot[2] + dims_rot[2] * dims_rot[0])
        )

        if vol_ombb > 0:
            features["volume_density_ombb_ZH1A"] = mesh_volume / vol_ombb
        if area_ombb > 0:
            features["area_density_ombb_IQYR"] = surface_area / area_ombb

    return features


def _get_mvee_features(
    hull_points: Optional[npt.NDArray[np.float64]],
    mesh_volume: float,
    surface_area: float,
) -> dict[str, float]:
    """Calculate MVEE features of the convex hull vertices (float64, in order)."""
    features: dict[str, float] = {}
    if hull_points is None:
        return features

    A_mvee, _ = _mvee_khachiyan_numba(hull_points)

    if A_mvee is not None:
        evals_mvee, _ = np.linalg.eigh(A_mvee)
        evals_mvee[evals_mvee < 0] = 0

        with np.errstate(divide="ignore"):
            semi_axes_mvee = 1.0 / np.sqrt(evals_mvee)

        if np.all(np.isfinite(semi_axes_mvee)):
            vol_mvee = (4 * np.pi / 3) * np.prod(semi_axes_mvee)
            area_mvee = _calculate_ellipsoid_surface_area(*semi_axes_mvee)

            if vol_mvee > 0:
                features["volume_density_mvee_SWZ1"] = mesh_volume / vol_mvee
            if area_mvee > 0:
                features["area_density_mvee_BRI8"] = surface_area / area_mvee

    return features


def _get_intensity_morphology_features(
    mask: Image,
    image: Image,
    intensity_mask: Image,
    mesh_volume: float,
    mask_moments: Optional[
        tuple[float, float, float, float, float, float, float, float, float, float]
    ] = None,
    mask_bbox: Optional[tuple[slice, slice, slice]] = None,
) -> dict[str, float]:
    """Calculate intensity-weighted morphological features.

    When `mask_bbox` is given, `mask_moments` must have been computed on
    `mask.array[mask_bbox]`; the bbox offset is added back to the geometric
    center of mass here.
    """
    features: dict[str, float] = {}

    # Crop to the intensity mask's nonzero bbox before scanning: this avoids a
    # full read of the (large, float64) image. Reuse the morph bbox when the
    # masks share one array; otherwise scan the mask for its own bbox.
    i_bbox: Optional[tuple[slice, slice, slice]]
    if intensity_mask.array is mask.array and mask_bbox is not None:
        i_bbox = mask_bbox
    else:
        i_bbox = compute_nonzero_bbox(intensity_mask.array)
    if i_bbox is None:
        return features

    # The kernel reads a float64 or float32 image, the types that the import compiles; an
    # image of another type goes as a float64 copy of the box (the same values).
    values = image.array[i_bbox]
    if values.dtype not in (np.float64, np.float32):
        values = values.astype(np.float64)
    roi = _uint8_roi(intensity_mask.array[i_bbox], keep=True)
    weighted = sized(
        _accumulate_intensity_weighted_moments_numba,
        _accumulate_intensity_weighted_moments_numba_serial,
        roi.size,
        _MOMENTS_PARALLEL_MIN,
    )
    count_i, sum_w, sum_i0_w, sum_i1_w, sum_i2_w = weighted(roi, values)
    if count_i > 0:
        mean_intensity = sum_w / float(count_i)
        features["integrated_intensity_99N0"] = mesh_volume * mean_intensity

        # Center of Mass Shift
        if mask_moments is not None:
            n_m, s0_m, s1_m, s2_m = (
                mask_moments[0],
                mask_moments[1],
                mask_moments[2],
                mask_moments[3],
            )
        else:
            part = mask.array[mask_bbox] if mask_bbox is not None else mask.array
            moments = sized(
                _accumulate_moments_from_mask_numba,
                _accumulate_moments_from_mask_numba_serial,
                part.size,
                _MOMENTS_PARALLEL_MIN,
            )
            n_m, s0_m, s1_m, s2_m, _, _, _, _, _, _ = moments(part)
        if n_m > 0 and sum_w != 0.0:
            # The shift between the geometric and the intensity-weighted centre, in index
            # units per axis. The sums are over the cropped arrays, and the crop starts
            # (whole numbers) add after the subtraction, so the shift does not depend on
            # where the arrays start in the image; the origin cancels, so it is not used.
            m_off = [b.start for b in mask_bbox] if mask_bbox is not None else [0, 0, 0]
            spacing = np.asarray(mask.spacing, dtype=np.float64)
            shift = [
                (s_m / float(n_m) - s_w / sum_w + float(m_off[a] - i_bbox[a].start)) * spacing[a]
                for a, (s_m, s_w) in enumerate(
                    ((s0_m, sum_i0_w), (s1_m, sum_i1_w), (s2_m, sum_i2_w))
                )
            ]
            features["center_of_mass_shift_KLMA"] = float(
                math.sqrt(shift[0] * shift[0] + shift[1] * shift[1] + shift[2] * shift[2])
            )

    return features


class _Rest(NamedTuple):
    """The morphology features after the PCA features (see _morphology_first): the input of
    part 2, and the bounding box and intensity-weighted features of part 1 (or their
    errors)."""

    verts: npt.NDArray[np.floating[Any]]
    mesh_volume: float
    surface_area: float
    spacing: tuple[float, float, float]
    box: dict[str, float] | Exception
    intensity: dict[str, float] | Exception


def _section(
    function: Callable[..., dict[str, float]], *args: Any, **kwargs: Any
) -> dict[str, float] | Exception:
    """The features `function(*args, **kwargs)`, or the error that it raises."""
    try:
        return function(*args, **kwargs)
    except Exception as e:
        return e


def _morphology_first(
    mask: Image,
    image: Optional[Image] = None,
    intensity_mask: Optional[Image] = None,
    roi_bbox: Optional[tuple[slice, slice, slice]] = None,
    grid_offset: Optional[tuple[int, int, int]] = None,
) -> tuple[dict[str, float], Optional[_Rest]]:
    """Part 1 of calculate_morphology_features, with all its parallel kernels: the voxel,
    mesh, shape and PCA features, and the rest. The rest is None when these are all the
    features (no ROI voxel, or no mesh). It holds the bounding box and intensity-weighted
    features, or their errors: _morphology_merge raises them in the order of the features.
    """
    features: dict[str, float] = {}
    i_mask = intensity_mask if intensity_mask is not None else mask

    voxel_volume = np.prod(mask.spacing)

    # Compute the ROI bounding box once; the scans below run on the cropped region
    # instead of the full volume, which dominates runtime for sparse ROIs.
    bbox = roi_bbox if roi_bbox is not None else compute_nonzero_bbox(mask.array)
    if bbox is None:
        # Empty mask: no ROI voxels.
        features["volume_voxel_counting_YEKZ"] = 0.0
        return features, None

    # 1. Voxel Based Features + mask moments (shared by PCA and intensity
    # morphology features). The moments are computed in cropped index space: the
    # PCA covariance is translation-invariant, and the center-of-mass consumer
    # adds the bbox offset back. The kernel's voxel count doubles as the count
    # for the voxel-counting volume.
    roi = _uint8_roi(mask.array[bbox], keep=True)
    moments = sized(
        _accumulate_moments_from_mask_numba,
        _accumulate_moments_from_mask_numba_serial,
        roi.size,
        _MOMENTS_PARALLEL_MIN,
    )
    mask_moments = moments(roi)
    n_voxels = mask_moments[0]
    features["volume_voxel_counting_YEKZ"] = float(n_voxels * voxel_volume)

    # 2. Mesh Based Features
    mesh_feats, verts, faces = _get_mesh_features(mask, roi_bbox=bbox, grid_offset=grid_offset)
    features.update(mesh_feats)

    if verts is None or faces is None:
        return features, None

    mesh_volume = features.get("volume_RNU0", 0.0)
    surface_area = features.get("surface_area_C0JK", 0.0)

    # 3. Shape Features
    features.update(_get_shape_features(surface_area, mesh_volume))

    # 4. PCA Based Features
    pca_feats, evals, evecs = _get_pca_features(
        mask, mesh_volume, surface_area, mask_moments=mask_moments
    )
    features.update(pca_feats)

    # 6. Bounding Box Features (5, the convex hull, is in part 2)
    box = _section(_get_bounding_box_features, verts, evecs, mesh_volume, surface_area)

    # 8. Intensity Based Features (7, the MVEE, is in part 2)
    intensity = (
        {}
        if image is None
        else _section(
            _get_intensity_morphology_features,
            mask,
            image,
            i_mask,
            mesh_volume,
            mask_moments=mask_moments,
            mask_bbox=bbox,
        )
    )
    return features, _Rest(verts, mesh_volume, surface_area, mask.spacing, box, intensity)


def _morphology_second(
    rest: _Rest, serial: bool = False
) -> tuple[dict[str, float] | Exception, dict[str, float] | Exception]:
    """Part 2 of calculate_morphology_features: the convex hull features and the MVEE
    features, or the error of each. With `serial`, part 2 runs no parallel kernel, so it can
    run in a thread next to them (numba's workqueue layer stops at two parallel regions at
    once). Its kernels release the GIL, and so does Qhull when it runs."""
    try:
        hull_features, hull_points = _get_convex_hull_features(
            rest.verts, rest.mesh_volume, rest.surface_area, rest.spacing, serial
        )
    except Exception as e:
        return e, {}
    return hull_features, _section(
        _get_mvee_features, hull_points, rest.mesh_volume, rest.surface_area
    )


def _morphology_merge(
    features: dict[str, float],
    rest: _Rest,
    second: tuple[dict[str, float] | Exception, dict[str, float] | Exception],
) -> dict[str, float]:
    """The part 1 `features` with the convex hull, bounding box, MVEE and intensity-weighted
    features, in this order. The first error in this order is raised, as in one pass."""
    hull, mvee = second
    for section in (hull, rest.box, mvee, rest.intensity):
        if isinstance(section, Exception):
            raise section
        features.update(section)
    return features


def calculate_morphology_features(
    mask: Image,
    image: Optional[Image] = None,
    intensity_mask: Optional[Image] = None,
    roi_bbox: Optional[tuple[slice, slice, slice]] = None,
    grid_offset: Optional[tuple[int, int, int]] = None,
) -> dict[str, float]:
    """
    Calculate morphological features from the ROI mask.
    Includes both voxel-based and mesh-based features (IBSI compliant).

    Args:
        mask: Image object containing the morphological mask. Nonzero values
            are treated as ROI membership.
        image: Optional Image object containing intensity data (required for some features).
        intensity_mask: Optional Image object containing the intensity mask (e.g. after outlier filtering).
                        If provided, used for intensity-weighted features (99N0, KLMA).
                        If None, defaults to `mask`.
        roi_bbox: Optional precomputed tight bounding box of the mask's nonzero voxels
                  (tuple of slices, as returned by an internal bbox scan). Skips
                  rescanning the full mask volume. If None, computed internally.
        grid_offset: Optional index of the first voxel of the arrays in a larger grid,
                  when they hold a region of it. The mesh is then made in the index frame
                  of that grid, so the features are those of the whole grid bit for bit
                  (the MVEE fit depends on the frame at about 1e-5).

    Returns:
        Dictionary of calculated features.

    Example:
        ```python
        import numpy as np
        from pictologics.loader import Image
        from pictologics.features.morphology import calculate_morphology_features

        mask_arr = np.zeros((50, 50, 50), dtype=np.uint8)
        mask_arr[10:40, 10:40, 10:40] = 1
        mask = Image(array=mask_arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

        features = calculate_morphology_features(mask)
        print(round(features["volume_voxel_counting_YEKZ"], 1))
        # 27000.0
        print(round(features["sphericity_QCFX"], 2))
        # 0.82
        ```
    """
    features, rest = _morphology_first(mask, image, intensity_mask, roi_bbox, grid_offset)
    if rest is None:
        return features
    return _morphology_merge(features, rest, _morphology_second(rest))
