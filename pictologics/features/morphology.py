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
Uses `numba` for optimizing the Khachiyan algorithm for MVEE calculation.
"""

from __future__ import annotations

import math
from typing import Optional

import mcubes
import numpy as np
from numba import jit, prange
from scipy.spatial import ConvexHull
from scipy.special import eval_legendre

from ..loader import Image


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _accumulate_moments_from_mask_numba(
    mask: np.ndarray,
) -> tuple[int, float, float, float, float, float, float, float, float, float]:
    """Accumulate first/second moments of voxel indices for mask>0."""
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

    d0, d1, d2 = mask.shape
    # Flatten loops for better parallelization or just parallelize outer
    for i in prange(d0):
        for j in range(d1):
            for k in range(d2):
                if mask[i, j, k] > 0:
                    n += 1
                    fi = float(i)
                    fj = float(j)
                    fk = float(k)
                    s0 += fi
                    s1 += fj
                    s2 += fk
                    s00 += fi * fi
                    s11 += fj * fj
                    s22 += fk * fk
                    s01 += fi * fj
                    s02 += fi * fk
                    s12 += fj * fk

    return n, s0, s1, s2, s00, s11, s22, s01, s02, s12


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _accumulate_intensity_weighted_moments_numba(
    mask: np.ndarray, image: np.ndarray
) -> tuple[int, float, float, float, float]:
    """Accumulate intensity-weighted index sums over mask>0."""
    count = 0
    sum_w = 0.0
    sum_i0_w = 0.0
    sum_i1_w = 0.0
    sum_i2_w = 0.0

    d0, d1, d2 = mask.shape
    for i in prange(d0):
        for j in range(d1):
            for k in range(d2):
                if mask[i, j, k] > 0:
                    w = float(image[i, j, k])
                    count += 1
                    sum_w += w
                    sum_i0_w += float(i) * w
                    sum_i1_w += float(j) * w
                    sum_i2_w += float(k) * w

    return count, sum_w, sum_i0_w, sum_i1_w, sum_i2_w


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _ombb_extents_numba(
    verts: np.ndarray, center: np.ndarray, evecs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    min_rot = np.empty(3, dtype=np.float64)
    max_rot = np.empty(3, dtype=np.float64)
    min_rot[:] = np.inf
    max_rot[:] = -np.inf

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


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _max_pairwise_distance_numba(points: np.ndarray) -> float:
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


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _mesh_area_volume_numba(
    verts: np.ndarray, faces: np.ndarray
) -> tuple[float, float]:
    """Compute mesh surface area and absolute volume in one deterministic pass."""
    n_faces = faces.shape[0]
    area = 0.0
    vol6 = 0.0

    # Parallel reduction for sum of area and volume
    for f in prange(n_faces):
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

    vol = vol6 / 6.0
    if vol < 0.0:
        vol = -vol
    return float(area), float(vol)


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _mvee_khachiyan_numba(
    points: np.ndarray, tol: float = 0.001
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find Minimum Volume Enclosing Ellipsoid (MVEE) using the Khachiyan algorithm.

    Optimized with Numba for performance:
    1. Transposed Q layout (N, d+1) for cache locality.
    2. Rank-1 updates (Sherman-Morrison) for matrix inversion.
    3. Periodic full recomputation for numerical stability.
    4. Pre-allocated working arrays to minimize memory churn.

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

        # Bottleneck loop: O(N * d^2)
        for k in range(N):
            val = 0.0
            # Compute quadratic form: q_k^T * invX * q_k
            for r in range(d1):
                dot_val = 0.0
                for c_idx in range(d1):
                    dot_val += invX[r, c_idx] * Q[k, c_idx]
                val += Q[k, r] * dot_val

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


def _get_mesh_features(
    mask: Image,
) -> tuple[dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculate mesh-based features (Surface Area, Volume) and return mesh data.

    Uses PyMCubes for marching cubes mesh generation, which produces IBSI-compliant
    results for the digital phantom.
    """
    features: dict[str, float] = {}
    mask_arr = (mask.array > 0).astype(np.uint8, copy=False)
    mask_padded_u8 = np.pad(mask_arr, 1, mode="constant", constant_values=0)
    mask_padded = np.ascontiguousarray(mask_padded_u8.astype(np.float32, copy=False))

    try:
        # Use PyMCubes marching cubes implementation
        verts, faces = mcubes.marching_cubes(mask_padded, 0.5)

        if len(verts) == 0 or len(faces) == 0:
            return {}, None, None

        # Apply spacing to vertices
        verts = np.asarray(verts, dtype=np.float64) * np.asarray(
            mask.spacing, dtype=np.float64
        )
        faces_i64 = np.asarray(faces, dtype=np.int64)

        surface_area, mesh_volume = _mesh_area_volume_numba(verts, faces_i64)
        features["surface_area_C0JK"] = float(surface_area)
        features["volume_RNU0"] = float(mesh_volume)

        return features, verts, faces_i64
    except (ValueError, RuntimeError):
        # Marching cubes failed
        return {}, None, None


def _get_shape_features(surface_area: float, mesh_volume: float) -> dict[str, float]:
    """Calculate shape features based on mesh volume and area."""
    features: dict[str, float] = {}
    if mesh_volume <= 0 or surface_area <= 0:
        return features

    features["surface_to_volume_ratio_2PR5"] = surface_area / mesh_volume
    features["compactness_1_SKGS"] = mesh_volume / (
        np.sqrt(np.pi) * (surface_area**1.5)
    )
    features["compactness_2_BQWJ"] = (36 * np.pi * (mesh_volume**2)) / (surface_area**3)
    features["spherical_disproportion_KRCK"] = surface_area / (
        (36 * np.pi * (mesh_volume**2)) ** (1 / 3)
    )
    features["sphericity_QCFX"] = (
        (36 * np.pi * (mesh_volume**2)) ** (1 / 3)
    ) / surface_area
    features["asphericity_25C7"] = (
        (1 / (36 * np.pi)) * (surface_area**3) / (mesh_volume**2)
    ) ** (1 / 3) - 1

    return features


def _get_pca_features(
    mask: Image, mesh_volume: float, surface_area: float
) -> tuple[dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Calculate PCA-based features and return eigenvalues/vectors."""
    features: dict[str, float] = {}

    n, s0, s1, s2, s00, s11, s22, s01, s02, s12 = _accumulate_moments_from_mask_numba(
        mask.array
    )
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
    verts: np.ndarray, mesh_volume: float, surface_area: float
) -> tuple[dict[str, float], Optional[ConvexHull]]:
    """Calculate Convex Hull features."""
    features: dict[str, float] = {}
    if len(verts) <= 3:
        return features, None

    try:
        hull = ConvexHull(verts)
        vol_convex = hull.volume
        area_convex = hull.area

        if vol_convex > 0:
            features["volume_density_convex_hull_R3ER"] = mesh_volume / vol_convex
        if area_convex > 0:
            features["area_density_convex_hull_7T7F"] = surface_area / area_convex

        # Max 3D Diameter
        hull_points = verts[hull.vertices]
        if hull_points.shape[0] > 1:
            features["maximum_3d_diameter_L0JK"] = float(
                _max_pairwise_distance_numba(np.asarray(hull_points, dtype=np.float64))
            )

        return features, hull
    except Exception:
        return features, None


def _get_bounding_box_features(
    verts: np.ndarray,
    evecs: Optional[np.ndarray],
    mesh_volume: float,
    surface_area: float,
) -> dict[str, float]:
    """Calculate AABB and OMBB features."""
    features: dict[str, float] = {}
    if len(verts) == 0:
        return features

    # AABB
    min_bound = np.min(verts, axis=0)
    max_bound = np.max(verts, axis=0)
    dims = max_bound - min_bound
    vol_aabb = np.prod(dims)
    area_aabb = 2 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[2] * dims[0])

    if vol_aabb > 0:
        features["volume_density_aabb_PBX1"] = mesh_volume / vol_aabb
    if area_aabb > 0:
        features["area_density_aabb_R59B"] = surface_area / area_aabb

    # OMBB
    if evecs is not None:
        center = np.mean(verts, axis=0)

        # Deterministic streaming extents in Numba (avoids allocating rotated_verts and Python loop overhead)
        min_rot, max_rot = _ombb_extents_numba(
            np.asarray(verts, dtype=np.float64),
            np.asarray(center, dtype=np.float64),
            np.asarray(evecs, dtype=np.float64),
        )

        dims_rot = max_rot - min_rot
        vol_ombb = float(np.prod(dims_rot))
        area_ombb = float(
            2
            * (
                dims_rot[0] * dims_rot[1]
                + dims_rot[1] * dims_rot[2]
                + dims_rot[2] * dims_rot[0]
            )
        )

        if vol_ombb > 0:
            features["volume_density_ombb_ZH1A"] = mesh_volume / vol_ombb
        if area_ombb > 0:
            features["area_density_ombb_IQYR"] = surface_area / area_ombb

    return features


def _get_mvee_features(
    hull: Optional[ConvexHull],
    verts: np.ndarray,
    mesh_volume: float,
    surface_area: float,
) -> dict[str, float]:
    """Calculate MVEE features."""
    features: dict[str, float] = {}
    if hull is None:
        return features

    hull_points = verts[hull.vertices]
    hull_points_f64 = (
        hull_points
        if hull_points.dtype == np.float64
        else hull_points.astype(np.float64)
    )
    A_mvee, _ = _mvee_khachiyan_numba(hull_points_f64)

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
    mask: Image, image: Image, intensity_mask: Image, mesh_volume: float
) -> dict[str, float]:
    """Calculate intensity-weighted morphological features."""
    features: dict[str, float] = {}

    count_i, sum_w, sum_i0_w, sum_i1_w, sum_i2_w = (
        _accumulate_intensity_weighted_moments_numba(
            intensity_mask.array,
            image.array,
        )
    )
    if count_i > 0:
        mean_intensity = sum_w / float(count_i)
        features["integrated_intensity_99N0"] = mesh_volume * mean_intensity

        # Center of Mass Shift
        # CoM_geom (from morph mask)
        n_m, s0_m, s1_m, s2_m, _, _, _, _, _, _ = _accumulate_moments_from_mask_numba(
            mask.array
        )
        if n_m > 0:
            m0 = s0_m / float(n_m)
            m1 = s1_m / float(n_m)
            m2 = s2_m / float(n_m)
            sp_m = np.asarray(mask.spacing, dtype=np.float64)
            org_m = np.asarray(mask.origin, dtype=np.float64)
            com_geom0 = m0 * sp_m[0] + org_m[0]
            com_geom1 = m1 * sp_m[1] + org_m[1]
            com_geom2 = m2 * sp_m[2] + org_m[2]

            # CoM_gl (from intensity mask; intensity-weighted)
            if sum_w != 0.0:
                w0 = sum_i0_w / sum_w
                w1 = sum_i1_w / sum_w
                w2 = sum_i2_w / sum_w
                sp_i = np.asarray(intensity_mask.spacing, dtype=np.float64)
                org_i = np.asarray(intensity_mask.origin, dtype=np.float64)
                com_gl0 = w0 * sp_i[0] + org_i[0]
                com_gl1 = w1 * sp_i[1] + org_i[1]
                com_gl2 = w2 * sp_i[2] + org_i[2]

                dx0 = com_geom0 - com_gl0
                dx1 = com_geom1 - com_gl1
                dx2 = com_geom2 - com_gl2
                features["center_of_mass_shift_KLMA"] = float(
                    math.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
                )

    return features


def calculate_morphology_features(
    mask: Image, image: Optional[Image] = None, intensity_mask: Optional[Image] = None
) -> dict[str, float]:
    """
    Calculate morphological features from the ROI mask.
    Includes both voxel-based and mesh-based features (IBSI compliant).

    Args:
        mask: Image object containing the binary mask (Morphological Mask).
        image: Optional Image object containing intensity data (required for some features).
        intensity_mask: Optional Image object containing the intensity mask (e.g. after outlier filtering).
                        If provided, used for intensity-weighted features (99N0, KLMA).
                        If None, defaults to `mask`.

    Returns:
        Dictionary of calculated features.
    """
    features: dict[str, float] = {}
    i_mask = intensity_mask if intensity_mask is not None else mask

    # 1. Voxel Based Features
    voxel_volume = np.prod(mask.spacing)
    n_voxels = np.sum(mask.array)
    features["volume_voxel_counting_YEKZ"] = float(n_voxels * voxel_volume)

    # 2. Mesh Based Features
    mesh_feats, verts, faces = _get_mesh_features(mask)
    features.update(mesh_feats)

    if verts is None or faces is None:
        return features

    mesh_volume = features.get("volume_RNU0", 0.0)
    surface_area = features.get("surface_area_C0JK", 0.0)

    # 3. Shape Features
    features.update(_get_shape_features(surface_area, mesh_volume))

    # 4. PCA Based Features
    pca_feats, evals, evecs = _get_pca_features(mask, mesh_volume, surface_area)
    features.update(pca_feats)

    # 5. Convex Hull Features
    hull_feats, hull = _get_convex_hull_features(verts, mesh_volume, surface_area)
    features.update(hull_feats)

    # 6. Bounding Box Features
    features.update(_get_bounding_box_features(verts, evecs, mesh_volume, surface_area))

    # 7. MVEE Features
    features.update(_get_mvee_features(hull, verts, mesh_volume, surface_area))

    # 8. Intensity Based Features
    if image is not None:
        features.update(
            _get_intensity_morphology_features(mask, image, i_mask, mesh_volume)
        )

    return features
