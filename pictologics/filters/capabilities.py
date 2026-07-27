# pictologics/filters/capabilities.py
"""
Versioned, machine-readable capability metadata for the public IBSI 2 filters.

Compliance tooling (e.g. IBSI 2 conformance checkers) otherwise has to infer what
a filter supports by inspecting signatures, docstrings, and implementation
details. This module declares that information once, as data, grounded in the
actual implementations in ``mean.py``, ``log.py``, ``laws.py``, ``gabor.py``,
``wavelets.py``, and ``riesz.py``.

Keying scheme:
    :data:`FILTER_CAPABILITIES` is keyed primarily by the pipeline filter-type
    name used in ``RadiomicsPipeline`` filter steps (``"mean"``, ``"log"``,
    ``"laws"``, ``"gabor"``, ``"wavelet"``, ``"simoncelli"``, ``"riesz"``). The
    ``"riesz"`` key describes the plain ``riesz_transform`` (the pipeline's
    default ``variant="base"``). The pipeline also dispatches two other Riesz
    variants (``variant="log"`` -> ``riesz_log``, ``variant="simoncelli"`` ->
    ``riesz_simoncelli``), which have distinct capabilities (e.g. anisotropic
    spacing support) and are therefore exposed under their own
    ``"riesz_log"`` / ``"riesz_simoncelli"`` keys rather than folded into
    ``"riesz"``.

Example:
    Look up what a filter supports before invoking it:

    ```python
    from pictologics.filters import get_filter_capabilities

    capability = get_filter_capabilities("gabor")
    print(capability.kernel_dimensionality, capability.slice_plane_execution)
    # 2 True
    ```
"""

from dataclasses import dataclass
from typing import Optional

CAPABILITIES_SCHEMA_VERSION = "1.0.0"
"""Semantic version of the :data:`FILTER_CAPABILITIES` schema.

Bump the major component on breaking changes (field removal/retyping), the
minor component when adding fields or filter entries, and the patch component
for corrections that don't change the schema shape.
"""


@dataclass(frozen=True)
class FilterCapability:
    """
    Declarative capability record for one IBSI 2 filter (or Riesz variant).

    Every field is grounded in the corresponding filter implementation; see
    :data:`FILTER_CAPABILITIES` for the per-filter values and the source
    evidence behind each one.

    Attributes:
        input_dimensionality: Dimensionality/dimensionalities of the top-level
            image array the filter function accepts, e.g. ``(3,)`` for
            3D-only.
        kernel_dimensionality: Dimensionality of the convolution/transform
            kernel itself (2 or 3). May differ from `input_dimensionality`,
            e.g. Gabor uses a 2D kernel on a 3D input.
        slice_plane_execution: True if the filter applies its kernel
            plane-wise (slice-by-slice) rather than as a single volumetric
            operation.
        orthogonal_plane_averaging: True if the filter supports averaging its
            plane-wise response over the 3 orthogonal anatomical planes.
        rotation_pooling: Pooling method names accepted for pseudo-rotational
            invariance (e.g. ``("max", "average", "min")``), or ``()`` if the
            filter has no rotation-invariance/pooling mechanism.
        supported_boundaries: `BoundaryCondition` member names the filter
            actually accepts via its `boundary` parameter, or ``()`` if it has
            no such parameter.
        effective_boundary: What actually happens at the image border. One of
            ``"as_specified"`` (a spatial-convolution filter that applies the
            requested boundary directly), ``"as_specified_via_padding"``
            (FFT-domain filters: the requested boundary is realised by a
            defined pad-filter-crop procedure — the transform itself remains
            periodic on the padded domain, so the boundary is honoured
            approximately rather than exactly), or ``"periodic"`` (periodic
            regardless of what is requested).
        supported_riesz_orders: Description of the Riesz derivative orders
            accepted (Riesz-family filters only), or `None` if the filter has
            no Riesz order concept.
        structure_tensor_steering: True if the filter can steer its kernel
            using a structure tensor. Currently `False` for every filter (no
            `tensor_sigma`/steering exists anywhere in the package).
        anisotropic_spacing: How the filter handles anisotropic
            `spacing_mm`: ``"supported"`` (correctly converts per-axis),
            ``"warns_uses_first_axis"`` (warns and derives scale from the
            first axis only), or ``"not_applicable"`` (no `spacing_mm`
            parameter / concept).
    """

    input_dimensionality: tuple[int, ...]
    kernel_dimensionality: int
    slice_plane_execution: bool
    orthogonal_plane_averaging: bool
    rotation_pooling: tuple[str, ...]
    supported_boundaries: tuple[str, ...]
    effective_boundary: str
    supported_riesz_orders: Optional[str]
    structure_tensor_steering: bool
    anisotropic_spacing: str


# All four `BoundaryCondition` members, honored by `boundary=...` on mean/log/laws/
# gabor/wavelet today. Declared here once since it's shared verbatim by every entry
# below, including the FFT filters that forward-declare it (see module docstring
# and the "riesz*" comments further down).
_ALL_BOUNDARIES: tuple[str, ...] = ("ZERO", "NEAREST", "PERIODIC", "MIRROR")

FILTER_CAPABILITIES: dict[str, FilterCapability] = {
    # mean.py: mean_filter(image, support, boundary, source_mask). Isotropic M^3
    # box support in voxel units only (no spacing_mm/mm concept at all), applied
    # as a single 3D uniform_filter. No pooling/rotation-invariance, no plane
    # averaging, no Riesz concept. `boundary` is forwarded straight to
    # scipy.ndimage.uniform_filter's `mode`, which honors it exactly.
    "mean": FilterCapability(
        input_dimensionality=(3,),
        kernel_dimensionality=3,
        slice_plane_execution=False,
        orthogonal_plane_averaging=False,
        rotation_pooling=(),
        supported_boundaries=_ALL_BOUNDARIES,
        effective_boundary="as_specified",
        supported_riesz_orders=None,
        structure_tensor_steering=False,
        anisotropic_spacing="not_applicable",
    ),
    # log.py: laplacian_of_gaussian(image, sigma_mm, spacing_mm, truncate, boundary,
    # source_mask). Spherically symmetric 3D gaussian_laplace; spacing_mm may be a
    # per-axis (x, y, z) tuple and is converted to a per-axis sigma_voxels tuple
    # independently (sigma_voxels = tuple(sigma_mm / s for s in spacing_mm)), so
    # anisotropic spacing is genuinely supported. No pooling/rotation-invariance,
    # no plane averaging, no Riesz concept.
    "log": FilterCapability(
        input_dimensionality=(3,),
        kernel_dimensionality=3,
        slice_plane_execution=False,
        orthogonal_plane_averaging=False,
        rotation_pooling=(),
        supported_boundaries=_ALL_BOUNDARIES,
        effective_boundary="as_specified",
        supported_riesz_orders=None,
        structure_tensor_steering=False,
        anisotropic_spacing="supported",
    ),
    # laws.py: laws_filter(image, kernels, boundary, rotation_invariant, pooling,
    # compute_energy, energy_distance, use_parallel, source_mask). `kernels` must
    # parse into exactly 3 codes combined via separable 1D convolution into a 3D
    # kernel; voxel units only (no spacing_mm). `pooling` is validated against
    # ("max", "average", "min") when rotation_invariant=True.
    "laws": FilterCapability(
        input_dimensionality=(3,),
        kernel_dimensionality=3,
        slice_plane_execution=False,
        orthogonal_plane_averaging=False,
        rotation_pooling=("max", "average", "min"),
        supported_boundaries=_ALL_BOUNDARIES,
        effective_boundary="as_specified",
        supported_riesz_orders=None,
        structure_tensor_steering=False,
        anisotropic_spacing="not_applicable",
    ),
    # gabor.py: gabor_filter(image, sigma_mm, lambda_mm, gamma, theta, spacing_mm,
    # boundary, rotation_invariant, delta_theta, pooling, average_over_planes,
    # use_parallel, source_mask). The 2D kernel is applied plane-wise (one plane by
    # default, or all 3 orthogonal planes when average_over_planes=True) via
    # per-slice FFT convolution with an explicit pad (by `boundary`'s mode) then
    # crop, so the requested boundary is genuinely realized (gabor.py ~L231-263).
    # `pooling` is validated against ("max", "average", "min") unconditionally
    # (gabor.py ~L130-132). Anisotropic spacing: each plane derives its own two
    # in-plane axes and their true spacings (gabor.py ~L212-234); when they match,
    # the kernel is built on the original voxel-unit grid, and when they differ,
    # it is built on a physical (mm) grid with a per-axis radius, so anisotropic
    # in-plane spacing (including anisotropic z, relevant only when
    # average_over_planes=True) is genuinely supported rather than warned about.
    "gabor": FilterCapability(
        input_dimensionality=(3,),
        kernel_dimensionality=2,
        slice_plane_execution=True,
        orthogonal_plane_averaging=True,
        rotation_pooling=("max", "average", "min"),
        supported_boundaries=_ALL_BOUNDARIES,
        effective_boundary="as_specified",
        supported_riesz_orders=None,
        structure_tensor_steering=False,
        anisotropic_spacing="supported",
    ),
    # wavelets.py: wavelet_transform(image, wavelet, level, decomposition, boundary,
    # rotation_invariant, pooling, use_parallel, source_mask). Undecimated (a
    # trous) separable 3D wavelet: 3 axis-wise 1D convolutions per level via
    # scipy.ndimage.convolve1d, which honors `mode` (boundary) exactly on every
    # call. No spacing_mm (scale is voxel-based via `level`, not mm). `pooling` is
    # validated against ("max", "average", "min") when rotation_invariant=True.
    "wavelet": FilterCapability(
        input_dimensionality=(3,),
        kernel_dimensionality=3,
        slice_plane_execution=False,
        orthogonal_plane_averaging=False,
        rotation_pooling=("max", "average", "min"),
        supported_boundaries=_ALL_BOUNDARIES,
        effective_boundary="as_specified",
        supported_riesz_orders=None,
        structure_tensor_steering=False,
        anisotropic_spacing="not_applicable",
    ),
    # wavelets.py: simoncelli_wavelet(image, level, boundary, source_mask). Isotropic
    # 3D band-pass transfer function applied via full FFT (scipy.fft.fftn/ifftn); no
    # spacing_mm. The `boundary` parameter already exists on the signature
    # (default BoundaryCondition.PERIODIC). PERIODIC runs the FFT directly; any other
    # boundary is realised by the shared pad-filter-crop helper in base.py, so the
    # requested boundary is honoured approximately (the transform is still periodic on
    # the padded domain) — hence effective_boundary="as_specified_via_padding".
    "simoncelli": FilterCapability(
        input_dimensionality=(3,),
        kernel_dimensionality=3,
        slice_plane_execution=False,
        orthogonal_plane_averaging=False,
        rotation_pooling=(),
        supported_boundaries=_ALL_BOUNDARIES,
        effective_boundary="as_specified_via_padding",
        supported_riesz_orders=None,
        structure_tensor_steering=False,
        anisotropic_spacing="not_applicable",
    ),
    # riesz.py: riesz_transform(image, order, source_mask) -- the pipeline's
    # variant="base" dispatch for filter_type="riesz". All-pass derivative filter
    # applied via real FFT (scipy.fft.rfftn/irfftn); `order` is a per-axis tuple of
    # non-negative ints with sum L >= 1 (any such tuple is accepted -- see
    # get_riesz_orders() for an enumeration helper). No spacing_mm/rotation/pooling.
    # `boundary` (default PERIODIC) is honoured via the same pad-filter-crop helper as
    # "simoncelli". Note the Riesz kernel has power-law tails rather than compact
    # support, so no finite pad fully removes truncation error — another reason the
    # effective behaviour is "as_specified_via_padding" rather than "as_specified".
    "riesz": FilterCapability(
        input_dimensionality=(3,),
        kernel_dimensionality=3,
        slice_plane_execution=False,
        orthogonal_plane_averaging=False,
        rotation_pooling=(),
        supported_boundaries=_ALL_BOUNDARIES,
        effective_boundary="as_specified_via_padding",
        supported_riesz_orders=(
            "Tuple[int, ...] (l1, ..., ld): any non-negative ints summing to L >= 1; "
            "see get_riesz_orders(max_order, ndim) to enumerate all combinations."
        ),
        structure_tensor_steering=False,
        anisotropic_spacing="not_applicable",
    ),
    # riesz.py: riesz_log(image, sigma_mm, spacing_mm, order, truncate, source_mask)
    # -- variant="log": applies laplacian_of_gaussian() then riesz_transform() to
    # its output. spacing_mm is forwarded verbatim to laplacian_of_gaussian, which
    # converts it per-axis (see "log" above), so anisotropic spacing is genuinely
    # supported here despite the Riesz stage itself having no spacing concept. No
    # `boundary` parameter (same forward-declaration caveat as "riesz" above).
    "riesz_log": FilterCapability(
        input_dimensionality=(3,),
        kernel_dimensionality=3,
        slice_plane_execution=False,
        orthogonal_plane_averaging=False,
        rotation_pooling=(),
        supported_boundaries=_ALL_BOUNDARIES,
        effective_boundary="as_specified_via_padding",
        supported_riesz_orders=(
            "Tuple[int, ...] (l1, ..., ld): any non-negative ints summing to L >= 1; "
            "see get_riesz_orders(max_order, ndim) to enumerate all combinations."
        ),
        structure_tensor_steering=False,
        anisotropic_spacing="supported",
    ),
    # riesz.py: riesz_simoncelli(image, level, order, source_mask) -- variant=
    # "simoncelli": applies simoncelli_wavelet() then riesz_transform(). No
    # spacing_mm anywhere in this call chain (Simoncelli assumes an isotropic
    # frequency grid regardless). No `boundary` parameter (same
    # forward-declaration caveat as "riesz" above).
    "riesz_simoncelli": FilterCapability(
        input_dimensionality=(3,),
        kernel_dimensionality=3,
        slice_plane_execution=False,
        orthogonal_plane_averaging=False,
        rotation_pooling=(),
        supported_boundaries=_ALL_BOUNDARIES,
        effective_boundary="as_specified_via_padding",
        supported_riesz_orders=(
            "Tuple[int, ...] (l1, ..., ld): any non-negative ints summing to L >= 1; "
            "see get_riesz_orders(max_order, ndim) to enumerate all combinations."
        ),
        structure_tensor_steering=False,
        anisotropic_spacing="not_applicable",
    ),
}
"""Capability record for every public filter, keyed by pipeline filter-type name.

See the module docstring for the ``"riesz"`` / ``"riesz_log"`` /
``"riesz_simoncelli"`` keying scheme.
"""


def get_filter_capabilities(name: str) -> FilterCapability:
    """
    Look up the capability record for a filter by its pipeline filter-type name.

    Args:
        name: Pipeline filter-type name, e.g. ``"mean"``, ``"log"``, ``"laws"``,
            ``"gabor"``, ``"wavelet"``, ``"simoncelli"``, ``"riesz"``,
            ``"riesz_log"``, or ``"riesz_simoncelli"``.

    Returns:
        The `FilterCapability` record registered under `name`.

    Raises:
        ValueError: If `name` is not a key in `FILTER_CAPABILITIES`.

    Example:
        ```python
        from pictologics.filters import get_filter_capabilities

        capability = get_filter_capabilities("riesz_log")
        print(capability.anisotropic_spacing)
        # supported
        ```
    """
    try:
        return FILTER_CAPABILITIES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(FILTER_CAPABILITIES))
        raise ValueError(f"Unknown filter name {name!r}; valid names are: {valid}") from exc
