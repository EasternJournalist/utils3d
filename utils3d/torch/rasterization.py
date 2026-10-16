import math
from contextlib import contextmanager
from numbers import Integral, Real
from typing import ContextManager, Dict, Iterator, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from .transforms import extrinsics_to_view, intrinsics_to_perspective


__all__ = [
    'rasterize_triangles',
    'rasterize_triangles_peeling',
    'rasterize_lines',
    'rasterize_point_cloud',
]


def rasterize_triangles(
    size: Tuple[int, int],
    *,
    vertices: Tensor,
    attributes: Optional[Tensor] = None,
    attributes_domain: Literal['vertex', 'face'] = 'vertex',
    faces: Optional[Tensor] = None,
    view: Optional[Tensor] = None,
    projection: Optional[Tensor] = None,
    extrinsics: Optional[Tensor] = None,
    intrinsics: Optional[Tensor] = None,
    near: float = 0.01,
    far: float = float('inf'),
    cull_backface: bool = False,
    return_depth: bool = False,
    return_interpolation: bool = False,
) -> Dict[str, Tensor]:
    """Rasterize triangles with Triton and interpolate attributes with PyTorch.

    Parameters
    ----
    - `size`: (height, width), positive integers.
    - `vertices`: float32 GPU tensor of shape `(..., N, D)` with indexed faces,
        or `(..., T, 3, D)` when `faces` is None. D is 2, 3, or 4; missing z/w
        components default to 0/1. Coordinates must be finite.
    - `attributes`: optional floating-point tensor of shape `(..., N, C)`,
        `(..., T, 3, C)` for unindexed vertex attributes, or `(..., T, C)` for
        face attributes. Any channel count is supported; dtype is preserved.
    - `attributes_domain`: 'vertex' for perspective-correct interpolation,
        or 'face' for constant per-triangle attributes.
    - `faces`: optional int32/int64 tensor of shape `(T, 3)`, shared across
        batches. Indices must be in `[0, N)`. All tensors must be on one GPU.
    - `view` | `extrinsics`: optional `(..., 4, 4)` float32 OpenGL view or
        OpenCV extrinsics matrix. Mutually exclusive; defaults to identity.
    - `projection` | `intrinsics`: optional `(..., 4, 4)` float32 OpenGL
        projection or `(..., 3, 3)` normalized OpenCV intrinsics matrix.
        Mutually exclusive; defaults to identity.
    - `near`, `far`: clipping distances used only with intrinsics; 0 < near < far.
        Infinite far is supported.
    - `cull_backface`: discard clockwise triangles in NDC. Defaults to False.
    - `return_depth`: include camera-space depth (-view z / view w).
    - `return_interpolation`: include original triangle IDs and barycentrics.

    Returns
    ----
    A dictionary, with broadcast batch dimensions `...`:
    - `mask`: `(..., H, W)` bool coverage mask.
    - `image`: `(..., H, W, C)`, present only when attributes are supplied.
    - `depth`: `(..., H, W)` float32, present when requested; background is inf.
    - `interpolation_id`: `(..., H, W)` int32 original face ID, present when
        requested; background is -1.
    - `interpolation_uv`: `(..., H, W, 2)` float32 first two perspective-correct
        barycentric weights, present when requested; background is zero.
    Images use a top-left origin. Background attributes are zero.

    Notes
    ----
    Only attributes are differentiable. Geometry, cameras, depth, and barycentrics
    are detached. Rendering is point-sampled, with six-plane homogeneous clipping,
    eight-bit subpixel coverage and a top-left shared-edge rule. NDC depth uses
    a strict less-than test against a far-plane clear value of 1; exact depth ties
    choose the lowest face ID. No antialiasing, derivatives, textures, or context
    are required. Faces and vertices must each fit in signed 32-bit indices;
    an image must have fewer than 2**31 pixels.

    Triton is imported on first use. No custom CUDA/C++ extension is built.
    Input validation synchronizes once to check finite transformed coordinates
    and index bounds. Work is proportional to triangle bounding-box areas;
    heavily overlapping or long, thin triangles are not optimized.
    """
    if attributes_domain not in ('vertex', 'face'):
        raise ValueError("attributes_domain must be 'vertex' or 'face'.")
    size, batch_shape, positions, view_positions, faces, _, attributes = _prepare_inputs(
        size, vertices, attributes, faces, 3, attributes_domain == 'vertex',
        view, projection, extrinsics, intrinsics, near, far,
    )
    assert faces is not None
    _validate_geometry(positions, faces, positions.shape[1])
    faces = faces.to(torch.int32).contiguous()
    with torch.no_grad():
        ids, uv, depth = _load_rasterization_kernels().rasterize_triangles_triton(
            positions, view_positions, faces, size, cull_backface,
            return_interpolation or (attributes is not None and attributes_domain == 'vertex'),
            return_depth,
        )
    return _triangle_output(ids, uv, depth, faces, attributes, attributes_domain, batch_shape, return_depth, return_interpolation)


def rasterize_triangles_peeling(
    size: Tuple[int, int],
    *,
    vertices: Tensor,
    attributes: Optional[Tensor] = None,
    attributes_domain: Literal['vertex', 'face'] = 'vertex',
    faces: Optional[Tensor] = None,
    view: Optional[Tensor] = None,
    projection: Optional[Tensor] = None,
    extrinsics: Optional[Tensor] = None,
    intrinsics: Optional[Tensor] = None,
    near: float = 0.01,
    far: float = float('inf'),
    cull_backface: bool = False,
    return_depth: bool = False,
    return_interpolation: bool = False,
) -> ContextManager[Iterator[Dict[str, Tensor]]]:
    """Rasterize successive visible triangle layers with Triton depth peeling.

    Parameters
    ----
    Parameters, broadcasting, camera conventions and attribute domains match
    `rasterize_triangles`. All tensors must share a GPU. Only attributes receive
    gradients; geometry and camera parameters are detached.

    Returns
    ----
    A context manager yielding an iterator of dictionaries with the same keys,
    layouts and background values as `rasterize_triangles`. The first layer is
    identical to ordinary rendering. Subsequent layers select the closest fragment
    with strictly greater float32 NDC depth at each pixel. Equal-depth fragments
    are skipped together; the smallest face ID wins within a layer.

    A final empty layer is yielded, including for empty meshes. Batched images
    continue together until all are empty. Pixels already exhausted stay empty.
    Layers own their output tensors and can be retained for attribute backprop.
    Accumulate layer losses before backward when sharing an input autograd graph.

    Notes
    ----
    Geometry preparation and validation run once. Each layer rerasterizes the
    mesh and reads a single coverage flag on the CPU for termination. There is no
    layer cap; break the iterator to limit work. Exit closes the iterator even on
    early termination. Do not mutate inputs during iteration. Layer ordering uses
    the same NDC depth test as ordinary rendering, not a camera-distance sort.

    Example
    ----
    ```
    with utils3d.pt.rasterize_triangles_peeling(
        (512, 512), vertices=vertices, faces=faces, attributes=colors,
        return_depth=True,
    ) as layers:
        for i, layer in enumerate(layers):
            image = layer['image']
            if i == 3:
                break
    ```
    """
    if attributes_domain not in ('vertex', 'face'):
        raise ValueError("attributes_domain must be 'vertex' or 'face'.")
    size, batch_shape, positions, view_positions, faces, _, attributes = _prepare_inputs(
        size, vertices, attributes, faces, 3, attributes_domain == 'vertex',
        view, projection, extrinsics, intrinsics, near, far,
    )
    assert faces is not None
    _validate_geometry(positions, faces, positions.shape[1])
    faces = faces.to(torch.int32).contiguous()
    kernels = _load_rasterization_kernels()
    need_uv = return_interpolation or (attributes is not None and attributes_domain == 'vertex')

    def iterate() -> Iterator[Dict[str, Tensor]]:
        previous = None
        while True:
            with torch.no_grad():
                current = torch.empty((positions.shape[0], *size), dtype=torch.int64, device=positions.device)
                ids, uv, depth = kernels.rasterize_triangles_triton(
                    positions, view_positions, faces, size, cull_backface, need_uv, return_depth,
                    previous_keys=previous, depth_keys=current,
                )
                nonempty = bool((ids >= 0).any())
            yield _triangle_output(
                ids, uv, depth, faces, attributes, attributes_domain, batch_shape, return_depth, return_interpolation,
            )
            if not nonempty:
                break
            previous = current

    @contextmanager
    def managed() -> Iterator[Iterator[Dict[str, Tensor]]]:
        layers = iterate()
        try:
            yield layers
        finally:
            layers.close()

    return managed()


def rasterize_lines(
    size: Tuple[int, int],
    *,
    vertices: Tensor,
    attributes: Optional[Tensor] = None,
    attributes_domain: Literal['vertex', 'line'] = 'vertex',
    lines: Optional[Tensor] = None,
    view: Optional[Tensor] = None,
    projection: Optional[Tensor] = None,
    extrinsics: Optional[Tensor] = None,
    intrinsics: Optional[Tensor] = None,
    near: float = 0.01,
    far: float = float('inf'),
    line_width: float = 1.0,
    return_depth: bool = False,
    return_interpolation: bool = False,
) -> Dict[str, Tensor]:
    """Rasterize constant-pixel-width line segments with Triton.

    Parameters
    ----
    - `size`: (height, width), positive integers.
    - `vertices`: float32 GPU tensor `(..., N, D)`, or `(..., L, 2, D)` when
        lines is None. D is 2, 3, or 4; missing z/w components default to 0/1.
    - `attributes`: floating-point `(..., N, C)`, `(..., L, 2, C)` for unindexed
        vertex attributes, or `(..., L, C)` for constant line attributes.
    - `attributes_domain`: 'vertex' for perspective-correct endpoint interpolation,
        or 'line' for one attribute vector per segment.
    - `lines`: optional `(L, 2)` int32/int64 indices, shared across batches.
    - `view` | `extrinsics`: optional float32 `(..., 4, 4)` OpenGL view or OpenCV
        extrinsics matrix. Mutually exclusive; defaults to identity.
    - `projection` | `intrinsics`: optional float32 `(..., 4, 4)` OpenGL projection
        or `(..., 3, 3)` normalized OpenCV intrinsics. Mutually exclusive.
    - `near`, `far`: clipping distances used only with intrinsics; 0 < near < far.
    - `line_width`: finite, positive width in pixels, perpendicular to the segment.
    - `return_depth`: include camera-space depth (-view z / view w).
    - `return_interpolation`: include original line IDs and endpoint weights.

    Returns
    ----
    A dictionary with broadcast batch dimensions `...` and a top-left origin:
    - `mask`: `(..., H, W)` bool.
    - `image`: `(..., H, W, C)`, when attributes are supplied; background is zero.
    - `depth`: `(..., H, W)` float32, when requested; background is inf.
    - `interpolation_id`: `(..., H, W)` int32 line ID, when requested; background -1.
    - `interpolation_uv`: `(..., H, W, 2)` perspective-correct weights of the
        original first and second endpoints, when requested; background is zero.

    Notes
    ----
    Centerlines are clipped against all six homogeneous frustum planes before
    expansion to rectangles with flat (butt) caps. Projected zero-length segments
    produce no fragments. Width is not rounded to an integer. This is geometric
    rectangle coverage, not OpenGL's implementation-dependent aliased line rule.
    The triangle rasterizer supplies subpixel coverage and deterministic depth
    testing. All inputs must share a GPU; batches broadcast. Only attributes
    receive gradients. No context, antialiasing, or background compositing is used.
    """
    if attributes_domain not in ('vertex', 'line'):
        raise ValueError("attributes_domain must be 'vertex' or 'line'.")
    if isinstance(line_width, bool) or not isinstance(line_width, Real):
        raise TypeError("line_width must be a real scalar.")
    if not math.isfinite(line_width) or not 0 < line_width <= torch.finfo(torch.float32).max:
        raise ValueError("line_width must be finite, positive, and representable in float32.")
    size, batch_shape, positions, view_positions, lines, _, attributes = _prepare_inputs(
        size, vertices, attributes, lines, 2, attributes_domain == 'vertex',
        view, projection, extrinsics, intrinsics, near, far,
    )
    assert lines is not None
    _validate_geometry(positions, lines, positions.shape[1])
    lines = lines.to(torch.int32).contiguous()
    need_uv = return_interpolation or (attributes is not None and attributes_domain == 'vertex')
    kernels = _load_rasterization_kernels()
    with torch.no_grad():
        expanded, view_expanded, faces, intervals = kernels.expand_lines_triton(
            positions, view_positions, lines, size, float(line_width), need_uv, return_depth,
        )
        triangle_ids, uv, depth = kernels.rasterize_triangles_triton(
            expanded, view_expanded, faces, size, False, need_uv, return_depth,
        )
        ids = torch.div(triangle_ids, 2, rounding_mode='floor')
        mask = ids >= 0
        if need_uv:
            assert uv is not None and intervals is not None
            batch_index = torch.arange(positions.shape[0], device=positions.device)[:, None, None]
            line_index = torch.where(mask, ids, lines.shape[0]).long()
            bounds = F.pad(intervals, (0, 0, 0, 1))[batch_index, line_index]
            end_weight = torch.where(triangle_ids % 2 == 0, 1.0 - uv.sum(-1), 1.0 - uv[..., 0])
            t = bounds[..., 0] + (bounds[..., 1] - bounds[..., 0]) * end_weight
            weights = torch.stack((1.0 - t, t), dim=-1).masked_fill(~mask[..., None], 0)
        else:
            weights = None
    output = {'mask': mask}
    if attributes is not None:
        if attributes_domain == 'vertex':
            assert weights is not None
            output['image'] = _interpolate_attributes(attributes, ids, lines, weights)
        else:
            output['image'] = _interpolate_attributes(attributes, ids)
    if return_depth:
        assert depth is not None
        output['depth'] = depth
    if return_interpolation:
        assert weights is not None
        output['interpolation_id'] = ids
        output['interpolation_uv'] = weights
    return _restore_batch(output, batch_shape)


def rasterize_point_cloud(
    size: Tuple[int, int],
    *,
    points: Tensor,
    point_sizes: Union[float, Tensor] = 10,
    point_size_in: Literal['2d', '3d'] = '2d',
    point_shape: Literal['triangle', 'square', 'pentagon', 'hexagon', 'circle'] = 'square',
    attributes: Optional[Tensor] = None,
    view: Optional[Tensor] = None,
    projection: Optional[Tensor] = None,
    extrinsics: Optional[Tensor] = None,
    intrinsics: Optional[Tensor] = None,
    near: float = 0.01,
    far: float = float('inf'),
    return_depth: bool = False,
    return_point_id: bool = False,
) -> Dict[str, Tensor]:
    """Rasterize point-cloud billboards with Triton.

    Parameters
    ----
    - `size`: (height, width), positive integers.
    - `points`: float32 GPU tensor `(..., N, D)`. D is 2, 3, or 4; missing z/w
        components default to 0/1.
    - `point_sizes`: finite, nonnegative scalar or floating-point GPU tensor
        `(..., N)` / `(..., 1)`. Zero-sized points are invisible.
    - `point_size_in`: '2d' for pixel sizes, or '3d' for sizes in the camera's
        XY plane (NumPy's world-unit billboard convention).
    - `point_shape`: 'triangle', 'square', 'pentagon', 'hexagon', or 'circle'.
        Shapes match NumPy: squares have side length point_sizes, circles have
        that diameter, and other polygons have that circumcircle diameter.
    - `attributes`: optional floating-point `(..., N, C)` point attributes.
    - `view` | `extrinsics`: optional float32 `(..., 4, 4)` OpenGL view or OpenCV
        extrinsics matrix. Mutually exclusive; defaults to identity.
    - `projection` | `intrinsics`: optional float32 `(..., 4, 4)` OpenGL projection
        or `(..., 3, 3)` normalized OpenCV intrinsics. Mutually exclusive.
    - `near`, `far`: clipping distances used only with intrinsics; 0 < near < far.
    - `return_depth`: include the point center's camera-space depth (-z/w).
    - `return_point_id`: include original point IDs.

    Returns
    ----
    A dictionary with broadcast batch dimensions `...` and a top-left origin:
    - `mask`: `(..., H, W)` bool.
    - `image`: `(..., H, W, C)`, when attributes are supplied; background is zero.
    - `depth`: `(..., H, W)` float32, when requested; background is inf.
    - `point_id`: `(..., H, W)` int32, when requested; background is -1.

    Notes
    ----
    Billboards, not just centers, are clipped to the frustum. A center outside
    the image can still contribute pixels. Circles reject outside fragments
    before depth testing, so their bounding squares do not occlude other points.
    2D billboards keep the center's NDC depth; 3D billboards are camera-facing
    planes and can project to ellipses or skewed polygons. Only attributes receive
    gradients, not positions, sizes, or cameras. All tensors must share a GPU.
    No context, antialiasing, or background compositing is used.
    """
    if point_shape not in ('triangle', 'square', 'pentagon', 'hexagon', 'circle'):
        raise ValueError("Unknown point_shape.")
    if point_size_in not in ('2d', '3d'):
        raise ValueError("point_size_in must be '2d' or '3d'.")
    if not isinstance(points, Tensor) or points.ndim < 2:
        raise ValueError("points must be a tensor of shape (..., N, D).")
    size_batch_shape = ()
    if isinstance(point_sizes, Tensor):
        if not point_sizes.is_floating_point() or point_sizes.device != points.device:
            raise ValueError("point_sizes must be floating-point and on the same GPU as points.")
        if point_sizes.ndim:
            if point_sizes.shape[-1] not in (1, points.shape[-2]):
                raise ValueError("point_sizes must end in N or 1.")
            size_batch_shape = point_sizes.shape[:-1]
    else:
        if isinstance(point_sizes, bool) or not isinstance(point_sizes, Real):
            raise TypeError("point_sizes must be a real scalar or floating-point tensor.")
        if not math.isfinite(point_sizes) or not 0 <= point_sizes <= torch.finfo(torch.float32).max:
            raise ValueError("point_sizes must be finite, nonnegative, and representable in float32.")
    size, batch_shape, positions, view_positions, _, projection, attributes = _prepare_inputs(
        size, points, attributes, None, 1, False,
        view, projection, extrinsics, intrinsics, near, far, size_batch_shape,
    )
    kernels = _load_rasterization_kernels()
    batch, num_points = positions.shape[:2]
    depth = None
    with torch.no_grad():
        raw_sizes = torch.as_tensor(point_sizes, dtype=torch.float32, device=positions.device).detach()
        sizes = raw_sizes.expand(*batch_shape, num_points).reshape(batch, num_points)
        expanded, faces = kernels.expand_points_triton(
            positions, view_positions, projection, sizes, size, point_size_in == '3d', point_shape,
        )
        _validate_geometry(expanded, None, 0, raw_sizes)
        triangle_ids, _, _ = kernels.rasterize_triangles_triton(
            expanded, expanded, faces, size, False, False, False, circle=point_shape == 'circle',
        )
        triangles_per_point = {'triangle': 1, 'square': 2, 'pentagon': 3, 'hexagon': 4, 'circle': 2}[point_shape]
        ids = torch.div(triangle_ids, triangles_per_point, rounding_mode='floor')
        mask = ids >= 0
        if return_depth:
            center_depth = -view_positions[..., 2] / view_positions[..., 3]
            center_depth = F.pad(center_depth, (0, 1), value=float('inf'))
            index = torch.where(mask, ids, num_points).long()
            batch_index = torch.arange(batch, device=positions.device)[:, None, None]
            depth = center_depth[batch_index, index]
    output = {'mask': mask}
    if attributes is not None:
        output['image'] = _interpolate_attributes(attributes, ids)
    if return_depth:
        assert depth is not None
        output['depth'] = depth
    if return_point_id:
        output['point_id'] = ids
    return _restore_batch(output, batch_shape)


def _prepare_inputs(
    size: Tuple[int, int], vertices: Tensor, attributes: Optional[Tensor],
    indices: Optional[Tensor], corners: int, vertex_attributes: bool,
    view: Optional[Tensor], projection: Optional[Tensor], extrinsics: Optional[Tensor],
    intrinsics: Optional[Tensor], near: float, far: float,
    extra_batch_shape: Tuple[int, ...] = (),
) -> Tuple[Tuple[int, int], Tuple[int, ...], Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("size must be a (height, width) pair.")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in size):
        raise ValueError("size must contain two positive integers.")
    height, width = int(size[0]), int(size[1])
    if height * width >= 2**31:
        raise ValueError("The output image must have fewer than 2**31 pixels.")
    if not isinstance(vertices, Tensor) or vertices.dtype != torch.float32:
        raise TypeError("vertices/points must be a float32 tensor.")
    if not vertices.is_cuda:
        raise ValueError("Triton rasterization requires GPU tensors.")
    if vertices.ndim < 2 or vertices.shape[-1] not in (2, 3, 4):
        raise ValueError("vertices/points must end in (N, D), with D equal to 2, 3, or 4.")
    if view is not None and extrinsics is not None:
        raise ValueError("Cannot specify both view and extrinsics.")
    if projection is not None and intrinsics is not None:
        raise ValueError("Cannot specify both projection and intrinsics.")
    device = vertices.device
    unindexed = indices is None and corners > 1
    if unindexed:
        if vertices.ndim < 3 or vertices.shape[-2] != corners:
            raise ValueError(f"Unindexed vertices must have shape (..., T, {corners}, D).")
        num_primitives = vertices.shape[-3]
        num_vertices = num_primitives * corners
        vertex_shape = (num_primitives, corners)
        batch_shapes = [vertices.shape[:-3], extra_batch_shape]
    else:
        num_vertices = vertices.shape[-2]
        vertex_shape = (num_vertices,)
        batch_shapes = [vertices.shape[:-2], extra_batch_shape]
        if indices is not None:
            if indices.ndim != 2 or indices.shape[-1] != corners:
                raise ValueError(f"Indices must have shape (T, {corners}), shared across batches.")
            if indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("Indices must have an int32 or int64 dtype.")
            if indices.device != device:
                raise ValueError("Indices and vertices must be on the same GPU.")
            num_primitives = indices.shape[0]
        else:
            num_primitives = num_vertices
    if max(num_vertices, num_primitives) >= 2**31:
        raise ValueError("Vertices and primitives must each fit in signed 32-bit indices.")
    attribute_shape = vertex_shape if vertex_attributes else (num_primitives,)
    attribute_ndim = len(attribute_shape) + 1
    if attributes is not None:
        if not attributes.is_floating_point():
            raise TypeError("attributes must have a floating-point dtype.")
        if attributes.device != device:
            raise ValueError("attributes and vertices must be on the same GPU.")
        if attributes.ndim < attribute_ndim or attributes.shape[-attribute_ndim:-1] != attribute_shape:
            raise ValueError(f"attributes must end in {attribute_shape} followed by a channel dimension.")
        batch_shapes.append(attributes.shape[:-attribute_ndim])
    for name, matrix, dimension in (
        ('view', view, 4), ('extrinsics', extrinsics, 4),
        ('projection', projection, 4), ('intrinsics', intrinsics, 3),
    ):
        if matrix is not None:
            if matrix.ndim < 2 or matrix.shape[-2:] != (dimension, dimension):
                raise ValueError(f"{name} must have shape (..., {dimension}, {dimension}).")
            if matrix.dtype != torch.float32 or matrix.device != device:
                raise ValueError(f"{name} must be float32 and on the same GPU as vertices.")
            batch_shapes.append(matrix.shape[:-2])
    if intrinsics is not None:
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in (near, far)):
            raise TypeError("near and far must be real scalars.")
        if not math.isfinite(near) or not 0 < near < far:
            raise ValueError("Clipping distances must satisfy 0 < near < far.")

    # The batching decorator does not handle dictionaries or unindexed primitives.
    batch_shape = torch.broadcast_shapes(*batch_shapes)
    batch = math.prod(batch_shape)
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
        vertices = vertices.detach()
        if unindexed:
            vertices = vertices.reshape((*vertices.shape[:-3], num_vertices, vertices.shape[-1]))
            indices = torch.arange(num_vertices, dtype=torch.int32, device=device).reshape(num_primitives, corners)
        if vertices.shape[-1] == 2:
            vertices = F.pad(vertices, (0, 1))
        if vertices.shape[-1] == 3:
            vertices = F.pad(vertices, (0, 1), value=1.0)
        if extrinsics is not None:
            view = extrinsics_to_view(extrinsics)
        if intrinsics is not None:
            projection = intrinsics_to_perspective(intrinsics, near, far)
        view_positions = vertices if view is None else vertices @ view.mT
        positions = view_positions if projection is None else view_positions @ projection.mT
        positions = positions.expand(*batch_shape, num_vertices, 4).reshape(batch, num_vertices, 4).contiguous()
        view_positions = view_positions.expand(*batch_shape, num_vertices, 4).reshape(batch, num_vertices, 4).contiguous()
        if projection is not None:
            projection = projection.expand(*batch_shape, 4, 4).reshape(batch, 4, 4).contiguous()
    if attributes is not None:
        count = num_vertices if vertex_attributes else num_primitives
        channels = attributes.shape[-1]
        attributes = attributes.reshape((*attributes.shape[:-attribute_ndim], count, channels))
        attributes = attributes.expand(*batch_shape, count, channels).reshape(batch, count, channels)
    return (height, width), batch_shape, positions, view_positions, indices, projection, attributes


def _validate_geometry(
    positions: Tensor, indices: Optional[Tensor], num_vertices: int, point_sizes: Optional[Tensor] = None,
) -> None:
    checks = [torch.isfinite(positions).all()]
    errors = ["Transformed vertices must contain only finite float32 coordinates."]
    if indices is not None and indices.numel():
        low, high = torch.aminmax(indices)
        checks.append((low >= 0) & (high < num_vertices))
        errors.append("Primitive indices contain an index outside [0, N).")
    if point_sizes is not None:
        checks.append(torch.isfinite(point_sizes).all() & (point_sizes >= 0).all())
        errors.append("point_sizes must contain only finite, nonnegative float32 values.")
    for valid, error in zip(torch.stack(checks).tolist(), errors):
        if not valid:
            raise ValueError(error)


def _interpolate_attributes(
    attributes: Tensor, ids: Tensor, indices: Optional[Tensor] = None, weights: Optional[Tensor] = None,
) -> Tensor:
    batch, count = attributes.shape[:2]
    attributes = F.pad(attributes, (0, 0, 0, 1))
    batch_index = torch.arange(batch, device=attributes.device)[:, None, None]
    primitive_count = indices.shape[0] if indices is not None else count
    primitive_index = torch.where(ids >= 0, ids, primitive_count).long()
    if indices is None:
        return attributes[batch_index, primitive_index]
    assert weights is not None
    vertex_indices = F.pad(indices.long(), (0, 0, 0, 1), value=count)[primitive_index]
    weights = weights.to(attributes.dtype)
    image = attributes[batch_index, vertex_indices[..., 0]] * weights[..., 0, None]
    for corner in range(1, indices.shape[-1]):
        image = image + attributes[batch_index, vertex_indices[..., corner]] * weights[..., corner, None]
    return image


def _restore_batch(output: Dict[str, Tensor], batch_shape: Tuple[int, ...]) -> Dict[str, Tensor]:
    return {name: value.reshape((*batch_shape, *value.shape[1:])) for name, value in output.items()}


def _triangle_output(
    ids: Tensor, uv: Optional[Tensor], depth: Optional[Tensor], faces: Tensor,
    attributes: Optional[Tensor], attributes_domain: str, batch_shape: Tuple[int, ...],
    return_depth: bool, return_interpolation: bool,
) -> Dict[str, Tensor]:
    output = {'mask': ids >= 0}
    if attributes is not None:
        if attributes_domain == 'vertex':
            assert uv is not None
            weights = torch.cat((uv, 1.0 - uv.sum(dim=-1, keepdim=True)), dim=-1)
            output['image'] = _interpolate_attributes(attributes, ids, faces, weights)
        else:
            output['image'] = _interpolate_attributes(attributes, ids)
    if return_depth:
        assert depth is not None
        output['depth'] = depth
    if return_interpolation:
        assert uv is not None
        output['interpolation_id'] = ids
        output['interpolation_uv'] = uv
    return _restore_batch(output, batch_shape)


def _load_rasterization_kernels():
    try:
        from ._triton import rasterization
    except ModuleNotFoundError as error:
        if error.name == 'triton':
            raise ImportError("Triton is required for GPU rasterization. Install it with `pip install triton`.") from error
        raise
    return rasterization
