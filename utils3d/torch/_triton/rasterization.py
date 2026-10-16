from typing import Optional, Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl


_POINT_CORNERS = {
    'triangle': ((0., .5), (-.433013, -.25), (.433013, -.25)),
    'square': ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)),
    'pentagon': ((0., .5), (-.475528, .154508), (-.293893, -.404508), (.293893, -.404508), (.475528, .154508)),
    'hexagon': ((0., .5), (-.433013, .25), (-.433013, -.25), (0., -.5), (.433013, -.25), (.433013, .25)),
    'circle': ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)),
}


@triton.jit
def _power_of_two_scale(magnitude):
    # Exact binary scaling prevents homogeneous determinants from under/overflowing.
    exponent = ((magnitude.to(tl.uint32, bitcast=True) >> 23) & 255).to(tl.int32)
    inverse_exponent = tl.minimum(tl.maximum(254 - exponent, 1), 254)
    return (inverse_exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _clip_distance(x, y, z, w, plane: tl.constexpr):
    coordinate = (x, y, z)[plane // 2]
    return w + coordinate if plane % 2 == 0 else w - coordinate


@triton.jit
def _clip_polygon(x, y, z, w):
    # Six half-spaces can grow a triangle to at most nine vertices.
    index = tl.arange(0, 16)
    count = tl.full((), 3, tl.int32)
    for plane in tl.static_range(6):
        if count > 0:
            distance = _clip_distance(x, y, z, w, plane)
            if tl.sum(((index < count) & (distance < 0)).to(tl.int32), 0) > 0:
                previous = (index + count - 1) % count
                px = tl.gather(x, previous, 0)
                py = tl.gather(y, previous, 0)
                pz = tl.gather(z, previous, 0)
                pw = tl.gather(w, previous, 0)
                pd = tl.gather(distance, previous, 0)
                inside = (index < count) & (distance >= 0)
                crossing = (index < count) & ((distance >= 0) != (pd >= 0))
                denominator = tl.where(crossing, pd - distance, 1.0)
                # Symmetric intersections keep a shared edge independent of its direction.
                ix = tl.div_rn(pd * x - distance * px, denominator)
                iy = tl.div_rn(pd * y - distance * py, denominator)
                iz = tl.div_rn(pd * z - distance * pz, denominator)
                iw = tl.div_rn(pd * w - distance * pw, denominator)
                if plane == 0:
                    ix = -iw
                elif plane == 1:
                    ix = iw
                elif plane == 2:
                    iy = -iw
                elif plane == 3:
                    iy = iw
                elif plane == 4:
                    iz = -iw
                else:
                    iz = iw

                emitted = crossing.to(tl.int32) + inside.to(tl.int32)
                end = tl.cumsum(emitted, 0)
                count = tl.sum(emitted, 0)
                source = tl.sum((end[None, :] <= index[:, None]).to(tl.int32), 1)
                source = tl.minimum(source, 15)
                intersection = tl.gather(crossing, source, 0) & (
                    index == tl.gather(end - emitted, source, 0)
                )
                x = tl.where(intersection, tl.gather(ix, source, 0), tl.gather(x, source, 0))
                y = tl.where(intersection, tl.gather(iy, source, 0), tl.gather(y, source, 0))
                z = tl.where(intersection, tl.gather(iz, source, 0), tl.gather(z, source, 0))
                w = tl.where(intersection, tl.gather(iw, source, 0), tl.gather(w, source, 0))
                x = tl.where(index < count, x, 0.0)
                y = tl.where(index < count, y, 0.0)
                z = tl.where(index < count, z, 0.0)
                w = tl.where(index < count, w, 0.0)
    return x, y, z, w, count


@triton.jit
def _edge_coefficients(ax, ay, aw, bx, by, bw):
    return ay * bw - aw * by, aw * bx - ax * bw, ax * by - ay * bx


@triton.jit
def _covered_edge(ax, ay, bx, by, px, py, orientation):
    dx = (bx - ax) * orientation
    dy = (by - ay) * orientation
    edge = dx * (py - ay) - dy * (px - ax)
    top_left = (dy < 0) | ((dy == 0) & (dx > 0))
    return (edge > 0) | ((edge == 0) & top_left)


@triton.jit
def _rasterize_kernel(
    Positions, Faces, Keys, PreviousKeys,
    NUM_VERTICES, NUM_FACES,
    HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
    CULL_BACKFACE: tl.constexpr, CIRCLE: tl.constexpr, PEEL: tl.constexpr, BLOCK: tl.constexpr,
):
    face = tl.program_id(0) % NUM_FACES
    batch = tl.program_id(0) // NUM_FACES
    index = tl.arange(0, 16)
    vertex = tl.load(Faces + face.to(tl.int64) * 3 + index, index < 3, other=0)
    offset = (batch.to(tl.int64) * NUM_VERTICES + vertex) * 4
    x = tl.load(Positions + offset, index < 3, other=0.0)
    y = tl.load(Positions + offset + 1, index < 3, other=0.0)
    z = tl.load(Positions + offset + 2, index < 3, other=0.0)
    w = tl.load(Positions + offset + 3, index < 3, other=0.0)
    scale = _power_of_two_scale(tl.max(tl.maximum(
        tl.maximum(tl.abs(x), tl.abs(y)), tl.maximum(tl.abs(z), tl.abs(w)),
    ), 0))
    x, y, z, w = x * scale, y * scale, z * scale, w * scale
    x0, y0 = tl.sum(tl.where(index == 0, x, 0.0), 0), tl.sum(tl.where(index == 0, y, 0.0), 0)
    x1, y1 = tl.sum(tl.where(index == 1, x, 0.0), 0), tl.sum(tl.where(index == 1, y, 0.0), 0)
    x2, y2 = tl.sum(tl.where(index == 2, x, 0.0), 0), tl.sum(tl.where(index == 2, y, 0.0), 0)
    z0, w0 = tl.sum(tl.where(index == 0, z, 0.0), 0), tl.sum(tl.where(index == 0, w, 0.0), 0)
    z1, w1 = tl.sum(tl.where(index == 1, z, 0.0), 0), tl.sum(tl.where(index == 1, w, 0.0), 0)
    z2, w2 = tl.sum(tl.where(index == 2, z, 0.0), 0), tl.sum(tl.where(index == 2, w, 0.0), 0)
    a0, b0, c0 = _edge_coefficients(x1, y1, w1, x2, y2, w2)
    a1, b1, c1 = _edge_coefficients(x2, y2, w2, x0, y0, w0)
    a2, b2, c2 = _edge_coefficients(x0, y0, w0, x1, y1, w1)
    determinant = w0 * c0 + w1 * c1 + w2 * c2

    if determinant != 0:
        depth_x = tl.div_rn(z0 * a0 + z1 * a1 + z2 * a2, determinant)
        depth_y = tl.div_rn(z0 * b0 + z1 * b1 + z2 * b2, determinant)
        depth_c = tl.div_rn(z0 * c0 + z1 * c1 + z2 * c2, determinant)
        x, y, z, w, count = _clip_polygon(x, y, z, w)
        divisor = tl.where(w > 0, w, 1.0)
        # Eight subpixel bits and integer edge tests give watertight shared edges.
        sx = tl.floor((tl.div_rn(x, divisor) * 0.5 + 0.5) * (WIDTH * 256) + 0.5).to(tl.int64)
        sy = tl.floor((0.5 - tl.div_rn(y, divisor) * 0.5) * (HEIGHT * 256) + 0.5).to(tl.int64)
        sx0 = tl.sum(tl.where(index == 0, sx, 0), 0)
        sy0 = tl.sum(tl.where(index == 0, sy, 0), 0)
        sw0 = tl.sum(tl.where(index == 0, w, 0.0), 0)
        pixels = tl.arange(0, BLOCK)
        keys = tl.cast(Keys, tl.pointer_type(tl.uint64))

        for corner in range(1, count - 1):
            sx1 = tl.sum(tl.where(index == corner, sx, 0), 0)
            sy1 = tl.sum(tl.where(index == corner, sy, 0), 0)
            sx2 = tl.sum(tl.where(index == corner + 1, sx, 0), 0)
            sy2 = tl.sum(tl.where(index == corner + 1, sy, 0), 0)
            sw1 = tl.sum(tl.where(index == corner, w, 0.0), 0)
            sw2 = tl.sum(tl.where(index == corner + 1, w, 0.0), 0)
            area = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0)
            visible = (area != 0) & (sw0 > 0) & (sw1 > 0) & (sw2 > 0)
            if CULL_BACKFACE:
                visible = visible & (area < 0)
            if visible:
                orientation = tl.where(area > 0, 1, -1)
                xmin = tl.maximum(tl.minimum(tl.minimum(sx0, sx1), sx2) // 256, 0)
                ymin = tl.maximum(tl.minimum(tl.minimum(sy0, sy1), sy2) // 256, 0)
                xmax = tl.minimum(tl.maximum(tl.maximum(sx0, sx1), sx2) // 256, WIDTH - 1)
                ymax = tl.minimum(tl.maximum(tl.maximum(sy0, sy1), sy2) // 256, HEIGHT - 1)
                box_width = xmax - xmin + 1
                box_size = box_width * (ymax - ymin + 1)
                start = tl.program_id(1).to(tl.int64) * BLOCK
                while start < box_size:
                    local = start + pixels
                    px = xmin + local % box_width
                    py = ymin + local // box_width
                    fx, fy = px * 256 + 128, py * 256 + 128
                    covered = (local < box_size)
                    covered &= _covered_edge(sx0, sy0, sx1, sy1, fx, fy, orientation)
                    covered &= _covered_edge(sx1, sy1, sx2, sy2, fx, fy, orientation)
                    covered &= _covered_edge(sx2, sy2, sx0, sy0, fx, fy, orientation)
                    nx = (px.to(tl.float32) + 0.5) * (2.0 / WIDTH) - 1.0
                    ny = 1.0 - (py.to(tl.float32) + 0.5) * (2.0 / HEIGHT)
                    if CIRCLE:
                        t0 = a0 * nx + b0 * ny + c0
                        t1 = a1 * nx + b1 * ny + c1
                        t2 = a2 * nx + b2 * ny + c2
                        total = t0 + t1 + t2
                        u, v = tl.div_rn(t0, total), tl.div_rn(t1, total)
                        # Fan triangles of the canonical square, before depth selection.
                        local_x = tl.where(face % 2 == 0, 0.5 - u, v - 0.5)
                        local_y = tl.where(face % 2 == 0, 0.5 - u - v, 0.5 - u)
                        covered &= local_x * local_x + local_y * local_y <= 0.25
                    depth = depth_x * nx + depth_y * ny + depth_c
                    depth = tl.maximum(depth, -1.0)
                    covered &= depth < 1.0
                    depth = tl.where(depth == 0.0, 0.0, depth)
                    bits = depth.to(tl.uint32, bitcast=True)
                    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
                    # A single unsigned atomic selects both depth and a deterministic face ID.
                    key = (ordered.to(tl.uint64) << 32) | face.to(tl.uint64)
                    pixel = batch.to(tl.int64) * (HEIGHT * WIDTH) + py * WIDTH + px
                    if PEEL:
                        previous = tl.load(
                            tl.cast(PreviousKeys, tl.pointer_type(tl.uint64)) + pixel,
                            covered, other=0xFFFFFFFFFFFFFFFF,
                        )
                        # Compare depth only: equal-depth faces are not separate layers.
                        covered &= ordered.to(tl.uint64) > (previous >> 32)
                    tl.atomic_min(keys + pixel, key, covered, sem="relaxed")
                    start += tl.num_programs(1) * BLOCK


@triton.jit
def _resolve_kernel(
    Positions, ViewPositions, Faces, Keys, Ids, UV, Depth,
    NUM_VERTICES, NUM_PIXELS,
    HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
    RETURN_UV: tl.constexpr, RETURN_DEPTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pixel = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = pixel < NUM_PIXELS
    key = tl.load(tl.cast(Keys, tl.pointer_type(tl.uint64)) + pixel, active, other=0xFFFFFFFFFFFFFFFF)
    valid = active & (key != 0xFFFFFFFFFFFFFFFF)
    face = key.to(tl.uint32).to(tl.int32)
    tl.store(Ids + pixel, tl.where(valid, face, -1), active)
    if RETURN_UV or RETURN_DEPTH:
        batch = pixel // (HEIGHT * WIDTH)
        face_offset = face.to(tl.int64) * 3
        v0 = tl.load(Faces + face_offset, valid, other=0).to(tl.int64) + batch * NUM_VERTICES
        v1 = tl.load(Faces + face_offset + 1, valid, other=0).to(tl.int64) + batch * NUM_VERTICES
        v2 = tl.load(Faces + face_offset + 2, valid, other=0).to(tl.int64) + batch * NUM_VERTICES
        x0 = tl.load(Positions + v0 * 4, valid, other=0.0)
        y0 = tl.load(Positions + v0 * 4 + 1, valid, other=0.0)
        w0 = tl.load(Positions + v0 * 4 + 3, valid, other=0.0)
        x1 = tl.load(Positions + v1 * 4, valid, other=0.0)
        y1 = tl.load(Positions + v1 * 4 + 1, valid, other=0.0)
        w1 = tl.load(Positions + v1 * 4 + 3, valid, other=0.0)
        x2 = tl.load(Positions + v2 * 4, valid, other=0.0)
        y2 = tl.load(Positions + v2 * 4 + 1, valid, other=0.0)
        w2 = tl.load(Positions + v2 * 4 + 3, valid, other=0.0)
        magnitude0 = tl.maximum(tl.maximum(tl.abs(x0), tl.abs(y0)), tl.abs(w0))
        magnitude1 = tl.maximum(tl.maximum(tl.abs(x1), tl.abs(y1)), tl.abs(w1))
        magnitude2 = tl.maximum(tl.maximum(tl.abs(x2), tl.abs(y2)), tl.abs(w2))
        scale = _power_of_two_scale(tl.maximum(tl.maximum(magnitude0, magnitude1), magnitude2))
        x0, y0, w0 = x0 * scale, y0 * scale, w0 * scale
        x1, y1, w1 = x1 * scale, y1 * scale, w1 * scale
        x2, y2, w2 = x2 * scale, y2 * scale, w2 * scale
        nx = ((pixel % WIDTH).to(tl.float32) + 0.5) * (2.0 / WIDTH) - 1.0
        ny = 1.0 - (((pixel // WIDTH) % HEIGHT).to(tl.float32) + 0.5) * (2.0 / HEIGHT)
        a0, b0, c0 = _edge_coefficients(x1, y1, w1, x2, y2, w2)
        a1, b1, c1 = _edge_coefficients(x2, y2, w2, x0, y0, w0)
        a2, b2, c2 = _edge_coefficients(x0, y0, w0, x1, y1, w1)
        t0 = a0 * nx + b0 * ny + c0
        t1 = a1 * nx + b1 * ny + c1
        t2 = a2 * nx + b2 * ny + c2
        total = tl.where(valid, t0 + t1 + t2, 1.0)
        u = tl.div_rn(t0, total)
        v = tl.div_rn(t1, total)
        if RETURN_UV:
            tl.store(UV + pixel * 2, tl.where(valid, u, 0.0), active)
            tl.store(UV + pixel * 2 + 1, tl.where(valid, v, 0.0), active)
        if RETURN_DEPTH:
            z0 = tl.load(ViewPositions + v0 * 4 + 2, valid, other=0.0)
            z1 = tl.load(ViewPositions + v1 * 4 + 2, valid, other=0.0)
            z2 = tl.load(ViewPositions + v2 * 4 + 2, valid, other=0.0)
            vw0 = tl.load(ViewPositions + v0 * 4 + 3, valid, other=1.0)
            vw1 = tl.load(ViewPositions + v1 * 4 + 3, valid, other=1.0)
            vw2 = tl.load(ViewPositions + v2 * 4 + 3, valid, other=1.0)
            z = u * z0 + v * z1 + (1.0 - u - v) * z2
            w = u * vw0 + v * vw1 + (1.0 - u - v) * vw2
            depth = tl.div_rn(-z, w)
            tl.store(Depth + pixel, tl.where(valid, depth, float("inf")), active)


def rasterize_triangles_triton(
    positions: Tensor,
    view_positions: Tensor,
    faces: Tensor,
    size: Tuple[int, int],
    cull_backface: bool,
    return_uv: bool,
    return_depth: bool,
    *,
    circle: bool = False,
    previous_keys: Optional[Tensor] = None,
    depth_keys: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    batch, num_vertices = positions.shape[:2]
    num_faces = faces.shape[0]
    height, width = size
    shape = (batch, height, width)
    keys = torch.full(shape, -1, dtype=torch.int64, device=positions.device) if depth_keys is None else depth_keys.fill_(-1)
    ids = torch.empty(shape, dtype=torch.int32, device=positions.device)
    uv = torch.empty((*shape, 2), dtype=torch.float32, device=positions.device) if return_uv else None
    depth = torch.empty(shape, dtype=torch.float32, device=positions.device) if return_depth else None
    with torch.cuda.device(positions.device):
        if batch and num_faces:
            block = 128
            # Split small meshes across programs without constructing a work queue.
            partitions = min(triton.cdiv(height * width, block), triton.cdiv(256, batch * num_faces))
            _rasterize_kernel[(batch * num_faces, partitions)](
                positions, faces, keys, previous_keys if previous_keys is not None else keys,
                num_vertices, num_faces,
                height, width, cull_backface, circle, previous_keys is not None, block,
                enable_fp_fusion=False,
            )
        if batch:
            _resolve_kernel[(triton.cdiv(ids.numel(), 256),)](
                positions, view_positions, faces, keys, ids,
                uv if uv is not None else ids, depth if depth is not None else ids,
                num_vertices, ids.numel(), height, width, return_uv, return_depth, 256,
                enable_fp_fusion=False,
            )
    return ids, uv, depth


@triton.jit
def _point_setup_kernel(
    Positions, ViewPositions, Projection, Sizes, Output,
    NUM_POINTS, NUM_INSTANCES, SIZE_BATCH_STRIDE, SIZE_POINT_STRIDE,
    HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
    CORNERS: tl.constexpr, SIZE_3D: tl.constexpr, HAS_PROJECTION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = index < NUM_INSTANCES
    batch, point = index // NUM_POINTS, index % NUM_POINTS
    x = tl.load(Positions + index * 4, active, other=0.)
    y = tl.load(Positions + index * 4 + 1, active, other=0.)
    z = tl.load(Positions + index * 4 + 2, active, other=0.)
    w = tl.load(Positions + index * 4 + 3, active, other=0.)
    size = tl.load(Sizes + batch * SIZE_BATCH_STRIDE + point * SIZE_POINT_STRIDE, active, other=0.)
    if SIZE_3D:
        size *= tl.load(ViewPositions + index * 4 + 3, active, other=0.)
    for corner in tl.static_range(len(CORNERS)):
        dx, dy = size * CORNERS[corner][0], size * CORNERS[corner][1]
        if SIZE_3D:
            if HAS_PROJECTION:
                ox = dx * tl.load(Projection + batch * 16, active, other=0.) + dy * tl.load(Projection + batch * 16 + 1, active, other=0.)
                oy = dx * tl.load(Projection + batch * 16 + 4, active, other=0.) + dy * tl.load(Projection + batch * 16 + 5, active, other=0.)
                oz = dx * tl.load(Projection + batch * 16 + 8, active, other=0.) + dy * tl.load(Projection + batch * 16 + 9, active, other=0.)
                ow = dx * tl.load(Projection + batch * 16 + 12, active, other=0.) + dy * tl.load(Projection + batch * 16 + 13, active, other=0.)
            else:
                ox, oy, oz, ow = dx, dy, 0., 0.
        else:
            ox, oy, oz, ow = dx * (2.0 / WIDTH) * w, dy * (2.0 / HEIGHT) * w, 0., 0.
        offset = (index * len(CORNERS) + corner) * 4
        tl.store(Output + offset, x + ox, active)
        tl.store(Output + offset + 1, y + oy, active)
        tl.store(Output + offset + 2, z + oz, active)
        tl.store(Output + offset + 3, w + ow, active)


@triton.jit
def _line_setup_kernel(
    Positions, ViewPositions, Lines, Output, ViewOutput, Intervals,
    NUM_VERTICES, NUM_LINES, NUM_INSTANCES, LINE_WIDTH,
    HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
    RETURN_DEPTH: tl.constexpr, RETURN_UV: tl.constexpr, BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = index < NUM_INSTANCES
    batch, line = index // NUM_LINES, index % NUM_LINES
    v0 = tl.load(Lines + line * 2, active, other=0).to(tl.int64) + batch * NUM_VERTICES
    v1 = tl.load(Lines + line * 2 + 1, active, other=0).to(tl.int64) + batch * NUM_VERTICES
    x0 = tl.load(Positions + v0 * 4, active, other=0.)
    y0 = tl.load(Positions + v0 * 4 + 1, active, other=0.)
    z0 = tl.load(Positions + v0 * 4 + 2, active, other=0.)
    w0 = tl.load(Positions + v0 * 4 + 3, active, other=0.)
    x1 = tl.load(Positions + v1 * 4, active, other=0.)
    y1 = tl.load(Positions + v1 * 4 + 1, active, other=0.)
    z1 = tl.load(Positions + v1 * 4 + 2, active, other=0.)
    w1 = tl.load(Positions + v1 * 4 + 3, active, other=0.)
    magnitude0 = tl.maximum(tl.maximum(tl.abs(x0), tl.abs(y0)), tl.maximum(tl.abs(z0), tl.abs(w0)))
    magnitude1 = tl.maximum(tl.maximum(tl.abs(x1), tl.abs(y1)), tl.maximum(tl.abs(z1), tl.abs(w1)))
    scale = _power_of_two_scale(tl.maximum(magnitude0, magnitude1))
    x0, y0, z0, w0 = x0 * scale, y0 * scale, z0 * scale, w0 * scale
    x1, y1, z1, w1 = x1 * scale, y1 * scale, z1 * scale, w1 * scale
    t0, t1 = tl.full((BLOCK,), 0., tl.float32), tl.full((BLOCK,), 1., tl.float32)
    visible = active
    for plane in tl.static_range(6):
        d0 = _clip_distance(x0, y0, z0, w0, plane)
        d1 = _clip_distance(x1, y1, z1, w1, plane)
        outside0, outside1 = d0 < 0, d1 < 0
        visible &= ~(outside0 & outside1)
        crossing = outside0 != outside1
        denominator = tl.where(crossing, d0 - d1, 1.)
        ix = tl.div_rn(d0 * x1 - d1 * x0, denominator)
        iy = tl.div_rn(d0 * y1 - d1 * y0, denominator)
        iz = tl.div_rn(d0 * z1 - d1 * z0, denominator)
        iw = tl.div_rn(d0 * w1 - d1 * w0, denominator)
        it = tl.div_rn(d0 * t1 - d1 * t0, denominator)
        boundary = -iw if plane % 2 == 0 else iw
        if plane // 2 == 0:
            ix = boundary
        elif plane // 2 == 1:
            iy = boundary
        else:
            iz = boundary
        replace0, replace1 = crossing & outside0, crossing & outside1
        x0, x1 = tl.where(replace0, ix, x0), tl.where(replace1, ix, x1)
        y0, y1 = tl.where(replace0, iy, y0), tl.where(replace1, iy, y1)
        z0, z1 = tl.where(replace0, iz, z0), tl.where(replace1, iz, z1)
        w0, w1 = tl.where(replace0, iw, w0), tl.where(replace1, iw, w1)
        t0, t1 = tl.where(replace0, it, t0), tl.where(replace1, it, t1)

    nx0 = tl.div_rn(x0, tl.where(w0 > 0, w0, 1.))
    ny0 = tl.div_rn(y0, tl.where(w0 > 0, w0, 1.))
    nx1 = tl.div_rn(x1, tl.where(w1 > 0, w1, 1.))
    ny1 = tl.div_rn(y1, tl.where(w1 > 0, w1, 1.))
    dx, dy = (nx1 - nx0) * WIDTH, (ny1 - ny0) * HEIGHT
    length_squared = dx * dx + dy * dy
    visible &= (w0 > 0) & (w1 > 0) & (t1 > t0) & (length_squared > 0)
    inverse_length = tl.rsqrt(tl.where(visible, length_squared, 1.))
    # Beyond this width every viewport sample is already inside the side edges.
    width = tl.minimum(LINE_WIDTH, 2.0 * tl.sqrt(float(WIDTH * WIDTH + HEIGHT * HEIGHT)))
    ox, oy = -dy * inverse_length * (width / WIDTH), dx * inverse_length * (width / HEIGHT)
    for corner in tl.static_range(4):
        if corner < 2:
            x, y, z, w, t = x0, y0, z0, w0, t0
        else:
            x, y, z, w, t = x1, y1, z1, w1, t1
        sign = 1. if corner == 0 or corner == 3 else -1.
        offset = (index * 4 + corner) * 4
        tl.store(Output + offset, tl.where(visible, x + sign * ox * w, 0.), active)
        tl.store(Output + offset + 1, tl.where(visible, y + sign * oy * w, 0.), active)
        tl.store(Output + offset + 2, tl.where(visible, z, 0.), active)
        tl.store(Output + offset + 3, tl.where(visible, w, 0.), active)
        if RETURN_DEPTH:
            for component in tl.static_range(4):
                a = tl.load(ViewPositions + v0 * 4 + component, active, other=0.)
                b = tl.load(ViewPositions + v1 * 4 + component, active, other=0.)
                tl.store(ViewOutput + offset + component, (1. - t) * a + t * b, active)
    if RETURN_UV:
        tl.store(Intervals + index * 2, t0, active)
        tl.store(Intervals + index * 2 + 1, t1, active)


def _polygon_faces(count: int, corners: int, device: torch.device) -> Tensor:
    if count * corners >= 2**31:
        raise ValueError("Expanded primitives must fit in signed 32-bit vertex indices.")
    fan = torch.tensor([(0, i, i + 1) for i in range(1, corners - 1)], dtype=torch.int32, device=device)
    return (torch.arange(count, dtype=torch.int32, device=device)[:, None, None] * corners + fan).reshape(-1, 3)


def expand_points_triton(
    positions: Tensor,
    view_positions: Tensor,
    projection: Optional[Tensor],
    sizes: Tensor,
    size: Tuple[int, int],
    size_in_3d: bool,
    point_shape: str,
) -> Tuple[Tensor, Tensor]:
    batch, num_points = positions.shape[:2]
    corners = _POINT_CORNERS[point_shape]
    faces = _polygon_faces(num_points, len(corners), positions.device)
    expanded = torch.empty((batch, num_points * len(corners), 4), dtype=positions.dtype, device=positions.device)
    if batch and num_points:
        with torch.cuda.device(positions.device):
            _point_setup_kernel[(triton.cdiv(batch * num_points, 128),)](
                positions, view_positions, projection if projection is not None else positions, sizes, expanded,
                num_points, batch * num_points, sizes.stride(0), sizes.stride(1),
                *size, corners, size_in_3d, projection is not None, 128,
                enable_fp_fusion=False,
            )
    return expanded, faces


def expand_lines_triton(
    positions: Tensor,
    view_positions: Tensor,
    lines: Tensor,
    size: Tuple[int, int],
    line_width: float,
    return_uv: bool,
    return_depth: bool,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    batch, num_vertices = positions.shape[:2]
    num_lines = lines.shape[0]
    faces = _polygon_faces(num_lines, 4, positions.device)
    expanded = torch.empty((batch, num_lines * 4, 4), dtype=positions.dtype, device=positions.device)
    view_expanded = torch.empty_like(expanded) if return_depth else expanded
    intervals = torch.empty((batch, num_lines, 2), dtype=positions.dtype, device=positions.device) if return_uv else None
    if batch and num_lines:
        with torch.cuda.device(positions.device):
            _line_setup_kernel[(triton.cdiv(batch * num_lines, 128),)](
                positions, view_positions, lines, expanded, view_expanded,
                intervals if intervals is not None else expanded,
                num_vertices, num_lines, batch * num_lines, line_width, *size, return_depth, return_uv, 128,
                enable_fp_fusion=False,
            )
    return expanded, view_expanded, faces, intervals
