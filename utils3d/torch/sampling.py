import math
from numbers import Integral, Real
from typing import List, Tuple

import torch
from torch import Tensor

from .utils import lexsort


__all__ = [
    'voxel_grid_sampling',
    'hierarchical_voxel_sampling',
]


def _check_points(points: Tensor) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if points.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("points must have a float16, bfloat16, float32 or float64 dtype.")


def _point_bounds(points: Tensor) -> Tuple[Tensor, List[float]]:
    bounds = torch.stack((points.amin(dim=0).to(torch.float64), points.amax(dim=0).to(torch.float64)))
    lower, upper = bounds.cpu().tolist()
    if not all(math.isfinite(value) for value in lower + upper):
        raise ValueError("points must contain only finite coordinates.")
    spans = [upper[i] - lower[i] for i in range(3)]
    if not all(math.isfinite(span) for span in spans):
        raise ValueError("The coordinate range is too large for float64 arithmetic.")
    return bounds[0], spans


def _spread_bits(values: Tensor) -> Tensor:
    # Interleave 21 coordinate bits into the nonnegative bits of an int64 key.
    values = (values | (values << 32)) & 0x1F00000000FFFF
    values = (values | (values << 16)) & 0x1F0000FF0000FF
    values = (values | (values << 8)) & 0x100F00F00F00F00F
    values = (values | (values << 4)) & 0x10C30C30C30C30C3
    return (values | (values << 2)) & 0x1249249249249249


def _cell_priority(keys: Tensor) -> Tensor:
    # Spatially disperse partial refinement without allocating random state.
    values = (keys ^ (keys >> 32)) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return values ^ (values >> 16)


@torch.no_grad()
def voxel_grid_sampling(points: Tensor, voxel_size: float) -> Tensor:
    """Select one original point per occupied cubic voxel.

    Parameters
    ----
    - `points`: (Tensor) floating-point tensor of shape `(N, 3)`, with finite coordinates.
    - `voxel_size`: (float) finite, positive voxel side length.

    Returns
    -------
    - `indices`: (Tensor) int64 tensor of shape `(M,)`, on the input device. The smallest
        original index in each occupied voxel, ordered lexicographically by voxel coordinates.

    Notes
    -----
    The grid origin is the coordinate-wise minimum of `points`. Grid arithmetic
    uses float64. Every input point is within `sqrt(3) * voxel_size` of a selected
    point, up to floating-point rounding; there is no minimum separation guarantee.
    This is voxel sampling, not FPS or Poisson-disk sampling.

    Sorting requires O(N log N) time in the comparison model and O(N) workspace.
    Empty input returns empty indices. Duplicate coordinates share a voxel.
    Nonfinite inputs, an overflowing coordinate range, or a coordinate span
    divided by `voxel_size` reaching 2**63 raise ValueError. When a packed voxel
    key would overflow, a collision-free lexicographic sort is used instead.
    Bounds are read on the CPU once; point data and sorting stay on the input
    device. Index selection does not build an autograd graph.
    """
    _check_points(points)
    if isinstance(voxel_size, bool) or not isinstance(voxel_size, Real):
        raise TypeError("voxel_size must be a real scalar.")
    voxel_size = float(voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive.")
    if len(points) == 0:
        return torch.empty(0, dtype=torch.int64, device=points.device)
    lower, spans = _point_bounds(points)
    ratios = [span / voxel_size for span in spans]
    if any(not math.isfinite(ratio) or ratio >= 2**63 for ratio in ratios):
        raise ValueError("voxel_size is too small for int64 voxel coordinates.")
    dimensions = [math.floor(ratio) + 1 for ratio in ratios]
    keys = torch.zeros(len(points), dtype=torch.int64, device=points.device) if math.prod(dimensions) <= 2**63 else None
    coordinates = []
    # A device divisor avoids scalar reciprocal-multiply rounding at voxel boundaries.
    scale = torch.tensor(voxel_size, dtype=torch.float64, device=points.device)
    for axis in range(3):
        coordinate = points[:, axis].to(torch.float64) - lower[axis]
        coordinate.div_(scale)
        coordinate = coordinate.to(torch.int64)
        if keys is not None:
            keys.mul_(dimensions[axis]).add_(coordinate)
        else:
            coordinates.append(coordinate)

    starts = torch.ones(len(points), dtype=torch.bool, device=points.device)
    if keys is not None:
        keys, order = torch.sort(keys, stable=True)
        starts[1:] = keys[1:] != keys[:-1]
    else:
        order = lexsort(tuple(reversed(coordinates)))
        starts[1:] = False
        for coordinate in coordinates:
            coordinate = coordinate[order]
            starts[1:] |= coordinate[1:] != coordinate[:-1]
    return order[starts]


@torch.no_grad()
def hierarchical_voxel_sampling(points: Tensor, num_samples: int) -> Tensor:
    """Select approximately a target number of spatially distributed original points.

    Parameters
    ----
    - `points`: (Tensor) floating-point tensor of shape `(N, 3)`, with finite coordinates.
    - `num_samples`: (int) nonnegative target number of samples.

    Returns
    -------
    - `indices`: (Tensor) int64 tensor of shape `(M,)`, on the input device, containing
        distinct original indices. Output order is spatial, not a progressive FPS ordering.

    Notes
    -----
    An implicit octree uses a cube anchored at the coordinate-wise minimum, with
    side length equal to the largest coordinate span. Coordinates are quantized
    in float64 to 21 bits per axis; the upper cube face belongs to the last cell.
    A single stable Morton sort is followed by a search over grid levels. Only
    some cells of the last level are refined, using deterministic hashed cell
    priorities. Final cells differ by at most one level, and each is represented
    by its first point in the sorted Morton order (original index breaks ties).

    For 0 < num_samples < N, let U be the number of occupied finest-grid cells
    and T = min(num_samples, U). Then max(1, T - 6) <= M <= T. Coincident points
    and points closer than the finest resolution can share a cell. A zero target
    or empty input returns empty indices; a target >= N returns all input indices
    in their original order, including coincident points.

    This is approximate uniform coverage, not FPS or Poisson-disk sampling.
    There is no minimum separation guarantee; results depend on grid orientation
    and are not nested across targets. Outliers can reduce the effective spatial
    resolution. Time is O(N log N) in the comparison model, with O(N) workspace
    and no loop over points or samples. Nonfinite inputs or an overflowing
    coordinate range raise ValueError. Bounds and at most five level counts are
    read on the CPU; point data and sorting stay on the input device. Index
    selection does not build an autograd graph.
    """
    _check_points(points)
    if isinstance(num_samples, bool) or not isinstance(num_samples, Integral):
        raise TypeError("num_samples must be an integer.")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative.")
    if len(points) == 0:
        return torch.empty(0, dtype=torch.int64, device=points.device)
    lower, spans = _point_bounds(points)
    if num_samples == 0:
        return torch.empty(0, dtype=torch.int64, device=points.device)
    if num_samples >= len(points):
        return torch.arange(len(points), dtype=torch.int64, device=points.device)
    extent = max(spans)
    if num_samples == 1 or extent == 0:
        return torch.zeros(1, dtype=torch.int64, device=points.device)

    scale = torch.tensor(extent, dtype=torch.float64, device=points.device)
    keys = torch.zeros(len(points), dtype=torch.int64, device=points.device)
    for axis in range(3):
        coordinate = points[:, axis].to(torch.float64) - lower[axis]
        coordinate.div_(scale).mul_(2**21).clamp_(0, 2**21 - 1)
        keys |= _spread_bits(coordinate.to(torch.int64)) << axis
    keys, order = torch.sort(keys, stable=True)
    differences = keys[1:] ^ keys[:-1]

    low, high, count = 0, 22, 1
    while high - low > 1:
        level = (low + high) // 2
        level_count = 1 + int(torch.count_nonzero(differences >> (3 * (21 - level))).item())
        if level_count <= num_samples:
            low, count = level, level_count
        else:
            high = level
    shift = 3 * (21 - low)
    zero = torch.zeros(1, dtype=torch.int64, device=points.device)
    if low == 21 or count == num_samples:
        starts = torch.cat((zero, torch.nonzero(differences >> shift, as_tuple=True)[0] + 1))
        return order[starts]

    child_starts = torch.cat((zero, torch.nonzero(differences >> (shift - 3), as_tuple=True)[0] + 1))
    parent_keys = keys[child_starts] >> shift
    parent_first = torch.ones(len(child_starts), dtype=torch.bool, device=points.device)
    parent_first[1:] = parent_keys[1:] != parent_keys[:-1]
    parent_starts = torch.nonzero(parent_first, as_tuple=True)[0]
    end = torch.full((1,), len(child_starts), dtype=torch.int64, device=points.device)
    child_counts = torch.diff(torch.cat((parent_starts, end)))
    gains = child_counts - 1
    candidates = torch.nonzero(gains, as_tuple=True)[0]
    priorities = _cell_priority(parent_keys[parent_starts[candidates]])
    candidates = candidates[torch.argsort(priorities, stable=True)]
    selected = candidates[torch.cumsum(gains[candidates], dim=0) <= num_samples - count]
    split = torch.zeros(count, dtype=torch.bool, device=points.device)
    split[selected] = True
    keep = parent_first | torch.repeat_interleave(split, child_counts, output_size=len(child_starts))
    return order[child_starts[keep]]
