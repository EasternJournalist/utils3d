import math
from numbers import Integral, Real
from typing import List, Tuple

import numpy as np
from numpy import ndarray


__all__ = [
    'voxel_grid_sampling',
    'hierarchical_voxel_sampling',
]


def _check_points(points: ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if points.dtype.kind != 'f' or points.dtype.itemsize > 8:
        raise TypeError("points must have a float16, float32 or float64 dtype.")


def _point_bounds(points: ndarray) -> Tuple[ndarray, List[float]]:
    lower = points.min(axis=0).astype(np.float64)
    upper = points.max(axis=0).astype(np.float64)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("points must contain only finite coordinates.")
    spans = [float(upper[i]) - float(lower[i]) for i in range(3)]
    if not all(math.isfinite(span) for span in spans):
        raise ValueError("The coordinate range is too large for float64 arithmetic.")
    return lower, spans


def _spread_bits(values: ndarray) -> ndarray:
    # Interleave 21 coordinate bits into the nonnegative bits of an int64 key.
    values = (values | (values << 32)) & 0x1F00000000FFFF
    values = (values | (values << 16)) & 0x1F0000FF0000FF
    values = (values | (values << 8)) & 0x100F00F00F00F00F
    values = (values | (values << 4)) & 0x10C30C30C30C30C3
    return (values | (values << 2)) & 0x1249249249249249


def _cell_priority(keys: ndarray) -> ndarray:
    # Spatially disperse partial refinement without allocating random state.
    values = (keys ^ (keys >> 32)) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return values ^ (values >> 16)


def voxel_grid_sampling(points: ndarray, voxel_size: float) -> ndarray:
    """Select one original point per occupied cubic voxel.

    Parameters
    ----
    - `points`: (ndarray) floating-point array of shape `(N, 3)`, with finite coordinates.
    - `voxel_size`: (float) finite, positive voxel side length.

    Returns
    -------
    - `indices`: (ndarray) int64 array of shape `(M,)`. The smallest original index
        in each occupied voxel, ordered lexicographically by voxel coordinates.

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
    """
    _check_points(points)
    if isinstance(voxel_size, (bool, np.bool_)) or not isinstance(voxel_size, Real):
        raise TypeError("voxel_size must be a real scalar.")
    voxel_size = float(voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive.")
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    lower, spans = _point_bounds(points)
    ratios = [span / voxel_size for span in spans]
    if any(not math.isfinite(ratio) or ratio >= 2**63 for ratio in ratios):
        raise ValueError("voxel_size is too small for int64 voxel coordinates.")
    dimensions = [math.floor(ratio) + 1 for ratio in ratios]
    keys = np.zeros(len(points), dtype=np.int64) if math.prod(dimensions) <= 2**63 else None
    coordinates = []
    for axis in range(3):
        coordinate = points[:, axis].astype(np.float64)
        coordinate -= lower[axis]
        coordinate /= voxel_size
        coordinate = coordinate.astype(np.int64)
        if keys is not None:
            keys *= dimensions[axis]
            keys += coordinate
        else:
            coordinates.append(coordinate)

    starts = np.ones(len(points), dtype=bool)
    if keys is not None:
        order = np.argsort(keys, kind='stable')
        keys = keys[order]
        starts[1:] = keys[1:] != keys[:-1]
    else:
        order = np.lexsort(tuple(reversed(coordinates)))
        starts[1:] = False
        for coordinate in coordinates:
            coordinate = coordinate[order]
            starts[1:] |= coordinate[1:] != coordinate[:-1]
    return order[starts].astype(np.int64, copy=False)


def hierarchical_voxel_sampling(points: ndarray, num_samples: int) -> ndarray:
    """Select approximately a target number of spatially distributed original points.

    Parameters
    ----
    - `points`: (ndarray) floating-point array of shape `(N, 3)`, with finite coordinates.
    - `num_samples`: (int) nonnegative target number of samples.

    Returns
    -------
    - `indices`: (ndarray) int64 array of shape `(M,)`, containing distinct original
        indices. Output order is spatial, not a progressive FPS ordering.

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
    coordinate range raise ValueError.
    """
    _check_points(points)
    if isinstance(num_samples, (bool, np.bool_)) or not isinstance(num_samples, Integral):
        raise TypeError("num_samples must be an integer.")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative.")
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    lower, spans = _point_bounds(points)
    if num_samples == 0:
        return np.empty(0, dtype=np.int64)
    if num_samples >= len(points):
        return np.arange(len(points), dtype=np.int64)
    extent = max(spans)
    if num_samples == 1 or extent == 0:
        return np.zeros(1, dtype=np.int64)

    keys = np.zeros(len(points), dtype=np.int64)
    for axis in range(3):
        coordinate = points[:, axis].astype(np.float64)
        coordinate -= lower[axis]
        coordinate /= extent
        coordinate *= 2**21
        np.clip(coordinate, 0, 2**21 - 1, out=coordinate)
        keys |= _spread_bits(coordinate.astype(np.int64)) << axis
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    differences = keys[1:] ^ keys[:-1]

    low, high, count = 0, 22, 1
    while high - low > 1:
        level = (low + high) // 2
        level_count = 1 + np.count_nonzero(differences >> (3 * (21 - level)))
        if level_count <= num_samples:
            low, count = level, level_count
        else:
            high = level
    shift = 3 * (21 - low)
    if low == 21 or count == num_samples:
        starts = np.concatenate((np.zeros(1, dtype=np.int64), np.flatnonzero(differences >> shift) + 1))
        return order[starts].astype(np.int64, copy=False)

    child_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.flatnonzero(differences >> (shift - 3)) + 1))
    parent_keys = keys[child_starts] >> shift
    parent_first = np.ones(len(child_starts), dtype=bool)
    parent_first[1:] = parent_keys[1:] != parent_keys[:-1]
    parent_starts = np.flatnonzero(parent_first)
    child_counts = np.diff(np.append(parent_starts, len(child_starts)))
    gains = child_counts - 1
    candidates = np.flatnonzero(gains)
    priorities = _cell_priority(parent_keys[parent_starts[candidates]])
    candidates = candidates[np.argsort(priorities, kind='stable')]
    selected = candidates[np.cumsum(gains[candidates]) <= num_samples - count]
    split = np.zeros(count, dtype=bool)
    split[selected] = True
    keep = parent_first | np.repeat(split, child_counts)
    return order[child_starts[keep]].astype(np.int64, copy=False)
