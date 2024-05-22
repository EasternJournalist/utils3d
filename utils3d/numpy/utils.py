import numpy as np
from typing import *
from numbers import Number

from ._helpers import batched
from . import transforms
from . import mesh

__all__ = [
    'sliding_window_1d',
    'sliding_window_nd',
    'sliding_window_2d',
    'max_pool_1d',
    'max_pool_2d',
    'max_pool_nd',
    'depth_edge',
    'depth_aliasing',
    'interpolate',
    'image_scrcoord',
    'image_uv',
    'image_mesh',
    'image_mesh_from_depth',
    'chessboard',
    'cube',
    'camera_frustum',
    'to4x4'
]


def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Return x view of the input array with x sliding window of the given kernel size and stride.
    The sliding window is performed over the given axis, and the window dimension is append to the end of the output array's shape.

    Args:
        x (np.ndarray): input array with shape (..., axis_size, ...)
        kernel_size (int): size of the sliding window
        stride (int): stride of the sliding window
        axis (int): axis to perform sliding window over
    
    Returns:
        a_sliding (np.ndarray): view of the input array with shape (..., n_windows, ..., kernel_size), where n_windows = (axis_size - kernel_size + 1) // stride
    """
    assert x.shape[axis] >= window_size, f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    axis = axis % x.ndim
    shape = (*x.shape[:axis], (x.shape[axis] - window_size + 1) // stride, *x.shape[axis + 1:], window_size)
    strides = (*x.strides[:axis], stride * x.strides[axis], *x.strides[axis + 1:], x.strides[axis])
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding


def sliding_window_nd(x: np.ndarray, window_size: Tuple[int,...], stride: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def sliding_window_2d(x: np.ndarray, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)


def max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        padding_arr = np.full((*x.shape[:axis], padding, *x.shape[axis + 1:]), fill_value=fill_value, dtype=x.dtype)
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool


def max_pool_nd(x: np.ndarray, kernel_size: Tuple[int,...], stride: Tuple[int,...], padding: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x


def max_pool_2d(x: np.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)


def depth_edge(depth: np.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute the edge mask of x depth map. The edge is defined as the pixels whose neighbors have x large difference in depth.
    
    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff = (max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (max_pool_2d(np.where(mask, depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(np.where(mask, -depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def depth_aliasing(depth: np.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute the map that indicates the aliasing of x depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff_max = max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2) + depth
    else:
        diff_max = max_pool_2d(np.where(mask, depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = max_pool_2d(np.where(mask, -depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2) + depth
    diff = np.minimum(diff_max, diff_min)

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def interpolate(bary: np.ndarray, tri_id: np.ndarray, attr: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Interpolate with given barycentric coordinates and triangle indices

    Args:
        bary (np.ndarray): shape (..., 3), barycentric coordinates
        tri_id (np.ndarray): int array of shape (...), triangle indices
        attr (np.ndarray): shape (N, M), vertices attributes
        faces (np.ndarray): int array of shape (T, 3), face vertex indices

    Returns:
        np.ndarray: shape (..., M) interpolated result
    """
    faces_ = np.concatenate([np.zeros((1, 3), dtype=faces.dtype), faces + 1], axis=0)
    attr_ = np.concatenate([np.zeros((1, attr.shape[1]), dtype=attr.dtype), attr], axis=0)
    return np.sum(bary[..., None] * attr_[faces_[tri_id + 1]], axis=-2)


def image_scrcoord(
        width: int,
        height: int,
    ) -> np.ndarray:
    """
    Get image space coordinates, ranging in [0, 1].
    [0, 0] is the bottom-left corner of the image.

    Args:
        width (int): image width
        height (int): image height

    Returns:
        (np.ndarray): shape (height, width, 2)
    """
    x, y = np.meshgrid(
        np.linspace(0.5 / width, 1 - 0.5 / width, width, dtype=np.float32),
        np.linspace(1 - 0.5 / height, 0.5 / height, height, dtype=np.float32),
        indexing='xy'
    )
    return np.stack([x, y], axis=2)


def image_uv(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Get image space UV grid, ranging in [0, 1]. 

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        np.ndarray: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = np.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, dtype=dtype)
    v = np.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, dtype=dtype)
    return np.concatenate([
        u[None, :, None].repeat(bottom - top, axis=0),
        v[:, None, None].repeat(right - left, axis=1)
    ], axis=2)


def image_mesh(
    height: int,
    width: int,
    mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get x quad mesh regarding image pixel uv coordinates as vertices and image grid as faces.

    Args:
        width (int): image width
        height (int): image height
        mask (np.ndarray, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

    Returns:
        uv (np.ndarray): uv corresponding to pixels as described in image_uv()
        faces (np.ndarray): quad faces connecting neighboring pixels
        indices (np.ndarray, optional): indices of vertices in the original mesh
    """
    if mask is not None:
        assert mask.shape[0] == height and mask.shape[1] == width
        assert mask.dtype == np.bool_
    uv = image_uv(height, width).reshape((-1, 2))
    row_faces = np.stack([np.arange(0, width - 1, dtype=np.int32), np.arange(width, 2 * width - 1, dtype=np.int32), np.arange(1 + width, 2 * width, dtype=np.int32), np.arange(1, width, dtype=np.int32)], axis=1)
    faces = (np.arange(0, (height - 1) * width, width, dtype=np.int32)[:, None, None] + row_faces[None, :, :]).reshape((-1, 4))
    if mask is not None:
        quad_mask = (mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]).ravel()
        faces = faces[quad_mask]
        faces, uv, indices = mesh.remove_unreferenced_vertices(faces, uv, return_indices=True)
        return uv, faces, indices
    return uv, faces


def image_mesh_from_depth(
    depth: np.ndarray,
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None,
    *vertice_attrs: np.ndarray,
    atol: float = None,
    rtol: float = None,
    remove_by_depth: bool = False,
    return_uv: bool = False,
    return_indices: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Get x triangle mesh by lifting depth map to 3D.

    Args:
        depth (np.ndarray): [H, W] depth map
        extrinsics (np.ndarray, optional): [4, 4] extrinsics matrix. Defaults to None.
        intrinsics (np.ndarray, optional): [3, 3] intrinsics matrix. Defaults to None.
        *vertice_attrs (np.ndarray): [H, W, C] vertex attributes. Defaults to None.
        atol (float, optional): absolute tolerance. Defaults to None.
        rtol (float, optional): relative tolerance. Defaults to None.
            triangles with vertices having depth difference larger than atol + rtol * depth will be marked.
        remove_by_depth (bool, optional): whether to remove triangles with large depth difference. Defaults to True.
        return_uv (bool, optional): whether to return uv coordinates. Defaults to False.
        return_indices (bool, optional): whether to return indices of vertices in the original mesh. Defaults to False.

    Returns:
        vertices (np.ndarray): [N, 3] vertices
        faces (np.ndarray): [T, 3] faces
        *vertice_attrs (np.ndarray): [N, C] vertex attributes
        image_uv (np.ndarray, optional): [N, 2] uv coordinates
        ref_indices (np.ndarray, optional): [N] indices of vertices in the original mesh
    """
    height, width = depth.shape
    image_uv, image_face = image_mesh(height, width)
    depth = depth.reshape(-1)
    pts = transforms.unproject_cv(image_uv, depth, extrinsics, intrinsics)
    image_face = mesh.triangulate(image_face, vertices=pts)
    ref_indices = None
    ret = []
    if atol is not None or rtol is not None:
        atol = 0 if atol is None else atol
        rtol = 0 if rtol is None else rtol
        mean = depth[image_face].mean(axis=1)
        diff = np.max(np.abs(depth[image_face] - depth[image_face[:, [1, 2, 0]]]), axis=1)
        mask = (diff <= atol + rtol * mean)
        image_face_ = image_face[mask]
        image_face_, ref_indices = mesh.remove_unreferenced_vertices(image_face_, return_indices=True)

    remove = remove_by_depth and ref_indices is not None
    if remove:
        pts = pts[ref_indices]
        image_face = image_face_
    ret += [pts, image_face]
    for attr in vertice_attrs:
        ret.append(attr.reshape(-1, attr.shape[-1]) if not remove else attr.reshape(-1, attr.shape[-1])[ref_indices])
    if return_uv:
        ret.append(image_uv if not remove else image_uv[ref_indices])
    if return_indices and ref_indices is not None:
        ret.append(ref_indices)
    return tuple(ret)


def chessboard(width: int, height: int, grid_size: int, color_a: np.ndarray, color_b: np.ndarray) -> np.ndarray:
    """get x chessboard image

    Args:
        width (int): image width
        height (int): image height
        grid_size (int): size of chessboard grid
        color_a (np.ndarray): color of the grid at the top-left corner
        color_b (np.ndarray): color in complementary grid cells

    Returns:
        image (np.ndarray): shape (height, width, channels), chessboard image
    """
    x = np.arange(width) // grid_size
    y = np.arange(height) // grid_size
    mask = (x[None, :] + y[:, None]) % 2
    image = (1 - mask[..., None]) * color_a + mask[..., None] * color_b
    return image

def cube():
    """
    Get x cube mesh of size 1 centered at origin.

    Returns:
        vertices (np.ndarray): shape (8, 3) 
        faces (np.ndarray): shape (12, 3)
    """
    vertices = np.array([
        [-0.5, 0.5, 0.5],   [0.5, 0.5, 0.5],   [0.5, -0.5, 0.5],   [-0.5, -0.5, 0.5], # v0-v1-v2-v3
        [-0.5, 0.5, -0.5],  [0.5, 0.5, -0.5],  [0.5, -0.5, -0.5],  [-0.5, -0.5, -0.5] # v4-v5-v6-v7
    ], dtype=np.float32).reshape((-1, 3))

    faces = np.array([
        [0, 1, 2, 3], # v0-v1-v2-v3 (front)
        [4, 5, 1, 0], # v4-v5-v1-v0 (top)
        [3, 2, 6, 7], # v3-v2-v6-v7 (bottom)
        [5, 4, 7, 6], # v5-v4-v7-v6 (back)
        [1, 5, 6, 2], # v1-v5-v6-v2 (right)
        [4, 0, 3, 7]  # v4-v0-v3-v7 (left)
    ], dtype=np.int32)

    return vertices, faces


def camera_frustum(extrinsics: np.ndarray, intrinsics: np.ndarray, depth: float = 1.0) -> Tuple[np.ndarray, ...]:
    """
    Get x triangle mesh of camera frustum.
    """
    assert extrinsics.shape == (4, 4) and intrinsics.shape == (3, 3)
    vertices = transforms.unproject_cv(
        np.array([[0, 0], [0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32), 
        np.array([0] + [depth] * 4, dtype=np.float32), 
        extrinsics, 
        intrinsics
    )
    edges = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4], 
        [1, 2], [2, 3], [3, 4], [4, 1]
    ], dtype=np.int32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [1, 2, 3],
        [1, 3, 4]
    ], dtype=np.int32)
    return vertices, edges, faces


def to4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix and translation vector to 4x4 transformation matrix.

    Args:
        R (np.ndarray): [..., 3, 3] rotation matrix
        t (np.ndarray): [..., 3] translation vector

    Returns:
        np.ndarray: [..., 4, 4] transformation matrix
    """
    assert R.shape[:-2] == t.shape[:-1]
    assert R.shape[-1] == 3 and R.shape[-2] == 3
    return np.concatenate([
        np.concatenate([R, t[:, None]], axis=1), 
        np.concatenate([np.zeros_like(t), np.ones_like(t[..., :1])], axis=-1)[..., None, :]
    ], axis=-2)