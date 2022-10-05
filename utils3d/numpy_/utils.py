import numpy as np
from typing import Tuple

__all__ = [
    'interpolate',
    'to_linear_depth',
    'to_depth_buffer',
    'image_uv',
    'image_mesh',
    'chessboard'
]

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

def to_linear_depth(screen_depth: np.ndarray, near: float, far: float) -> np.ndarray:
    return (2 * near * far) / (far + near - (2 * screen_depth - 1) * (far - near))

def to_screen_depth(linear_depth: np.ndarray, near: float, far: float) -> np.ndarray:
    ndc_depth = (near + far - 2. * near * far / linear_depth) / (far - near)
    return 0.5 * ndc_depth + 0.5

def image_uv(width: int, height: int) -> np.ndarray:
    """Get image space UV grid, ranging in [0, 1]. 

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
    u = np.linspace(0.5 / width, 1. - 0.5 / width, width, dtype=np.float32)
    v = np.linspace(0.5 / height, 1. - 0.5 / height, height, dtype=np.float32)
    return np.concatenate([u[None, :, None].repeat(height, axis=0), v[:, None, None].repeat(width, axis=1)], axis=2)

def image_mesh(width: int, height: int, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Get a quad mesh regarding image pixel uv coordinates as vertices and image grid as faces.

    Args:
        width (int): image width
        height (int): image height
        mask (np.ndarray, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

    Returns:
        uv (np.ndarray): uv corresponding to pixels as described in image_uv()
        faces (np.ndarray): quad faces connecting neighboring pixels
    """
    if mask is not None:
        assert mask.shape[0] == height and mask.shape[1] == width
        assert mask.dtype == np.bool_
    uv = image_uv(width, height).reshape((-1, 2))
    row_faces = np.stack([np.arange(0, width - 1, dtype=np.int32), np.arange(width, 2 * width - 1, dtype=np.int32), np.arange(1 + width, 2 * width, dtype=np.int32), np.arange(1, width, dtype=np.int32)], axis=1)
    faces = (np.arange(0, (height - 1) * width, width, dtype=np.int32)[:, None, None] + row_faces[None, :, :]).reshape((-1, 4))
    if mask is not None:
        quad_mask = (mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]).ravel()
        faces = faces[quad_mask]
        fewer_indices, inv_map = np.unique(faces, return_inverse=True)
        faces = inv_map.reshape((-1, 4))
        uv = uv[fewer_indices]
    return uv, faces

def chessboard(width: int, height: int, grid_size: int, color_a: np.ndarray, color_b: np.ndarray) -> np.ndarray:
    """get a chessboard image

    Args:
        width (int): image width
        height (int): image height
        grid_size (int): size of chessboard grid
        color_a (np.ndarray): color of the grid at the top-left corner
        color_b (np.ndarray): color in complementary grids

    Returns:
        image (np.ndarray): shape (height, width, channels), chessboard image
    """
    x = np.arange(width) // grid_size
    y = np.arange(height) // grid_size
    mask = (x[None, :] + y[:, None]) % 2
    image = (1 - mask[..., None]) * color_a + mask[..., None] * color_b
    return image
