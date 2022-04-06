import numpy as np
from typing import Tuple

def to_linear_depth(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    return (2 * near * far) / (far + near - (2 * depth_buffer - 1) * (far - near))

def triangulate(indices: np.ndarray) -> np.ndarray:
    assert len(indices.shape) == 2
    if indices.shape[1] == 3:
        return indices
    n = indices.shape[1]
    loop_indice = np.stack([np.zeros(n - 2, dtype=int), np.arange(1, n - 1, 1, dtype=int), np.arange(2, n, 1, dtype=int)], axis=1)
    return indices[:, loop_indice].reshape(-1, 3)

def perspective_from_image(fov: float, width: int, height: int, near: float, far: float) -> np.ndarray:
    return np.array([
        [1. / (np.tan(fov / 2) * (width / max(width, height))), 0., 0., 0.],
        [0., 1. / (np.tan(fov / 2) * (height / max(width, height))), 0., 0.],
        [0., 0., (near + far) / (near - far), 2. * near * far / (near - far)],
        [0., 0., -1., 0.] 
    ])

def perspective_from_fov_xy(fov_x: float, fov_y: float, near: float, far: float) -> np.ndarray:
    return np.array([
        [1. / np.tan(fov_x / 2), 0., 0., 0.],
        [0., 1. / np.tan(fov_y / 2), 0., 0.],
        [0., 0., (near + far) / (near - far), 2. * near * far / (near - far)],
        [0., 0., -1., 0.] 
    ])

def image_uv(width: int, height: int) -> np.ndarray:
    u = np.linspace(0.5 / width, 1. - 0.5 / width, width)
    v = np.linspace(0.5 / height, 1. - 0.5 / height, height)
    return np.concatenate([u[None, :, None].repeat(height, axis=0), v[:, None, None].repeat(width, axis=1)], axis=2)

def image_mesh(width: int, height: int, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    if mask is not None:
        assert mask.shape[0] == height and mask.shape[1] == width
        assert mask.dtype == np.bool_
    vertices = image_uv(width, height).reshape((-1, 2))
    row_indices = np.stack([np.arange(0, width - 1), np.arange(1, width), np.arange(1 + width, 2 * width), np.arange(width, 2 * width - 1)], axis=1)
    indices = (np.arange(0, (height - 1) * width, width)[:, None, None] + row_indices[None, :, :]).reshape((-1, 4))
    if mask is not None:
        quad_mask = (mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]).ravel()
        indices = indices[quad_mask]
        fewer_indices, inv_map = np.unique(indices, return_inverse=True)
        indices = inv_map.reshape((-1, 4))
        vertices = vertices[fewer_indices]
    return vertices, indices