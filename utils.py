import numpy as np
from typing import Tuple

def to_linear_depth(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    return (2 * near * far) / (far + near - (2 * depth_buffer - 1) * (far - near))

def to_depth_buffer(linear_depth: np.ndarray, near: float, far: float) -> np.ndarray:
    ndc_depth =(near + far - 2. * near * far / linear_depth) / (far - near)
    return 0.5 * ndc_depth + 0.5

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

def instrinsic_from_image(fov: float, width: int, height: int) -> np.ndarray:
    return np.array([
        [0.5 / (np.tan(fov / 2) * (width / max(width, height))), 0., 0.5],
        [0., 0.5 / (np.tan(fov / 2) * (height / max(width, height))), 0.5],
        [0., 0., 1.],
    ])

def intrinsic_from_fov_xy(fov_x: float, fov_y: float) -> np.ndarray:
    return np.array([
        [0.5 / np.tan(fov_x / 2), 0., 0.5],
        [0., 0.5 / np.tan(fov_y / 2), 0.5],
        [0., 0., 1.],
    ])

def pixel_to_uv(pixel: np.ndarray, W: int, H: int, revese_h: bool=True) -> np.ndarray:
    """pixel: pixel coordinrates defined on image space,  x range is (0, W - 1), y range is (0, H - 1)
    
    """
    pixel_x = pixel[..., 0]
    pixel_y = pixel[..., 1]
    if revese_h:
        pixel_y = H - 1 - pixel_y
    u = (pixel_x + 0.5) / W
    v = (pixel_y + 0.5) / H
    return np.stack([u, v], axis=-1)

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

def image_mesh_3d_cv(width: int, height: int, normalized_intrinsic: np.ndarray, linear_depth: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    vertices_uv, indices = image_mesh(width, height, mask)
    vertices_uv_ndc = vertices_uv * 2 - 1
    vertices = (np.concatenate([vertices_uv_ndc, np.ones_like(vertices_uv[:, :1])], axis=1) @ np.linalg.inv(normalized_intrinsic).transpose()) * linear_depth[:, None]
    return vertices, indices

def image_mesh_3d_gl(width: int, height: int, perspective: np.ndarray, linear_depth: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    vertices_uv, indices = image_mesh(width, height, mask)
    vertices_xy = (2 * vertices_uv - 1) * linear_depth[:, None] @ np.linalg.inv(perspective[:2, :2]).transpose()
    vertices = np.concatenate([vertices_xy, -linear_depth[:, None]], axis=1)
    return vertices, indices

def projection(vertices: np.ndarray, model_matrix: np.ndarray = None, view_matrix: np.ndarray = None, projection_matrix: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrice)

    Args:
        vertices (np.ndarray): 3D vertices positions of shape (..., 3)
        model_matrix (np.ndarray): row major model to world matrix of shape (4, 4)
        view_matrix (np.ndarray): camera to world matrix of shape (4, 4)
        projection_matrix (np.ndarray): projection matrix of shape (4, 4)

    Returns:
        scr_coord (np.ndarray): vertex screen space coordinates of shape (..., 2), value ranging in [0, 1]. The origin (0., 0.) is corresponding to the bottom-left corner of the screen
        zbuffer (np.ndarray): vertex z-buffer of shape (...)
    """
    assert vertices.shape[-1] == 3
    if model_matrix is None: model_matrix = np.eye(4, dtype=vertices.dtype)
    if view_matrix is None: view_matrix = np.eye(4, dtype=vertices.dtype)
    if projection_matrix is None: projection_matrix = np.eye(4, dtype=vertices.dtype)

    vertices = np.concatenate([vertices, np.ones_like(vertices[..., :1])], axis=1)
    clip_coord = vertices (model_matrix @ np.linalg.inv(view_matrix).T @ projection_matrix.T)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5

    zbuffer = scr_coord[..., 2]
    return scr_coord, zbuffer