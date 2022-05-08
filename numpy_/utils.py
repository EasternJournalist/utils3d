import numpy as np
from typing import Tuple

def to_linear_depth(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    return (2 * near * far) / (far + near - (2 * depth_buffer - 1) * (far - near))

def to_depth_buffer(linear_depth: np.ndarray, near: float, far: float) -> np.ndarray:
    ndc_depth =(near + far - 2. * near * far / linear_depth) / (far - near)
    return 0.5 * ndc_depth + 0.5

def triangulate(faces: np.ndarray) -> np.ndarray:
    assert len(faces.shape) == 2
    if faces.shape[1] == 3:
        return faces
    n = faces.shape[1]
    loop_indice = np.stack([np.zeros(n - 2, dtype=int), np.arange(1, n - 1, 1, dtype=int), np.arange(2, n, 1, dtype=int)], axis=1)
    return faces[:, loop_indice].reshape(-1, 3)

def perspective_from_fov(fov: float, width: int, height: int, near: float, far: float) -> np.ndarray:
    return np.array([
        [1. / (np.tan(fov / 2) * (width / max(width, height))), 0., 0., 0.],
        [0., 1. / (np.tan(fov / 2) * (height / max(width, height))), 0., 0.],
        [0., 0., (near + far) / (near - far), 2. * near * far / (near - far)],
        [0., 0., -1., 0.] 
    ], dtype=np.float32)

def perspective_from_fov_xy(fov_x: float, fov_y: float, near: float, far: float) -> np.ndarray:
    return np.array([
        [1. / np.tan(fov_x / 2), 0., 0., 0.],
        [0., 1. / np.tan(fov_y / 2), 0., 0.],
        [0., 0., (near + far) / (near - far), 2. * near * far / (near - far)],
        [0., 0., -1., 0.] 
    ], dtype=np.float32)

def instrinsic_from_fov(fov: float, width: int, height: int) -> np.ndarray:
    return np.array([
        [0.5 / (np.tan(fov / 2) * (width / max(width, height))), 0., 0.5],
        [0., 0.5 / (np.tan(fov / 2) * (height / max(width, height))), 0.5],
        [0., 0., 1.],
    ], dtype=np.float32)

def intrinsic_from_fov_xy(fov_x: float, fov_y: float) -> np.ndarray:
    return np.array([
        [0.5 / np.tan(fov_x / 2), 0., 0.5],
        [0., 0.5 / np.tan(fov_y / 2), 0.5],
        [0., 0., 1.],
    ], dtype=np.float32)

def perspective_to_intrinsic(perspective: np.ndarray) -> np.ndarray:
    """OpenGL convention perspective matrix to OpenCV convention intrinsic

    Args:
        perspective (np.ndarray): shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix

    Returns:
        np.ndarray: shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
    """
    fx, fy = perspective[..., 0, 0], perspective[..., 1, 1]
    cx, cy = perspective[..., 0, 2], perspective[..., 1, 2]
    zero = np.zeros_like(fx)
    one = np.full_like(fx, -1)

    matrix = [
        [0.5 * fx,     zero, -0.5 * cx + 0.5],
        [    zero, 0.5 * fy,  0.5 * cy + 0.5],
        [    zero,     zero,             one]]
    return np.stack([np.stack(row, axis=-1) for row in matrix], axis=-2)

def intrinsic_to_perspective(intrinsic: np.ndarray, near: float, far: float) -> np.ndarray:
    """OpenGL convention perspective matrix to OpenCV convention intrinsic

    Args:
        intrinsic (np.ndarray): shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        np.ndarray: shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix
    """
    fx, fy = intrinsic[..., 0, 0], intrinsic[..., 1, 1]
    cx, cy = intrinsic[..., 0, 2], intrinsic[..., 1, 2]
    zero = np.zeros_like(fx)
    negone = np.full_like(fx, -1)
    a = np.full_like(fx, (near + far) / (near - far))
    b = np.full_like(fx, 2. * near * far / (near - far))

    matrix = [
        [2 * fx,   zero,  -2 * cx + 1, zero],
        [  zero, 2 * fy,   2 * cy - 1, zero],
        [  zero,   zero,            a,    b],
        [  zero,   zero,       negone, zero]]
    return np.stack([np.stack(row, axis=-1) for row in matrix], axis=-2)

def extrinsic_to_view(extrinsic: np.ndarray) -> np.ndarray:
    """OpenCV convention camera extrinsic to OpenGL convention view matrix

    Args:
        extrinsic (np.ndarray): shape (4, 4) or (..., 4, 4), OpenCV convention camera extrinsic

    Returns:
        np.ndarray: shape (4, 4) or (..., 4, 4) OpenGL convention view matrix
    """
    return np.linalg.inv(extrinsic) @ np.diag(np.array([1, -1, -1, 1], dtype=extrinsic.dtype))

def view_to_extrinsic(view: np.ndarray) -> np.ndarray:
    """OpenCV convention camera extrinsic to OpenGL convention view matrix

    Args:
        view (np.ndarray): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix

    Returns:
        np.ndarray: shape (4, 4) or (..., 4, 4) OpenCV convention camera extrinsic
    """
    return np.diag([1, -1, -1, 1], dtype=view.dtype) @ np.linalg.inv(view)

def camera_cv_to_gl(extrinsic: np.ndarray, intrinsic: np.ndarray, near: float, far: float):
    """Convert OpenCV convention camera extrinsic & intrinsic to OpenGL convention view matrix and perspective matrix

    Args:
        extrinsic (np.ndarray): shape (4, 4) or (..., 4, 4), OpenCV convention camera extrinsic
        intrinsic (np.ndarray): shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
        near (float): near plane to clip
        far (float): far plane to clip

    Returns:
        view (np.ndarray): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix
        perspective (np.ndarrray): shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix
    """
    return extrinsic_to_view(extrinsic), intrinsic_to_perspective(intrinsic, near, far)

def camera_gl_to_cv(view: np.ndarray, perspective: np.ndarray):
    """Convert OpenGL convention view matrix & perspective matrix to OpenCV convention camera extrinsic & intrinsic 

    Args:
        view (np.ndarray): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix
        perspective (np.ndarray): shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix

    Returns:
        view (np.ndarray): shape (4, 4) or (..., 4, 4), OpenCV convention camera extrinsic
        perspective (np.ndarrray): shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
    """
    return view_to_extrinsic(view), perspective_to_intrinsic(perspective)

def view_look_at(eye: np.ndarray, look_at: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return a view matrix looking at something

    Args:
        eye (np.ndarray): shape (3,) eye position
        look_at (np.ndarray): shape (3,) the point to look at
        up (np.ndarray): shape (3,) head up direction (y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        view: shape (4, 4), view matrix
    """
    z = eye - look_at
    z = z / np.linalg.norm(z, keepdims=True)
    y = up - np.sum(up * z, axis=-1, keepdims=True) * z
    y = y / np.linalg.norm(y, keepdims=True)
    x = np.cross(y, z)
    return np.concatenate([np.stack([x, y, z, eye], axis=-1), np.array([[0., 0., 0., 1.]])], axis=-2).astype(np.float32)

def pixel_to_uv(pixel: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Args:
        pixel(np.ndarray): pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        W (int): image width
        H (int): image height
    Returns:
        uv(np.ndarray): pixel coordinrates defined in uv space, the range is (0, 1)
    """
    uv = (pixel + 0.5) / np.array([width, height])
    return uv

def pixel_to_ndc(pixel: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Args:
        pixel(np.ndarray): pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
        W (int): image width
        H (int): image height
    Returns:
        ndc(np.ndarray): pixel coordinrates defined in ndc space, the range is (-1, 1)
    """
    return np.array([2, -2]) * pixel_to_uv(pixel, width, height) + np.array([-1, 1])

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

def projection(points: np.ndarray, model_matrix: np.ndarray = None, view_matrix: np.ndarray = None, projection_matrix: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrice)

    Args:
        points (np.ndarray): 3D points of shape (n, 3) or (n, 4), which means (x, y, z) and (x, y, z, 1) are both okay.
            NOTE: shapes of higher dimensions are supported if you handle numpy broadcast properply.
        model_matrix (np.ndarray): row major model to world matrix of shape (4, 4)
        view_matrix (np.ndarray): camera to world matrix of shape (4, 4)
        projection_matrix (np.ndarray): projection matrix of shape (4, 4)

    Returns:
        scr_coord (np.ndarray): vertex screen space coordinates of shape (n, 3), value ranging in [0, 1]. The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        linear_depth (np.ndarray): vertex linear depth of shape
    """
    if model_matrix is None: model_matrix = np.eye(4, dtype=points.dtype)
    if view_matrix is None: view_matrix = np.eye(4, dtype=points.dtype)
    if projection_matrix is None: projection_matrix = np.eye(4, dtype=points.dtype)

    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    clip_coord = points @ np.swapaxes(projection_matrix @ np.linalg.inv(view_matrix) @ model_matrix, -2, -1)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord, linear_depth

def inverse_projection(screen_coord: np.ndarray, linear_depth: np.ndarray, fovX: float, fovY: float) -> np.ndarray:
    """inverse project screen space coordinates to 3d view space 

    Args:
        screen_coord (np.ndarray): screen space coordinates ranging in [0, 1]. Note that the origin (0, 0) of screen space is the left-buttom corner of the screen
        linear_depth (np.ndarray): linear depth values
        fovX (float): x-axis field of view
        fovY (float): y-axis field of view

    Returns:
        points (np.ndarray): 3d points
    """
    ndc_xy = screen_coord * 2 - 1
    clip_coord = np.concatenate([ndc_xy, np.ones_like(ndc_xy[..., :1])], axis=-1) * linear_depth
    points = clip_coord * np.array([np.tan(fovX / 2), np.tan(fovY / 2), -1], dtype=clip_coord.dtype)
    return points

def projection_cv(points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    image_coord = points @ np.swapaxes(intrinsic @ extrinsic[..., :3, :], -2, -1)
    image_coord = points[..., :2] / points[..., 2:]
    linear_depth = points[..., 2]
    return image_coord, linear_depth

def compute_face_normal(vertices: np.ndarray, faces: np.ndarray):
    """Compute face normals of a triangular mesh

    Args:
        vertices (np.ndarray):  3-dimensional vertices of shape (N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)

    Returns:
        normals (np.ndarray): face normals of shape (T, 3)
    """
    normal = np.cross(vertices[faces[..., 1]] - vertices[faces[..., 0]], vertices[faces[..., 2]] - vertices[faces[..., 0]])
    normal = np.nan_to_num(normal / np.sum(normal ** 2, axis=-1, keepdims=True) ** 0.5)
    return normal

def compute_vertex_normal(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute vertex normals of a triangular mesh by averaging neightboring face normals

    Args:
        vertices (np.ndarray): 3-dimensional vertices of shape (N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)

    Returns:
        normals (np.ndarray): vertex normals of shape (N, 3)
    """
    face_normal = compute_face_normal(vertices, faces)
    face_normal = np.repeat(face_normal[..., None, :], 3, -2).reshape((-1, 3))
    face_indices = faces.reshape((-1,))
    vertex_normal = np.zeros_like(vertices)
    while len(face_normal) > 0:
        v_id, f_i = np.unique(face_indices, return_index=True)
        vertex_normal[v_id] += face_normal[f_i]
        face_normal = np.delete(face_normal, f_i, axis=0)
        face_indices = np.delete(face_indices, f_i)
    vertex_normal = np.nan_to_num(vertex_normal / np.sum(vertex_normal ** 2, axis=-1, keepdims=True) ** 0.5)
    return vertex_normal