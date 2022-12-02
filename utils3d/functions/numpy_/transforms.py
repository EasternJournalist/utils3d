import numpy as np
from typing import Tuple


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

def intrinsic_from_fov(fov: float, width: int, height: int) -> np.ndarray:
    normed_int =  np.array([
        [0.5 / (np.tan(fov / 2) * (width / max(width, height))), 0., 0.5],
        [0., 0.5 / (np.tan(fov / 2) * (height / max(width, height))), 0.5],
        [0., 0., 1.],
    ], dtype=np.float32)
    return normed_int * np.array([width, height, 1], dtype=np.float32).reshape(3, 1)

def intrinsic_from_fov_xy(fov_x: float, fov_y: float) -> np.ndarray:
    return np.array([
        [0.5 / np.tan(fov_x / 2), 0., 0.5],
        [0., 0.5 / np.tan(fov_y / 2), 0.5],
        [0., 0., 1.],
    ], dtype=np.float32)

def camera_from_window(view_point: np.ndarray, window_center: np.ndarray, window_x: np.ndarray, window_y: np.ndarray, near: float, far: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get camera view given a window in world space

    Args:
        view_point (np.ndarray): shape (3,)
        window_center (np.ndarray): shape (3,), the center of window in world space
        window_x (np.ndarray): shape (3,), the x axis (right) of window in world space
        window_y (np.ndarray): shape (3,), the y axis (up) of window in world space
        near (float): 
        far (float): 

    Returns:
        view: camera view matrix
        perspective: camera perspective matrix
    """
    x = window_x / np.sum(window_x ** 2, axis=-1, keepdims=True) ** 0.5
    window_y = window_y - x * np.sum(window_y * x, axis=-1, keepdims=True)
    y = window_y / np.sum(window_y ** 2, axis=-1, keepdims=True) ** 0.5
    z = np.cross(window_x, window_y)
    z = z / np.sum(z ** 2, axis=-1, keepdims=True) ** 0.5

    view = np.concatenate([np.stack([x, y, z, view_point], axis=-1), np.array([[0, 0, 0, 1]], dtype=view_point.dtype)], axis=-2)
    
    screen_distance = np.sum((view_point - window_center) * z, axis=-1)
    fx = screen_distance / np.sum(window_x ** 2, axis=-1) ** 0.5
    fy = screen_distance / np.sum(window_y ** 2, axis=-1) ** 0.5
    cx = np.sum((view_point - window_center) * x, axis=-1) / np.sum(window_x ** 2, axis=-1) ** 0.5
    cy = np.sum((view_point - window_center) * y, axis=-1) / np.sum(window_y ** 2, axis=-1) ** 0.5
    a = (near + far) / (near - far)
    b = 2. * near * far / (near - far)
    perspective = np.array([
        [fx, 0,  cx, 0],
        [ 0, fy, cy, 0], 
        [ 0,  0,  a, b],
        [ 0,  0, -1, 0]
    ], dtype=view_point.dtype)

    return view, perspective

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
    one = np.full_like(fx, 1)

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
    return np.diag(np.array([1, -1, -1, 1], dtype=view.dtype)) @ np.linalg.inv(view)

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

def normalize_intrinsic(intrinsic: np.ndarray, width: int, height: int) -> np.ndarray:
    """normalize camera intrinsic
    Args:
        intrinsic (torch.Tensor): shape (..., 3, 3) 
        width (int): image width
        height (int): image height

    Returns:
        (torch.Tensor): shape (..., 3, 3), same as input intrinsic. Normalized intrinsic(s)
    """
    return intrinsic * np.array([1 / width, 1 / height, 1], dtype=intrinsic.dtype)[:, None]

def crop_intrinsic(intrinsic: np.ndarray, width: int, height: int, left: int, top: int, crop_width: int, crop_height: int):
    """Evaluate the new intrinsic(s) after crop the image: cropped_img = img[top:bottom, left:right]

    Args:
        intrinsic (torch.Tensor): shape (3, 3), a normalized camera intrinsic
        width (int): 
        height (int): 
        top (int): 
        left (int): 
        bottom (int): 
        right (int): 
    """
    s = np.array([
        [width / crop_width, 0, -left / crop_width], 
        [0, height / crop_height,  -top / crop_height], 
        [0., 0., 1.]], dtype=intrinsic.dtype)
    return s @ intrinsic

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