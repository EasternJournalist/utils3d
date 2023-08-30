import numpy as np
from typing import *
from ._helpers import batched


@batched(0,0,0,0)
def perspective(
        fov_y: Union[float, np.ndarray],
        aspect: Union[float, np.ndarray],
        near: Union[float, np.ndarray],
        far: Union[float, np.ndarray]
    ) -> np.ndarray:
    """
    Get OpenGL perspective matrix

    Args:
        fov_y (float | np.ndarray): field of view in y axis
        aspect (float | np.ndarray): aspect ratio
        near (float | np.ndarray): near plane to clip
        far (float | np.ndarray): far plane to clip

    Returns:
        (np.ndarray): [..., 4, 4] perspective matrix
    """
    N = fov_y.shape[0]
    ret = np.zeros((N, 4, 4), dtype=fov_y.dtype)
    ret[:, 0, 0] = 1. / (np.tan(fov_y / 2) * aspect)
    ret[:, 1, 1] = 1. / (np.tan(fov_y / 2))
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2. * near * far / (near - far)
    ret[:, 3, 2] = -1.
    return ret


def perspective_from_fov(
        fov: Union[float, np.ndarray],
        width: Union[int, np.ndarray],
        height: Union[int, np.ndarray],
        near: Union[float, np.ndarray],
        far: Union[float, np.ndarray]
    ) -> np.ndarray:
    """
    Get OpenGL perspective matrix from field of view in largest dimension

    Args:
        fov (float | np.ndarray): field of view in largest dimension
        width (int | np.ndarray): image width
        height (int | np.ndarray): image height
        near (float | np.ndarray): near plane to clip
        far (float | np.ndarray): far plane to clip

    Returns:
        (np.ndarray): [..., 4, 4] perspective matrix
    """
    fov_y = 2 * np.arctan(np.tan(fov / 2) * height / np.maximum(width, height))
    aspect = width / height
    return perspective(fov_y, aspect, near, far)


def perspective_from_fov_xy(
        fov_x: Union[float, np.ndarray],
        fov_y: Union[float, np.ndarray],
        near: Union[float, np.ndarray],
        far: Union[float, np.ndarray]
    ) -> np.ndarray:
    """
    Get OpenGL perspective matrix from field of view in x and y axis

    Args:
        fov_x (float | np.ndarray): field of view in x axis
        fov_y (float | np.ndarray): field of view in y axis
        near (float | np.ndarray): near plane to clip
        far (float | np.ndarray): far plane to clip

    Returns:
        (np.ndarray): [..., 4, 4] perspective matrix
    """
    aspect = np.tan(fov_x / 2) / np.tan(fov_y / 2)
    return perspective(fov_y, aspect, near, far)


@batched(0,0,0,0)
def intrinsic(
        focal_x: Union[float, np.ndarray],
        focal_y: Union[float, np.ndarray],
        cx: Union[float, np.ndarray],
        cy: Union[float, np.ndarray]
    ) -> np.ndarray:
    """
    Get OpenCV intrinsic matrix

    Args:
        focal_x (float | np.ndarray): focal length in x axis
        focal_y (float | np.ndarray): focal length in y axis
        cx (float | np.ndarray): principal point in x axis
        cy (float | np.ndarray): principal point in y axis

    Returns:
        (np.ndarray): [..., 3, 3] OpenCV intrinsic matrix
    """
    N = focal_x.shape[0]
    ret = np.zeros((N, 3, 3), dtype=focal_x.dtype)
    ret[:, 0, 0] = focal_x
    ret[:, 1, 1] = focal_y
    ret[:, 0, 2] = cx
    ret[:, 1, 2] = cy
    ret[:, 2, 2] = 1.
    return ret


def intrinsic_from_fov(
        fov: Union[float, np.ndarray],
        width: Union[int, np.ndarray],
        height: Union[int, np.ndarray],
        normalize: bool = False
    ) -> np.ndarray:
    """
    Get OpenCV intrinsic matrix from field of view in largest dimension

    Args:
        fov (float | np.ndarray): field of view in largest dimension
        width (int | np.ndarray): image width
        height (int | np.ndarray): image height
        normalize (bool): whether to normalize the intrinsic to uv space

    Returns:
        (np.ndarray): [..., 3, 3] OpenCV intrinsic matrix
    """
    focal = np.maximum(width, height) / (2 * np.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    ret = intrinsic(focal, focal, cx, cy)
    if normalize:
        ret = normalize_intrinsic(ret, width, height)
    return ret


def intrinsic_from_fov_xy(
        fov_x: Union[float, np.ndarray],
        fov_y: Union[float, np.ndarray]
    ) -> np.ndarray:
    """
    Get OpenCV intrinsic matrix from field of view in x and y axis

    Args:
        fov_x (float | np.ndarray): field of view in x axis
        fov_y (float | np.ndarray): field of view in y axis

    Returns:
        (np.ndarray): [..., 3, 3] OpenCV intrinsic matrix
    """
    focal_x = 0.5 / np.tan(fov_x / 2)
    focal_y = 0.5 / np.tan(fov_y / 2)
    cx = cy = 0.5
    return intrinsic(focal_x, focal_y, cx, cy)


@batched(1,1,1)
def view_look_at(
        eye: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray
    ) -> np.ndarray:
    """
    Get OpenGL view matrix looking at something

    Args:
        eye (np.ndarray): [..., 3] the eye position
        look_at (np.ndarray): [..., 3] the position to look at
        up (np.ndarray): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (np.ndarray): [..., 4, 4], view matrix
    """
    z = eye - look_at
    x = np.cross(up, z)
    y = np.cross(z, x)
    # x = np.cross(y, z)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    R = np.stack([x, y, z], axis=-2)
    t = -np.matmul(R, eye[..., None])
    return np.concatenate([
        np.concatenate([R, t], axis=-1),
        np.array([[[0., 0., 0., 1.]]]).repeat(eye.shape[0], axis=0)
    ], axis=-2)


@batched(1,1,1)
def extrinsic_look_at(
        eye: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray
    ) -> np.ndarray:
    """
    Get OpenCV extrinsic matrix looking at something

    Args:
        eye (np.ndarray): [..., 3] the eye position
        look_at (np.ndarray): [..., 3] the position to look at
        up (np.ndarray): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (np.ndarray): [..., 4, 4], extrinsic matrix
    """
    z = look_at - eye
    x = np.cross(-up, z)
    y = np.cross(z, x)
    # x = np.cross(y, z)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    R = np.stack([x, y, z], axis=-2)
    t = -np.matmul(R, eye[..., None])
    return np.concatenate([
        np.concatenate([R, t], axis=-1),
        np.array([[[0., 0., 0., 1.]]]).repeat(eye.shape[0], axis=0)
    ], axis=-2)


@batched(2)
def perspective_to_intrinsic(
        perspective: np.ndarray
    ) -> np.ndarray:
    """
    OpenGL perspective matrix to OpenCV intrinsic

    Args:
        perspective (np.ndarray): [..., 4, 4] OpenGL perspective matrix

    Returns:
        (np.ndarray): shape [..., 3, 3] OpenCV intrinsic
    """
    N = perspective.shape[0]
    fx, fy = perspective[:, 0, 0], perspective[:, 1, 1]
    cx, cy = perspective[:, 0, 2], perspective[:, 1, 2]
    ret = np.zeros((N, 3, 3), dtype=perspective.dtype)
    ret[:, 0, 0] = 0.5 * fx
    ret[:, 1, 1] = 0.5 * fy
    ret[:, 0, 2] = -0.5 * cx + 0.5
    ret[:, 1, 2] = 0.5 * cy + 0.5
    ret[:, 2, 2] = 1.
    return ret


@batched(2,0,0)
def intrinsic_to_perspective(
        intrinsic: np.ndarray,
        near: Union[float, np.ndarray],
        far: Union[float, np.ndarray],
    ) -> np.ndarray:
    """
    OpenCV intrinsic to OpenGL perspective matrix

    Args:
        intrinsic (np.ndarray): [..., 3, 3] OpenCV intrinsic matrix
        near (float | np.ndarray): [...] near plane to clip
        far (float | np.ndarray): [...] far plane to clip
    Returns:
        (np.ndarray): [..., 4, 4] OpenGL perspective matrix
    """
    N = intrinsic.shape[0]
    fx, fy = intrinsic[:, 0, 0], intrinsic[:, 1, 1]
    cx, cy = intrinsic[:, 0, 2], intrinsic[:, 1, 2]
    ret = np.zeros((N, 4, 4), dtype=intrinsic.dtype)
    ret[:, 0, 0] = 2 * fx
    ret[:, 1, 1] = 2 * fy
    ret[:, 0, 2] = -2 * cx + 1
    ret[:, 1, 2] = 2 * cy - 1
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2. * near * far / (near - far)
    ret[:, 3, 2] = -1.
    return ret


@batched(2)
def extrinsic_to_view(
        extrinsic: np.ndarray
    ) -> np.ndarray:
    """
    OpenCV camera extrinsic to OpenGL view matrix

    Args:
        extrinsic (np.ndarray): [..., 4, 4] OpenCV camera extrinsic matrix

    Returns:
        (np.ndarray): [..., 4, 4] OpenGL view matrix
    """
    return extrinsic * np.array([1, -1, -1, 1], dtype=extrinsic.dtype)[:, None]


@batched(2)
def view_to_extrinsic(
        view: np.ndarray
    ) -> np.ndarray:
    """
    OpenGL view matrix to OpenCV camera extrinsic

    Args:
        view (np.ndarray): [..., 4, 4] OpenGL view matrix

    Returns:
        (np.ndarray): [..., 4, 4] OpenCV camera extrinsic matrix
    """
    return view * np.array([1, -1, -1, 1], dtype=view.dtype)[:, None]


@batched(2,0,0)
def normalize_intrinsic(
        intrinsic: np.ndarray,
        width: Union[int, np.ndarray],
        height: Union[int, np.ndarray]
    ) -> np.ndarray:
    """
    Normalize camera intrinsic(s) to uv space

    Args:
        intrinsic (np.ndarray): [..., 3, 3] camera intrinsic(s) to normalize
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 3, 3] normalized camera intrinsic(s)
    """
    return intrinsic * np.stack([1 / width, 1 / height, np.ones_like(width)], axis=-1).astype(intrinsic.dtype)[..., None]


@batched(2,0,0,0,0,0,0)
def crop_intrinsic(
        intrinsic: np.ndarray,
        width: Union[int, np.ndarray],
        height: Union[int, np.ndarray],
        left: Union[int, np.ndarray],
        top: Union[int, np.ndarray],
        crop_width: Union[int, np.ndarray],
        crop_height: Union[int, np.ndarray]
    ) -> np.ndarray:
    """
    Evaluate the new intrinsic(s) after crop the image: cropped_img = img[top:top+crop_height, left:left+crop_width]

    Args:
        intrinsic (np.ndarray): [..., 3, 3] camera intrinsic(s) to crop
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)
        left (int | np.ndarray): [...] left crop boundary
        top (int | np.ndarray): [...] top crop boundary
        crop_width (int | np.ndarray): [...] crop width
        crop_height (int | np.ndarray): [...] crop height

    Returns:
        (np.ndarray): [..., 3, 3] cropped camera intrinsic(s)
    """
    intrinsic = intrinsic.copy()
    intrinsic[..., 0, 0] *= width / crop_width
    intrinsic[..., 1, 1] *= height / crop_height
    intrinsic[..., 0, 2] = (intrinsic[..., 0, 2] * width - left) / crop_width
    intrinsic[..., 1, 2] = (intrinsic[..., 1, 2] * height - top) / crop_height
    return intrinsic


@batched(1,0,0)
def pixel_to_uv(
        pixel: np.ndarray,
        width: Union[int, np.ndarray],
        height: Union[int, np.ndarray]
    ) -> np.ndarray:
    """
    Args:
        pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    uv = np.zeros(pixel.shape, dtype=np.float32)
    uv[..., 0] = (pixel[..., 0] + 0.5) / width
    uv[..., 1] = (pixel[..., 1] + 0.5) / height
    return uv


@batched(1,0,0)
def pixel_to_ndc(
        pixel: np.ndarray,
        width: Union[int, np.ndarray],
        height: Union[int, np.ndarray]
    ) -> np.ndarray:
    """
    Args:
        pixel (np.ndarray): [..., 2] pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)

    Returns:
        (np.ndarray): [..., 2] pixel coordinrates defined in ndc space, the range is (-1, 1)
    """
    ndc = np.zeros(pixel.shape, dtype=np.float32)
    ndc[..., 0] = (pixel[..., 0] + 0.5) / width * 2 - 1
    ndc[..., 1] = -((pixel[..., 1] + 0.5) / height * 2 - 1)
    return ndc


@batched(0,0,0)
def project_depth(
        depth: np.ndarray,
        near: Union[float, np.ndarray],
        far: Union[float, np.ndarray]
    ) -> np.ndarray:
    """
    Project linear depth to depth value in screen space

    Args:
        depth (np.ndarray): [...] depth value
        near (float | np.ndarray): [...] near plane to clip
        far (float | np.ndarray): [...] far plane to clip

    Returns:
        (np.ndarray): [..., 1] depth value in screen space, value ranging in [0, 1]
    """
    return (far - near * far / depth) / (far - near)


@batched(0,0,0)
def linearize_depth(
        depth: np.ndarray,
        near: Union[float, np.ndarray],
        far: Union[float, np.ndarray]
    ) -> np.ndarray:
    """
    Linearize depth value to linear depth

    Args:
        depth (np.ndarray): [...] depth value
        near (float | np.ndarray): [...] near plane to clip
        far (float | np.ndarray): [...] far plane to clip

    Returns:
        (np.ndarray): [..., 1] linear depth
    """
    return near * far / (far - (far - near) * depth)


@batched(1,2,2,2)
def project_gl(
        points: np.ndarray,
        model: np.ndarray = None,
        view: np.ndarray = None,
        perspective: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D following the OpenGL convention (except for row major matrice)

    Args:
        points (np.ndarray): [..., 3] or [..., 4] 3D points to project, if the last 
            dimension is 4, the points are assumed to be in homogeneous coordinates
        model (np.ndarray): [..., 4, 4] model matrix
        view (np.ndarray): [..., 4, 4] view matrix
        perspective (np.ndarray): [..., 4, 4] perspective matrix

    Returns:
        scr_coord (np.ndarray): [..., 3] screen space coordinates, value ranging in [0, 1].
            The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        linear_depth (np.ndarray): [...] linear depth
    """
    assert perspective is not None, "perspective matrix is required"
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points = points[..., None]
    if model is not None:
        points = model @ points
    if view is not None:
        points = view @ points
    clip_coord = (perspective @ points)[..., 0]
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord, linear_depth


@batched(1,2,2)
def project_cv(
        points: np.ndarray,
        extrinsic: np.ndarray = None,
        intrinsic: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D following the OpenCV convention

    Args:
        points (np.ndarray): [..., 3] or [..., 4] 3D points to project, if the last
            dimension is 4, the points are assumed to be in homogeneous coordinates
        extrinsic (np.ndarray): [..., 4, 4] extrinsic matrix
        intrinsic (np.ndarray): [..., 3, 3] intrinsic matrix

    Returns:
        uv_coord (np.ndarray): [..., 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (np.ndarray): [...] linear depth
    """
    assert intrinsic is not None, "intrinsic matrix is required"
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points = points[..., None]
    if extrinsic is not None:
        points = extrinsic @ points
    points = (intrinsic @ points[..., :3, :])[..., 0]
    uv_coord = points[..., :2] / points[..., 2:]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth


@batched(1,2,2,2)
def unproject_gl(
        screen_coord: np.ndarray,
        model: np.ndarray = None,
        view: np.ndarray = None,
        perspective: np.ndarray = None
    ) -> np.ndarray:
    """
    Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrice)

    Args:
        screen_coord (np.ndarray): [..., 3] screen space coordinates, value ranging in [0, 1].
            The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        model (np.ndarray): [..., 4, 4] model matrix
        view (np.ndarray): [..., 4, 4] view matrix
        perspective (np.ndarray): [..., 4, 4] perspective matrix

    Returns:
        points (np.ndarray): [..., 3] 3d points
    """
    assert perspective is not None, "perspective matrix is required"
    ndc_xy = screen_coord * 2 - 1
    clip_coord = np.concatenate([ndc_xy, np.ones_like(ndc_xy[..., :1])], axis=-1)
    transform = perspective
    if view is not None:
        transform = transform @ view
    if model is not None:
        transform = transform @ model
    transform = np.linalg.inv(transform)
    points = (transform @ clip_coord[..., None])[..., 0]
    points = points[..., :3] / points[..., 3:]
    return points
    

@batched(1,0,2,2)
def unproject_cv(
        uv_coord: np.ndarray,
        depth: np.ndarray,
        extrinsic: np.ndarray = None,
        intrinsic: np.ndarray = None
    ) -> np.ndarray:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    Args:
        uv_coord (np.ndarray): [..., 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (np.ndarray): [...] depth value
        extrinsic (np.ndarray): [..., 4, 4] extrinsic matrix
        intrinsic (np.ndarray): [..., 3, 3] intrinsic matrix

    Returns:
        points (np.ndarray): [..., 3] 3d points
    """
    assert intrinsic is not None, "intrinsic matrix is required"
    points = np.concatenate([uv_coord, np.ones_like(uv_coord[..., :1])], axis=-1)
    points = (np.linalg.inv(intrinsic) @ points[..., None])[..., 0]
    points = points * depth[..., None]
    if extrinsic is not None:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        points = (np.linalg.inv(extrinsic) @ points[..., None])[:, :3, 0]
    return points
