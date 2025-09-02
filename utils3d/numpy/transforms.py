import numpy as np
from numpy import ndarray
from typing import *
from numbers import Number
from ._helpers import toarray, batched
from .._helpers import no_warnings


__all__ = [
    'perspective_from_fov',
    'perspective_from_window',
    'intrinsics_from_fov',
    'intrinsics_from_focal_center',
    'fov_to_focal',
    'focal_to_fov',
    'intrinsics_to_fov',
    'view_look_at',
    'extrinsics_look_at',
    'perspective_to_intrinsics',
    'perspective_to_near_far',
    'intrinsics_to_perspective',
    'extrinsics_to_view',
    'view_to_extrinsics',
    'normalize_intrinsics',
    'crop_intrinsics',
    'pixel_to_uv',
    'pixel_to_ndc',
    'uv_to_pixel',
    'depth_linear_to_buffer',
    'depth_buffer_to_linear',
    'unproject_cv',
    'unproject_gl',
    'project_cv',
    'project_gl',
    'project',
    'unproject',
    'screen_coord_to_view_coord',
    'quaternion_to_matrix',
    'axis_angle_to_matrix',
    'matrix_to_quaternion',
    'extrinsics_to_essential',
    'euler_axis_angle_rotation',
    'euler_angles_to_matrix',
    'skew_symmetric',
    'rotation_matrix_from_vectors',
    'ray_intersection',
    'make_se3_matrix',
    'slerp_quaternion',
    'slerp_vector',
    'lerp',
    'lerp_se3_matrix',
    'piecewise_lerp',
    'piecewise_lerp_se3_matrix',
    'transform',
    'angle_between'
]


@toarray(_others=np.float32)
@batched(_others=0)
def perspective_from_fov(
    *,
    fov_x: Optional[Union[float, ndarray]] = None,
    fov_y: Optional[Union[float, ndarray]] = None,
    fov_min: Optional[Union[float, ndarray]] = None,
    fov_max: Optional[Union[float, ndarray]] = None,
    aspect_ratio: Optional[Union[float, ndarray]] = None,
    near: Optional[Union[float, ndarray]],
    far: Optional[Union[float, ndarray]],
) -> ndarray:
    """
    Get OpenGL perspective matrix from field of view 

    ## Returns
        (ndarray): [..., 4, 4] perspective matrix
    """
    if fov_max is not None:
        fx = np.maximum(1, 1 / aspect_ratio) / np.tan(fov_max / 2)
        fy = np.maximum(1, aspect_ratio) / np.tan(fov_max / 2)
    elif fov_min is not None:
        fx = np.minimum(1, 1 / aspect_ratio) / np.tan(fov_min / 2)
        fy = np.minimum(1, aspect_ratio) / np.tan(fov_min / 2)
    elif fov_x is not None and fov_y is not None:
        fx = 1 / np.tan(fov_x / 2)
        fy = 1 / np.tan(fov_y / 2)
    elif fov_x is not None:
        fx = 1 / np.tan(fov_x / 2)
        fy = fx * aspect_ratio
    elif fov_y is not None:
        fy = 1 / np.tan(fov_y / 2)
        fx = fy / aspect_ratio
    perspective = np.zeros((fx.shape[0], 4, 4), dtype=fx.dtype)
    perspective[:, 0, 0] = fx
    perspective[:, 1, 1] = fy
    perspective[:, 2, 2] = (near / far + 1) / (near / far - 1)
    perspective[:, 2, 3] = 2. * near / (near / far - 1)
    perspective[:, 3, 2] = -1.
    return perspective


@toarray(_others=np.float32)
@batched(_others=0)
def perspective_from_window(
    left: Union[float, ndarray],
    right: Union[float, ndarray],
    bottom: Union[float, ndarray],
    top: Union[float, ndarray],
    near: Union[float, ndarray],
    far: Union[float, ndarray]
) -> ndarray:
    """
    Get OpenGL perspective matrix from the window of z=-1 projection plane

    ## Returns
        (ndarray): [..., 4, 4] perspective matrix
    """
    perspective = np.zeros((left.shape[0], 4, 4), dtype=left.dtype)
    perspective[:, 0, 0] = 2 / (right - left)
    perspective[:, 0, 2] = (right + left) / (right - left)
    perspective[:, 1, 1] = 2 / (top - bottom)
    perspective[:, 1, 2] = (top + bottom) / (top - bottom)
    perspective[:, 2, 2] = (near / far + 1) / (near / far - 1)
    perspective[:, 2, 3] = 2. * near / (near / far - 1)
    perspective[:, 3, 2] = -1.
    return perspective


@toarray(_others=np.float32)
@batched(_others=0)
def intrinsics_from_focal_center(
    fx: Union[float, ndarray],
    fy: Union[float, ndarray],
    cx: Union[float, ndarray],
    cy: Union[float, ndarray],
) -> ndarray:
    """
    Get OpenCV intrinsics matrix

    ## Returns
        (ndarray): [..., 3, 3] OpenCV intrinsics matrix
    """
    if any(isinstance(x, ndarray) for x in (fx, fy, cx, cy)):
        dtype = np.result_type(fx, fy, cx, cy)
    fx, fy, cx, cy = np.broadcast_arrays(fx, fy, cx, cy)
    ret = np.zeros((*fx.shape, 3, 3), dtype=dtype)
    ret[..., 0, 0] = fx
    ret[..., 1, 1] = fy
    ret[..., 0, 2] = cx
    ret[..., 1, 2] = cy
    ret[..., 2, 2] = 1.
    return ret


@toarray(_others=np.float32)
@batched(_others=0)
def intrinsics_from_fov(
    fov_x: Optional[Union[float, ndarray]] = None,
    fov_y: Optional[Union[float, ndarray]] = None,
    fov_max: Optional[Union[float, ndarray]] = None,
    fov_min: Optional[Union[float, ndarray]] = None,
    aspect_ratio: Optional[Union[float, ndarray]] = None,
) -> ndarray:
    """
    Get normalized OpenCV intrinsics matrix from given field of view.
    You can provide either fov_x, fov_y, fov_max or fov_min and aspect_ratio

    ## Parameters
        fov_x (float | ndarray): field of view in x axis
        fov_y (float | ndarray): field of view in y axis
        fov_max (float | ndarray): field of view in largest dimension
        fov_min (float | ndarray): field of view in smallest dimension
        aspect_ratio (float | ndarray): aspect ratio of the image

    ## Returns
        (ndarray): [..., 3, 3] OpenCV intrinsics matrix
    """
    if fov_max is not None:
        fx = np.maximum(1, 1 / aspect_ratio) / (2 * np.tan(fov_max / 2))
        fy = np.maximum(1, aspect_ratio) / (2 * np.tan(fov_max / 2))
    elif fov_min is not None:
        fx = np.minimum(1, 1 / aspect_ratio) / (2 * np.tan(fov_min / 2))
        fy = np.minimum(1, aspect_ratio) / (2 * np.tan(fov_min / 2))
    elif fov_x is not None and fov_y is not None:
        fx = 1 / (2 * np.tan(fov_x / 2))
        fy = 1 / (2 * np.tan(fov_y / 2))
    elif fov_x is not None:
        fx = 1 / (2 * np.tan(fov_x / 2))
        fy = fx * aspect_ratio
    elif fov_y is not None:
        fy = 1 / (2 * np.tan(fov_y / 2))
        fx = fy / aspect_ratio
    cx = 0.5
    cy = 0.5
    ret = intrinsics_from_focal_center(fx, fy, cx, cy)
    return ret


def focal_to_fov(focal: ndarray):
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal(fov: ndarray):
    return 0.5 / np.tan(fov / 2)


def intrinsics_to_fov(intrinsics: ndarray) -> Tuple[ndarray, ndarray]:
    fov_x = focal_to_fov(intrinsics[..., 0, 0])
    fov_y = focal_to_fov(intrinsics[..., 1, 1])
    return fov_x, fov_y


@toarray(_others=np.float32)
@batched(_others=1)
def view_look_at(
    eye: ndarray,
    look_at: ndarray,
    up: ndarray
) -> ndarray:
    """
    Get OpenGL view matrix looking at something

    ## Parameters
        eye (ndarray): [..., 3] the eye position
        look_at (ndarray): [..., 3] the position to look at
        up (ndarray): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (ndarray): [..., 4, 4], view matrix
    """
    z = eye - look_at
    x = np.cross(up, z)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-2)
    R = R / np.linalg.norm(R, axis=-1, keepdims=True)
    t = (-R @ eye[..., None]).squeeze(-1)
    return make_se3_matrix(R, t)


@toarray(_others=np.float32)
@batched(_others=1)
def extrinsics_look_at(
    eye: ndarray,
    look_at: ndarray,
    up: ndarray
) -> ndarray:
    """
    Get OpenCV extrinsics matrix looking at something

    ## Parameters
        eye (ndarray): [..., 3] the eye position
        look_at (ndarray): [..., 3] the position to look at
        up (ndarray): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (ndarray): [..., 4, 4], extrinsics matrix
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
        np.array([[[0., 0., 0., 1.]]], dtype=eye.dtype).repeat(eye.shape[0], axis=0)
    ], axis=-2)


def perspective_to_intrinsics(perspective: ndarray) -> ndarray:
    """
    OpenGL perspective matrix to OpenCV intrinsics

    ## Parameters
        perspective (ndarray): [..., 4, 4] OpenGL perspective matrix

    ## Returns
        (ndarray): shape [..., 3, 3] OpenCV intrinsics
    """
    assert np.allclose(perspective[:, [0, 1, 3], 3], 0), "The matrix is not a perspective projection matrix"
    ret = np.array([[0.5, 0., 0.5], [0., -0.5, 0.5], [0., 0., 1.]], dtype=perspective.dtype) \
        @ perspective[..., [0, 1, 3], :3] \
        @ np.diag(np.array([1, -1, -1], dtype=perspective.dtype))
    return ret


def perspective_to_near_far(perspective: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Get near and far planes from OpenGL perspective matrix

    ## Parameters
    """
    a, b = perspective[..., 2, 2], perspective[..., 2, 3]
    near, far =  b / (a - 1), b / (a + 1)
    return near, far


@toarray(None, _others='intrinsics')
@batched(2, 0, 0)
def intrinsics_to_perspective(
    intrinsics: ndarray,
    near: Union[float, ndarray],
    far: Union[float, ndarray],
) -> ndarray:
    """
    OpenCV intrinsics to OpenGL perspective matrix
    NOTE: not work for tile-shifting intrinsics currently

    ## Parameters
        intrinsics (ndarray): [..., 3, 3] OpenCV intrinsics matrix
        near (float | ndarray): [...] near plane to clip
        far (float | ndarray): [...] far plane to clip
    ## Returns
        (ndarray): [..., 4, 4] OpenGL perspective matrix
    """
    perspective = np.zeros((intrinsics.shape[0], 4, 4), dtype=intrinsics.dtype)
    perspective[..., [0, 1, 3], :3] = np.array([[2, 0, -1], [0, -2, 1], [0, 0, 1]], dtype=intrinsics.dtype) \
        @ intrinsics \
        @ np.diagonal(np.array([1, -1, -1], dtype=intrinsics.dtype))
    perspective[:, 2, 2] = (near / far + 1) / (near / far - 1)
    perspective[:, 2, 3] = 2. * near / (near / far - 1)
    perspective[:, 3, 2] = -1.
    return perspective


def extrinsics_to_view(extrinsics: ndarray) -> ndarray:
    """
    OpenCV camera extrinsics to OpenGL view matrix

    ## Parameters
        extrinsics (ndarray): [..., 4, 4] OpenCV camera extrinsics matrix

    ## Returns
        (ndarray): [..., 4, 4] OpenGL view matrix
    """
    return extrinsics * np.array([1, -1, -1, 1], dtype=extrinsics.dtype)[:, None]


def view_to_extrinsics(view: ndarray) -> ndarray:
    """
    OpenGL view matrix to OpenCV camera extrinsics

    ## Parameters
        view (ndarray): [..., 4, 4] OpenGL view matrix

    ## Returns
        (ndarray): [..., 4, 4] OpenCV camera extrinsics matrix
    """
    return view * np.array([1, -1, -1, 1], dtype=view.dtype)[:, None]


@toarray(None, 'intrinsics', 'intrinsics', None)
@batched(2, 0, 0, None)
def normalize_intrinsics(
    intrinsics: ndarray,
    width: Union[Number, ndarray],
    height: Union[Number, ndarray],
    integer_pixel_centers: bool = True
) -> ndarray:
    """
    Normalize intrinsics from pixel cooridnates to uv coordinates

    ## Parameters
        intrinsics (ndarray): [..., 3, 3] camera intrinsics(s) to normalize
        width (int | ndarray): [...] image width(s)
        height (int | ndarray): [...] image height(s)
        integer_pixel_centers (bool): whether the integer pixel coordinates are at the center of the pixel. If False, the integer coordinates are at the left-top corner of the pixel.

    ## Returns
        (ndarray): [..., 3, 3] normalized camera intrinsics(s)
    """
    zeros = np.zeros_like(width)
    ones = np.ones_like(width)
    if integer_pixel_centers:
        transform = np.stack([
            1 / width, zeros, 0.5 / width,
            zeros, 1 / height, 0.5 / height,
            zeros, zeros, ones
        ]).reshape(*zeros.shape, 3, 3)
    else:
        transform = np.stack([
            1 / width, zeros, zeros,
            zeros, 1 / height, zeros,
            zeros, zeros, ones
        ]).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


@toarray(None, _others='intrinsics')
@batched(2, _others=0)
def crop_intrinsics(
    intrinsics: ndarray,
    width: Union[Number, ndarray],
    height: Union[Number, ndarray],
    left: Union[Number, ndarray],
    top: Union[Number, ndarray],
    crop_width: Union[Number, ndarray],
    crop_height: Union[Number, ndarray]
) -> ndarray:
    """
    Evaluate the new intrinsics(s) after crop the image: cropped_img = img[top:top+crop_height, left:left+crop_width]

    ## Parameters
        intrinsics (ndarray): [..., 3, 3] camera intrinsics(s) to crop
        width (int | ndarray): [...] image width(s)
        height (int | ndarray): [...] image height(s)
        left (int | ndarray): [...] left crop boundary
        top (int | ndarray): [...] top crop boundary
        crop_width (int | ndarray): [...] crop width
        crop_height (int | ndarray): [...] crop height

    ## Returns
        (ndarray): [..., 3, 3] cropped camera intrinsics(s)
    """
    zeros = np.zeros_like(width)
    ones = np.ones_like(width)
    transform = np.stack([
        width / crop_width, zeros, -left / crop_width,
        zeros, height / crop_height, -top / crop_height,
        zeros, zeros, ones
    ]).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


@batched(1, 0, 0)
def pixel_to_uv(
    pixel: ndarray,
    width: Union[Number, ndarray],
    height: Union[Number, ndarray]
) -> ndarray:
    """
    ## Parameters
        pixel (ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (Number | ndarray): [...] image width(s)
        height (Number | ndarray): [...] image height(s)

    ## Returns
        (ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    if not np.issubdtype(pixel.dtype, np.floating):
        pixel = pixel.astype(np.float32)
    uv = (pixel + 0.5) / np.stack([width, height], axis=-1)
    return uv


@batched(1, 0, 0)
def uv_to_pixel(
    uv: ndarray,
    width: Union[int, ndarray],
    height: Union[int, ndarray]
) -> ndarray:
    """
    ## Parameters
        pixel (ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | ndarray): [...] image width(s)
        height (int | ndarray): [...] image height(s)

    ## Returns
        (ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    pixel = uv * np.stack([width, height], axis=-1).astype(uv.dtype) - 0.5
    return pixel


@batched(1,0,0)
def pixel_to_ndc(
    pixel: ndarray,
    width: Union[int, ndarray],
    height: Union[int, ndarray]
) -> ndarray:
    """
    ## Parameters
        pixel (ndarray): [..., 2] pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
        width (int | ndarray): [...] image width(s)
        height (int | ndarray): [...] image height(s)

    ## Returns
        (ndarray): [..., 2] pixel coordinrates defined in ndc space, the range is (-1, 1)
    """
    if not np.issubdtype(pixel.dtype, np.floating):
        pixel = pixel.astype(np.float32)
    dtype = pixel.dtype
    ndc = (pixel + np.array(0.5, dtype=dtype)) / (np.stack([width, height], dim=-1) * np.array([2, -2], dtype=dtype)) \
        + np.array([-1, 1], dtype=dtype)
    return ndc


@batched(0, 0, 0)
def depth_linear_to_buffer(
    depth: ndarray,
    near: Union[float, ndarray],
    far: Union[float, ndarray]
) -> ndarray:
    """
    Project linear depth to depth value in screen space

    ## Parameters
        depth (ndarray): [...] depth value
        near (float | ndarray): [...] near plane to clip
        far (float | ndarray): [...] far plane to clip

    ## Returns
        (ndarray): [..., 1] depth value in screen space, value ranging in [0, 1]
    """
    return (1 - near / depth) / (1 - near / far)


@batched(0, 0, 0)
def depth_buffer_to_linear(
    depth_buffer: ndarray,
    near: Union[float, ndarray],
    far: Union[float, ndarray]
) -> ndarray:
    """
    OpenGL depth buffer to linear depth

    ## Parameters
        depth_buffer (ndarray): [...] depth value
        near (float | ndarray): [...] near plane to clip
        far (float | ndarray): [...] far plane to clip

    ## Returns
        (ndarray): [..., 1] linear depth
    """
    return near / (1 - (1 - near / far) * depth_buffer)


def project_gl(
    points: ndarray,
    projection: ndarray,
    view: ndarray = None,
) -> Tuple[ndarray, ndarray]:
    """
    Project 3D points to 2D following the OpenGL convention (except for row major matrice)

    ## Parameters
        points (ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last 
            dimension is 4, the points are assumed to be in homogeneous coordinates
        view (ndarray): [..., 4, 4] view matrix
        projection (ndarray): [..., 4, 4] projection matrix

    ## Returns
        scr_coord (ndarray): [..., N, 2] OpenGL screen space XY coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & bottom
        linear_depth (ndarray): [..., N] linear depth
    """
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)], axis=-1)
    transform = projection @ view if view is not None else projection
    clip_coord = points @ transform.mT
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord[..., :2], linear_depth


@no_warnings()
def project_cv(
    points: ndarray,
    intrinsics: ndarray,
    extrinsics: Optional[ndarray] = None,
) -> Tuple[ndarray, ndarray]:
    """
    Project 3D points to 2D following the OpenCV convention

    ## Parameters
        points (ndarray): [..., N, 3]
        extrinsics (ndarray): [..., 4, 4] extrinsics matrix
        intrinsics (ndarray): [..., 3, 3] intrinsics matrix

    ## Returns
        uv_coord (ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (ndarray): [..., N] linear depth
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    points = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)], axis=-1)
    intrinsics = np.block([
        [intrinsics, np.zeros((*intrinsics.shape[:-2], 1, 3), dtype=intrinsics.dtype)],
        np.broadcast_to(np.array([0, 0, 0, 1], dtype=intrinsics.dtype), (*intrinsics.shape[:-2], 1, 4))
    ])
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = points @ transform.mT
    uv_coord = points[..., :2] / points[..., 2:]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth


def unproject_gl(
    uv: ndarray,
    depth: ndarray,
    projection: ndarray,
    view: Optional[ndarray] = None,
) -> ndarray:
    """
    Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrice)

    ## Parameters
        uv (ndarray): (..., N, 2) screen space XY coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & bottom
        depth (ndarray): (..., N) linear depth values
        projection (ndarray): (..., 4, 4) projection  matrix
        view (ndarray): (..., 4, 4) view matrix
        
    ## Returns
        points (ndarray): (..., N, 3) 3d points
    """
    ndc_xy = uv * 2 - 1
    view_z = -depth
    clip_xy = np.linalg.inv(projection[..., :2, :2] - ndc_xy[..., :, None] * projection[..., 3:, :2]) \
        @ ((ndc_xy[..., :, None] * projection[..., 3:, 2:] - projection[..., :2, 2:]) \
        @ np.concatenate([view_z[..., None, None], np.ones_like(view_z[..., None, None])], axis=-2))
    points = np.concatenate([clip_xy.squeeze(-1), view_z[..., None], np.ones_like(view_z)[..., None]], axis=-1)
    if view is not None:
        points = points @ np.linalg.inv(view).mT
    return points[..., :3]


@batched(2, 1, 2, 2)
def screen_coord_to_view_coord(
    screen_coord: ndarray,
    projection: ndarray,
) -> ndarray:
    """
    Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrice)

    ## Parameters
        screen_coord (ndarray): (..., N, 3) screen space XYZ coordinates, value ranging in [0, 1]
            The origin (0., 0.) is corresponding to the left & bottom
        projection (ndarray): (..., 4, 4) projection matrix

    ## Returns
        points (ndarray): [..., N, 3] 3d points
    """
    assert projection is not None, "projection matrix is required"
    ndc_xy = screen_coord * 2 - 1
    clip_coord = np.concatenate([ndc_xy, np.ones_like(ndc_xy[..., :1])], axis=-1)
    points = clip_coord @ np.linalg.inv(projection).swapaxes(-1, -2)
    points = points[..., :3] / points[..., 3:]
    return points


@batched(2, 1, 2, 2)
def unproject_cv(
    uv: ndarray,
    depth: Optional[ndarray],
    intrinsics: ndarray,
    extrinsics: Optional[ndarray] = None,
) -> ndarray:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    ## Parameters
        uv_coord (ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (ndarray): [..., N] depth value
        extrinsics (ndarray): [..., 4, 4] extrinsics matrix
        intrinsics (ndarray): [..., 3, 3] intrinsics matrix

    ## Returns
        points (ndarray): [..., N, 3] 3d points
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    points = np.concatenate([uv, np.ones_like(uv[..., :1])], axis=-1)
    points = points @ np.linalg.inv(intrinsics).swapaxes(-1, -2) 
    points = points * depth[..., None]
    if extrinsics is not None:
        points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        points = (points @ np.linalg.inv(extrinsics).swapaxes(-1, -2))[..., :3]
    return points


def project(
    points: ndarray,
    *,
    intrinsics: Optional[ndarray] = None,
    extrinsics: Optional[ndarray] = None,
    view: Optional[ndarray] = None,
    projection: Optional[ndarray] = None
) -> Tuple[ndarray, ndarray]:
    """
    Calculate projection. 
    - For OpenCV convention, use `intrinsics` and `extrinsics` matrice. 
    - For OpenGL convention, use `view` and `projection` matrice.

    ## Parameters

    - `points`: (..., N, 3) 3D world-space points
    - `intrinsics`: (..., 3, 3) intrinsics matrix
    - `extrinsics`: (..., 4, 4) extrinsics matrix
    - `view`: (..., 4, 4) view matrix
    - `projection`: (..., 4, 4) projection matrix

    ## Returns

    - `uv`: (..., N, 2) 2D coordinates. 
        - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
        - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
    - `depth`: (..., N) linear depth values, where `depth > 0` is visible.
        - For OpenCV convention, it is the Z coordinate in camera space.
        - For OpenGL convention, it is the -Z coordinate in camera space.
    """
    assert (intrinsics is not None or extrinsics is not None) ^ (view is not None or projection is not None), \
        "Either camera intrinsics (and extrinsics) or projection (and view) matrices must be provided."
    
    if intrinsics is not None:
        return project_cv(points, intrinsics, extrinsics)
    elif projection is not None:
        return project_gl(points, projection, view)
    else:
        raise ValueError("Invalid combination of input parameters.")


def unproject(
    uv: ndarray,
    depth: Optional[ndarray],
    *,
    intrinsics: Optional[ndarray] = None,
    extrinsics: Optional[ndarray] = None,
    projection: Optional[ndarray] = None,
    view: Optional[ndarray] = None,
) -> ndarray:
    """
    Calculate inverse projection. 
    - For OpenCV convention, use `intrinsics` and `extrinsics` matrice. 
    - For OpenGL convention, use `view` and `projection` matrice.

    ## Parameters

    - `uv`: (..., N, 2) 2D coordinates. 
        - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
        - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
    - `depth`: (..., N) linear depth values, where `depth > 0` is visible.
        - For OpenCV convention, it is the Z coordinate in camera space.
        - For OpenGL convention, it is the -Z coordinate in camera space.
    - `intrinsics`: (..., 3, 3) intrinsics matrix
    - `extrinsics`: (..., 4, 4) extrinsics matrix
    - `view`: (..., 4, 4) view matrix
    - `projection`: (..., 4, 4) projection matrix

    ## Returns

    - `points`: (..., N, 3) 3D world-space points
    """
    assert (intrinsics is not None or extrinsics is not None) ^ (view is not None or projection is not None), \
        "Either camera intrinsics (and extrinsics) or projection (and view) matrices must be provided."

    if intrinsics is not None:
        return unproject_cv(uv, depth, intrinsics, extrinsics)
    elif projection is not None:
        return unproject_gl(uv, depth, projection, view)
    else:
        raise ValueError("Invalid combination of input parameters.")


def quaternion_to_matrix(quaternion: ndarray, eps: float = 1e-12) -> ndarray:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices
    
    ## Parameters
        quaternion (ndarray): shape (..., 4), the quaternions to convert
    
    ## Returns
        ndarray: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=eps)
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    zeros = np.zeros_like(w)
    I = np.eye(3, dtype=quaternion.dtype)
    xyz = quaternion[..., 1:]
    A = xyz[..., :, None] * xyz[..., None, :] - I * (xyz ** 2).sum(axis=-1)[..., None, None]
    B = np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], axis=-1).reshape(*quaternion.shape[:-1], 3, 3)
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def matrix_to_quaternion(rot_mat: ndarray, eps: float = 1e-12) -> ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

    ## Parameters
        rot_mat (ndarray): shape (..., 3, 3), the rotation matrices to convert

    ## Returns
        ndarray: shape (..., 4), the quaternions corresponding to the given rotation matrices
    """
    # Extract the diagonal and off-diagonal elements of the rotation matrix
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [rot_mat[..., i, j] for i in range(3) for j in range(3)]

    diag = np.diagonal(rot_mat, axis1=-2, axis2=-1)
    M = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=rot_mat.dtype)
    wxyz = 0.5 * np.clip(1 + diag @ M.T, 0.0, None) ** 0.5
    max_idx = np.argmax(wxyz, axis=-1)
    xw = np.sign(m21 - m12)
    yw = np.sign(m02 - m20)
    zw = np.sign(m10 - m01)
    yz = np.sign(m21 + m12)
    xz = np.sign(m02 + m20)
    xy = np.sign(m01 + m10)
    ones = np.ones_like(xw)
    sign = np.where(
        max_idx[..., None] == 0,
        np.stack([ones, xw, yw, zw], axis=-1),
        np.where(
            max_idx[..., None] == 1,
            np.stack([xw, ones, xy, xz], axis=-1),
            np.where(
                max_idx[..., None] == 2,
                np.stack([yw, xy, ones, yz], axis=-1),
                np.stack([zw, xz, yz, ones], axis=-1)
            )
        )
    )
    quat = sign * wxyz
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=eps)
    return quat


def extrinsics_to_essential(extrinsics: ndarray):
    """
    extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

    ## Parameters
        extrinsics (np.ndaray): [..., 4, 4] extrinsics matrix

    ## Returns
        (np.ndaray): [..., 3, 3] essential matrix
    """
    assert extrinsics.shape[-2:] == (4, 4)
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    zeros = np.zeros_like(t[..., 0])
    t_x = np.stack([
        zeros, -t[..., 2], t[..., 1],
        t[..., 2], zeros, -t[..., 0],
        -t[..., 1], t[..., 0], zeros
    ]).reshape(*t.shape[:-1], 3, 3)
    return t_x @ R 


def euler_axis_angle_rotation(axis: str, angle: ndarray) -> ndarray:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    ## Parameters
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    ## Returns
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: ndarray, convention: str = 'XYZ') -> ndarray:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    ## Parameters
        euler_angles: Euler angles in radians as ndarray of shape (..., 3), XYZ
        convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

    ## Returns
        Rotation matrices as ndarray of shape (..., 3, 3).
    """
    if euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        euler_axis_angle_rotation(c, euler_angles[..., 'XYZ'.index(c)])
        for c in convention
    ]
    return matrices[2] @ matrices[1] @ matrices[0]


def skew_symmetric(v: ndarray):
    "Skew symmetric matrix from a 3D vector"
    assert v.shape[-1] == 3, "v must be 3D"
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = np.zeros_like(x)
    return np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], axis=-1).reshape(*v.shape[:-1], 3, 3)


def rotation_matrix_from_vectors(v1: ndarray, v2: ndarray):
    "Rotation matrix that rotates v1 to v2"
    I = np.eye(3, dtype=v1.dtype)
    v1 = v1 / np.linalg.norm(v1, axis=-1)
    v2 = v2 / np.linalg.norm(v2, axis=-1)
    v = np.cross(v1, v2, axis=-1)
    c = np.sum(v1 * v2, axis=-1)
    K = skew_symmetric(v)
    R = I + K + (1 / (1 + c)).astype(v1.dtype)[None, None] * (K @ K)    # Avoid numpy's default type casting for scalars
    return R


def axis_angle_to_matrix(axis_angle: ndarray, eps: float = 1e-12) -> ndarray:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

    ## Parameters
        axis_angle (ndarray): shape (..., 3), axis-angle vcetors

    ## Returns
        ndarray: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters
    """
    batch_shape = axis_angle.shape[:-1]
    dtype = axis_angle.dtype

    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True) 
    axis = axis_angle / (angle + eps)

    cos = np.cos(angle)[..., None, :]
    sin = np.sin(angle)[..., None, :]

    rx, ry, rz = np.split(axis, 3, axis=-1)
    zeros = np.zeros((*batch_shape, 1), dtype=dtype)
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=-1).reshape((*batch_shape, 3, 3))

    ident = np.eye(3, dtype=dtype)
    rot_mat = ident + sin * K + (1 - cos) * (K @ K)
    return rot_mat


def ray_intersection(p1: ndarray, d1: ndarray, p2: ndarray, d2: ndarray):
    """
    Compute the intersection/closest point of two D-dimensional rays
    If the rays are intersecting, the closest point is the intersection point.

    ## Parameters
        p1 (ndarray): (..., D) origin of ray 1
        d1 (ndarray): (..., D) direction of ray 1
        p2 (ndarray): (..., D) origin of ray 2
        d2 (ndarray): (..., D) direction of ray 2

    ## Returns
        (ndarray): (..., N) intersection point
    """
    p1, d1, p2, d2 = np.broadcast_arrays(p1, d1, p2, d2)
    dtype = p1.dtype
    dim = p1.shape[-1]
    d = np.stack([d1, d2], axis=-2)     # (..., 2, D)
    p = np.stack([p1, p2], axis=-2)     # (..., 2, D)
    A = np.concatenate([
        (np.eye(dim, dtype=dtype) * np.ones((*p.shape[:-2], 2, 1, 1))).reshape(*d.shape[:-2], 2 * dim, dim),         # (..., 2 * D, D)
        -(np.eye(2, dtype=dtype)[..., None] * d[..., None, :]).swapaxes(-2, -1).reshape(*d.shape[:-2], 2 * dim, 2)    # (..., 2 * D, 2)
    ], axis=-1)                             # (..., 2 * D, D + 2)
    b = p.reshape(*p.shape[:-2], 2 * dim)   # (..., 2 * D)
    x = np.linalg.solve(A.swapaxes(-1, -2) @ A + 1e-12 * np.eye(dim + 2, dtype=dtype), (A.swapaxes(-1, -2) @ b[..., :, None]))[..., 0]
    return x[..., :dim], (x[..., dim], x[..., dim + 1])


@batched(2, 1)
def make_se3_matrix(R: ndarray, t: ndarray) -> ndarray:
    """
    Convert rotation matrix and translation vector to 4x4 transformation matrix.

    ## Parameters
        R (ndarray): [..., 3, 3] rotation matrix
        t (ndarray): [..., 3] translation vector

    ## Returns
        ndarray: [..., 4, 4] transformation matrix
    """
    x = np.block([
        [R, t[..., None]], 
        [np.zeros((*R.shape[:-2], 1, R.shape[-1]), dtype=R.dtype), np.ones((*R.shape[:-2], 1, 1), dtype=R.dtype)]
    ])
    return x


def slerp_quaternion(q1: ndarray, q2: ndarray, t: ndarray) -> ndarray:
    """
    Spherical linear interpolation between two unit quaternions.

    ## Parameters
        q1 (ndarray): [..., d] unit vector 1
        q2 (ndarray): [..., d] unit vector 2
        t (ndarray): [...] interpolation parameter in [0, 1]

    ## Returns
        ndarray: [..., 3] interpolated unit vector
    """
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)

    dot = np.where(dot < 0, -dot, dot)  # handle negative dot product

    dot = np.minimum(dot, 1.)
    theta = np.arccos(dot) * t

    q_ortho = q2 - q1 * dot
    q_ortho = q_ortho / np.maximum(np.linalg.norm(q_ortho, axis=-1, keepdims=True), 1e-12)
    q = q1 * np.cos(theta) + q_ortho * np.sin(theta)
    return q


def slerp_rotation_matrix(R1: ndarray, R2: ndarray, t: ndarray) -> ndarray:
    """
    Spherical linear interpolation between two rotation matrices.

    ## Parameters
        R1 (ndarray): [..., 3, 3] rotation matrix 1
        R2 (ndarray): [..., 3, 3] rotation matrix 2
        t (ndarray): [...] interpolation parameter in [0, 1]

    ## Returns
        ndarray: [..., 3, 3] interpolated rotation matrix
    """
    quat1 = matrix_to_quaternion(R1)
    quat2 = matrix_to_quaternion(R2)
    quat = slerp_quaternion(quat1, quat2, t)
    return quaternion_to_matrix(quat)


def slerp_vector(v1: ndarray, v2: ndarray, t: ndarray) -> ndarray:
    """
    Spherical linear interpolation between two unit vectors. The vectors are assumed to be normalized.

    ## Parameters
        v1 (ndarray): [..., d] unit vector 1
        v2 (ndarray): [..., d] unit vector 2
        t (ndarray): [...] interpolation parameter in [0, 1]

    ## Returns
        ndarray: [..., d] interpolated unit vector
    """
    dot = np.sum(v1 * v2, axis=-1, keepdims=True)

    dot = np.minimum(dot, 1.)
    theta = np.arccos(dot) * t

    v_ortho = v2 - v1 * dot
    v_ortho = v_ortho / np.maximum(np.linalg.norm(v_ortho, axis=-1, keepdims=True), 1e-12)
    v = v1 * np.cos(theta) + v_ortho * np.sin(theta)
    return v


def lerp(x1: ndarray, x2: ndarray, t: ndarray) -> ndarray:
    """
    Linear interpolation between two vectors.

    ## Parameters
        x1 (ndarray): [..., d] vector 1
        x2 (ndarray): [..., d] vector 2
        t (ndarray): [...] interpolation parameter. [0, 1] for interpolation between x1 and x2, otherwise for extrapolation.

    ## Returns
        ndarray: [..., d] interpolated vector
    """
    return x1 + np.asarray(t)[..., None] * (x2 - x1)


def lerp_se3_matrix(T1: ndarray, T2: ndarray, t: ndarray) -> ndarray:
    """
    Linear interpolation between two SE(3) matrices.

    ## Parameters
        T1 (ndarray): [..., 4, 4] SE(3) matrix 1
        T2 (ndarray): [..., 4, 4] SE(3) matrix 2
        t (ndarray): [...] interpolation parameter in [0, 1]

    ## Returns
        ndarray: [..., 4, 4] interpolated SE(3) matrix
    """
    R1 = T1[..., :3, :3]
    R2 = T2[..., :3, :3]
    trans1 = T1[..., :3, 3]
    trans2 = T2[..., :3, 3]
    R = slerp_rotation_matrix(R1, R2, t)
    trans = lerp(trans1, trans2, t)
    return make_se3_matrix(R, trans)


def piecewise_lerp(x: ndarray, t: ndarray, s: ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> ndarray:
    """
    Linear spline interpolation.

    ## Parameters
    - `x`: ndarray, shape (n, d): the values of data points.
    - `t`: ndarray, shape (n,): the times of the data points.
    - `s`: ndarray, shape (m,): the times to be interpolated.
    - `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.
    
    ## Returns
    - `y`: ndarray, shape (..., m, d): the interpolated values.
    """
    i = np.searchsorted(t, s, side='left')
    if extrapolation_mode == 'constant':
        prev = np.clip(i - 1, 0, len(t) - 1)
        suc = np.clip(i, 0, len(t) - 1)
    elif extrapolation_mode == 'linear':
        prev = np.clip(i - 1, 0, len(t) - 2)
        suc = np.clip(i, 1, len(t) - 1)
    else:
        raise ValueError(f'Invalid extrapolation_mode: {extrapolation_mode}')
    
    u = (s - t[prev]) / np.maximum(t[suc] - t[prev], 1e-12)
    y = lerp(x[prev], x[suc], u)

    return y


def piecewise_lerp_se3_matrix(T: ndarray, t: ndarray, s: ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> ndarray:
    """
    Linear spline interpolation for SE(3) matrices.

    ## Parameters
    - `T`: ndarray, shape (n, 4, 4): the SE(3) matrices.
    - `t`: ndarray, shape (n,): the times of the data points.
    - `s`: ndarray, shape (m,): the times to be interpolated.
    - `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

    ## Returns
    - `T_interp`: ndarray, shape (..., m, 4, 4): the interpolated SE(3) matrices.
    """
    i = np.searchsorted(t, s, side='left')
    if extrapolation_mode == 'constant':
        prev = np.clip(i - 1, 0, len(t) - 1)
        suc = np.clip(i, 0, len(t) - 1)
    elif extrapolation_mode == 'linear':
        prev = np.clip(i - 1, 0, len(t) - 2)
        suc = np.clip(i, 1, len(t) - 1)
    else:
        raise ValueError(f'Invalid extrapolation_mode: {extrapolation_mode}')
    
    u = (s - t[prev]) / np.maximum(t[suc] - t[prev], 1e-12)
    T = lerp_se3_matrix(T[prev], T[suc], u)

    return T


def transform(x: ndarray, *Ts: ndarray) -> ndarray:
    """
    Apply affine transformation(s) to a point or a set of points.

    ## Parameters
    - `x`: ndarray, shape (..., D): the point or a set of points to be transformed.
    - `Ts`: ndarray, shape (..., D + 1, D + 1): the affine transformation matrix (matrice)
        If more than one transformation is given, they will be applied in corresponding order.
    ## Returns
    - `y`: ndarray, shape (..., D): the transformed point or a set of points.

    ## Example Usage
    ```
    y = transform(x, T1, T2, T3)
    ```
    """
    y = np.concatenate([x, np.ones_like(x[..., :1])], axis=-1)[..., None]
    for T in Ts:
        y = T @ y
    return y[..., :3, 0]


def angle_between(v1: ndarray, v2: ndarray):
    """
    Calculate the angle between two vectors.
    """
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    v1 = v1 / np.where(n1 == 0, 1, n1)
    v2 = v2 / np.where(n2 == 0, 1, n2)
    cos = (v1 * v2).sum(axis=-1)
    sin = np.minimum(np.linalg.norm(v2 - v1 * cos[..., None], axis=-1), np.linalg.norm(v1 - v2 * cos[..., None], axis=-1))
    return np.atan2(sin, cos)
