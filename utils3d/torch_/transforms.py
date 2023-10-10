from typing import *
from numbers import Number 

import torch
import torch.nn.functional as F

from ._helpers import batched


__all__ = [
    'perspective',
    'perspective_from_fov',
    'perspective_from_fov_xy',
    'intrinsic',
    'intrinsic_from_fov',
    'intrinsic_from_fov_xy',
    'view_look_at',
    'extrinsic_look_at',
    'perspective_to_intrinsic',
    'intrinsic_to_perspective',
    'extrinsic_to_view',
    'view_to_extrinsic',
    'normalize_intrinsic',
    'crop_intrinsic',
    'pixel_to_uv',
    'pixel_to_ndc',
    'project_depth',
    'linearize_depth',
    'project_gl',
    'project_cv',
    'unproject_gl',
    'unproject_cv',
    'euler_axis_angle_rotation',
    'euler_angles_to_matrix',
    'matrix_to_quaternion',
    'quaternion_to_matrix',
    'matrix_to_axis_angle',
    'axis_angle_to_matrix',
    'axis_angle_to_quaternion',
    'quaternion_to_axis_angle',
    'slerp',
    'interpolate_extrinsics',
    'interpolate_view',
]


@batched(0,0,0,0)
def perspective(
        fov_y: Union[float, torch.Tensor],
        aspect: Union[float, torch.Tensor],
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenGL perspective matrix

    Args:
        fov_y (float | torch.Tensor): field of view in y axis
        aspect (float | torch.Tensor): aspect ratio
        near (float | torch.Tensor): near plane to clip
        far (float | torch.Tensor): far plane to clip

    Returns:
        (torch.Tensor): [..., 4, 4] perspective matrix
    """
    N = fov_y.shape[0]
    ret = torch.zeros((N, 4, 4), dtype=fov_y.dtype, device=fov_y.device)
    ret[:, 0, 0] = 1. / (torch.tan(fov_y / 2) * aspect)
    ret[:, 1, 1] = 1. / (torch.tan(fov_y / 2))
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2. * near * far / (near - far)
    ret[:, 3, 2] = -1.
    return ret


def perspective_from_fov(
        fov: Union[float, torch.Tensor],
        width: Union[int, torch.Tensor],
        height: Union[int, torch.Tensor],
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenGL perspective matrix from field of view in largest dimension

    Args:
        fov (float | torch.Tensor): field of view in largest dimension
        width (int | torch.Tensor): image width
        height (int | torch.Tensor): image height
        near (float | torch.Tensor): near plane to clip
        far (float | torch.Tensor): far plane to clip

    Returns:
        (torch.Tensor): [..., 4, 4] perspective matrix
    """
    fov_y = 2 * torch.atan(torch.tan(fov / 2) * height / torch.maximum(width, height))
    aspect = width / height
    return perspective(fov_y, aspect, near, far)


def perspective_from_fov_xy(
        fov_x: Union[float, torch.Tensor],
        fov_y: Union[float, torch.Tensor],
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenGL perspective matrix from field of view in x and y axis

    Args:
        fov_x (float | torch.Tensor): field of view in x axis
        fov_y (float | torch.Tensor): field of view in y axis
        near (float | torch.Tensor): near plane to clip
        far (float | torch.Tensor): far plane to clip

    Returns:
        (torch.Tensor): [..., 4, 4] perspective matrix
    """
    aspect = torch.tan(fov_x / 2) / torch.tan(fov_y / 2)
    return perspective(fov_y, aspect, near, far)


@batched(0,0,0,0)
def intrinsic(
        focal_x: Union[float, torch.Tensor],
        focal_y: Union[float, torch.Tensor],
        cx: Union[float, torch.Tensor],
        cy: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenCV intrinsic matrix

    Args:
        focal_x (float | torch.Tensor): focal length in x axis
        focal_y (float | torch.Tensor): focal length in y axis
        cx (float | torch.Tensor): principal point in x axis
        cy (float | torch.Tensor): principal point in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsic matrix
    """
    N = focal_x.shape[0]
    ret = torch.zeros((N, 3, 3), dtype=focal_x.dtype, device=focal_x.device)
    ret[:, 0, 0] = focal_x
    ret[:, 1, 1] = focal_y
    ret[:, 0, 2] = cx
    ret[:, 1, 2] = cy
    ret[:, 2, 2] = 1.
    return ret


@batched(0, 0, 0, None)
def intrinsic_from_fov(
        fov: Union[float, torch.Tensor],
        width: Union[int, torch.Tensor],
        height: Union[int, torch.Tensor],
        normalize: bool = False
    ) -> torch.Tensor:
    """
    Get OpenCV intrinsic matrix from field of view in largest dimension

    Args:
        fov (float | torch.Tensor): field of view in largest dimension
        width (int | torch.Tensor): image width
        height (int | torch.Tensor): image height
        normalize (bool, optional): whether to normalize the intrinsic matrix. Defaults to False.

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsic matrix
    """
    focal = torch.maximum(width, height) / (2 * torch.tan(fov / 2))
    cx = width / 2
    cy = height / 2
    ret = intrinsic(focal, focal, cx, cy)
    if normalize:
        ret = normalize_intrinsic(ret, width, height)
    return ret


def intrinsic_from_fov_xy(
        fov_x: Union[float, torch.Tensor],
        fov_y: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenCV intrinsic matrix from field of view in x and y axis

    Args:
        fov_x (float | torch.Tensor): field of view in x axis
        fov_y (float | torch.Tensor): field of view in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsic matrix
    """
    focal_x = 0.5 / torch.tan(fov_x / 2)
    focal_y = 0.5 / torch.tan(fov_y / 2)
    cx = cy = 0.5
    return intrinsic(focal_x, focal_y, cx, cy)


@batched(1,1,1)
def view_look_at(
        eye: torch.Tensor,
        look_at: torch.Tensor,
        up: torch.Tensor
    ) -> torch.Tensor:
    """
    Get OpenGL view matrix looking at something

    Args:
        eye (torch.Tensor): [..., 3] the eye position
        look_at (torch.Tensor): [..., 3] the position to look at
        up (torch.Tensor): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (torch.Tensor): [..., 4, 4], view matrix
    """
    N = eye.shape[0]
    z = eye - look_at
    x = torch.cross(up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    # x = torch.cross(y, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    R = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(R, eye[..., None])
    ret = torch.zeros((N, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = R
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.
    return ret


@batched(1, 1, 1)
def extrinsic_look_at(
    eye: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """
    Get OpenCV extrinsic matrix looking at something

    Args:
        eye (torch.Tensor): [..., 3] the eye position
        look_at (torch.Tensor): [..., 3] the position to look at
        up (torch.Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (torch.Tensor): [..., 4, 4], extrinsic matrix
    """
    N = eye.shape[0]
    z = look_at - eye
    x = torch.cross(-up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    # x = torch.cross(y, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    R = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(R, eye[..., None])
    ret = torch.zeros((N, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = R
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.
    return ret


@batched(2)
def perspective_to_intrinsic(
        perspective: torch.Tensor
    ) -> torch.Tensor:
    """
    OpenGL perspective matrix to OpenCV intrinsic

    Args:
        perspective (torch.Tensor): [..., 4, 4] OpenGL perspective matrix

    Returns:
        (torch.Tensor): shape [..., 3, 3] OpenCV intrinsic
    """
    N = perspective.shape[0]
    fx, fy = perspective[:, 0, 0], perspective[:, 1, 1]
    cx, cy = perspective[:, 0, 2], perspective[:, 1, 2]
    ret = torch.zeros((N, 3, 3), dtype=perspective.dtype, device=perspective.device)
    ret[:, 0, 0] = 0.5 * fx
    ret[:, 1, 1] = 0.5 * fy
    ret[:, 0, 2] = -0.5 * cx + 0.5
    ret[:, 1, 2] = 0.5 * cy + 0.5
    ret[:, 2, 2] = 1.
    return ret


@batched(2,0,0)
def intrinsic_to_perspective(
        intrinsic: torch.Tensor,
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor],
    ) -> torch.Tensor:
    """
    OpenCV intrinsic to OpenGL perspective matrix

    Args:
        intrinsic (torch.Tensor): [..., 3, 3] OpenCV intrinsic matrix
        near (float | torch.Tensor): [...] near plane to clip
        far (float | torch.Tensor): [...] far plane to clip
    Returns:
        (torch.Tensor): [..., 4, 4] OpenGL perspective matrix
    """
    N = intrinsic.shape[0]
    fx, fy = intrinsic[:, 0, 0], intrinsic[:, 1, 1]
    cx, cy = intrinsic[:, 0, 2], intrinsic[:, 1, 2]
    ret = torch.zeros((N, 4, 4), dtype=intrinsic.dtype, device=intrinsic.device)
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
        extrinsic: torch.Tensor
    ) -> torch.Tensor:
    """
    OpenCV camera extrinsic to OpenGL view matrix

    Args:
        extrinsic (torch.Tensor): [..., 4, 4] OpenCV camera extrinsic matrix

    Returns:
        (torch.Tensor): [..., 4, 4] OpenGL view matrix
    """
    return extrinsic * torch.tensor([1, -1, -1, 1], dtype=extrinsic.dtype, device=extrinsic.device)[:, None]


@batched(2)
def view_to_extrinsic(
        view: torch.Tensor
    ) -> torch.Tensor:
    """
    OpenGL view matrix to OpenCV camera extrinsic

    Args:
        view (torch.Tensor): [..., 4, 4] OpenGL view matrix

    Returns:
        (torch.Tensor): [..., 4, 4] OpenCV camera extrinsic matrix
    """
    return view  * torch.tensor([1, -1, -1, 1], dtype=view.dtype, device=view.device)[:, None]


@batched(2,0,0)
def normalize_intrinsic(
        intrinsic: torch.Tensor,
        width: Union[int, torch.Tensor],
        height: Union[int, torch.Tensor]
    ) -> torch.Tensor:
    """
    Normalize camera intrinsic(s) to uv space

    Args:
        intrinsic (torch.Tensor): [..., 3, 3] camera intrinsic(s) to normalize
        width (int | torch.Tensor): [...] image width(s)
        height (int | torch.Tensor): [...] image height(s)

    Returns:
        (torch.Tensor): [..., 3, 3] normalized camera intrinsic(s)
    """
    return intrinsic * torch.stack([1 / width, 1 / height, torch.ones_like(width)], dim=-1)[..., None]


@batched(2,0,0,0,0,0,0)
def crop_intrinsic(
        intrinsic: torch.Tensor,
        width: Union[int, torch.Tensor],
        height: Union[int, torch.Tensor],
        left: Union[int, torch.Tensor],
        top: Union[int, torch.Tensor],
        crop_width: Union[int, torch.Tensor],
        crop_height: Union[int, torch.Tensor]
    ) -> torch.Tensor:
    """
    Evaluate the new intrinsic(s) after crop the image: cropped_img = img[top:top+crop_height, left:left+crop_width]

    Args:
        intrinsic (torch.Tensor): [..., 3, 3] camera intrinsic(s) to crop
        width (int | torch.Tensor): [...] image width(s)
        height (int | torch.Tensor): [...] image height(s)
        left (int | torch.Tensor): [...] left crop boundary
        top (int | torch.Tensor): [...] top crop boundary
        crop_width (int | torch.Tensor): [...] crop width
        crop_height (int | torch.Tensor): [...] crop height

    Returns:
        (torch.Tensor): [..., 3, 3] cropped camera intrinsic(s)
    """
    intrinsic = intrinsic.clone().detach()
    intrinsic[..., 0, 0] *= width / crop_width
    intrinsic[..., 1, 1] *= height / crop_height
    intrinsic[..., 0, 2] = (intrinsic[..., 0, 2] * width - left) / crop_width
    intrinsic[..., 1, 2] = (intrinsic[..., 1, 2] * height - top) / crop_height
    return intrinsic


@batched(1,0,0)
def pixel_to_uv(
        pixel: torch.Tensor,
        width: Union[int, torch.Tensor],
        height: Union[int, torch.Tensor]
    ) -> torch.Tensor:
    """
    Args:
        pixel (torch.Tensor): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | torch.Tensor): [...] image width(s)
        height (int | torch.Tensor): [...] image height(s)

    Returns:
        (torch.Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    uv = torch.zeros(pixel.shape, dtype=torch.float32)
    uv[..., 0] = (pixel[..., 0] + 0.5) / width
    uv[..., 1] = (pixel[..., 1] + 0.5) / height
    return uv


@batched(1,0,0)
def pixel_to_ndc(
        pixel: torch.Tensor,
        width: Union[int, torch.Tensor],
        height: Union[int, torch.Tensor]
    ) -> torch.Tensor:
    """
    Args:
        pixel (torch.Tensor): [..., 2] pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
        width (int | torch.Tensor): [...] image width(s)
        height (int | torch.Tensor): [...] image height(s)

    Returns:
        (torch.Tensor): [..., 2] pixel coordinrates defined in ndc space, the range is (-1, 1)
    """
    ndc = torch.zeros(pixel.shape, dtype=torch.float32)
    ndc[..., 0] = (pixel[..., 0] + 0.5) / width * 2 - 1
    ndc[..., 1] = -((pixel[..., 1] + 0.5) / height * 2 - 1)
    return ndc


@batched(0,0,0)
def project_depth(
        depth: torch.Tensor,
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Project linear depth to depth value in screen space

    Args:
        depth (torch.Tensor): [...] depth value
        near (float | torch.Tensor): [...] near plane to clip
        far (float | torch.Tensor): [...] far plane to clip

    Returns:
        (torch.Tensor): [..., 1] depth value in screen space, value ranging in [0, 1]
    """
    return (far - near * far / depth) / (far - near)


@batched(0,0,0)
def linearize_depth(
        depth: torch.Tensor,
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Linearize depth value to linear depth

    Args:
        depth (torch.Tensor): [...] screen depth value, ranging in [0, 1]
        near (float | torch.Tensor): [...] near plane to clip
        far (float | torch.Tensor): [...] far plane to clip

    Returns:
        (torch.Tensor): [...] linear depth
    """
    return near * far / (far - (far - near) * depth)


@batched(2, 2, 2, 2)
def project_gl(
    points: torch.Tensor,
    model: torch.Tensor = None,
    view: torch.Tensor = None,
    perspective: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D following the OpenGL convention (except for row major matrice)

    Args:
        points (torch.Tensor): [..., N, 3 or 4] 3D points to project, if the last 
            dimension is 4, the points are assumed to be in homogeneous coordinates
        model (torch.Tensor): [..., 4, 4] model matrix
        view (torch.Tensor): [..., 4, 4] view matrix
        perspective (torch.Tensor): [..., 4, 4] perspective matrix

    Returns:
        scr_coord (torch.Tensor): [..., N, 3] screen space coordinates, value ranging in [0, 1].
            The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        linear_depth (torch.Tensor): [..., N] linear depth
    """
    assert perspective is not None, "perspective matrix is required"

    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    mvp = perspective if perspective is not None else torch.eye(4).to(points)
    if view is not None:
        mvp = mvp @ view
    if model is not None:
        mvp = mvp @ model
    clip_coord = points @ mvp.transpose(-1, -2)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord, linear_depth


@batched(2, 2, 2)
def project_cv(
    points: torch.Tensor,
    extrinsic: torch.Tensor = None,
    intrinsic: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D following the OpenCV convention

    Args:
        points (torch.Tensor): [..., N, 3] or [..., N, 4] 3D points to project, if the last
            dimension is 4, the points are assumed to be in homogeneous coordinates
        extrinsic (torch.Tensor): [..., 4, 4] extrinsic matrix
        intrinsic (torch.Tensor): [..., 3, 3] intrinsic matrix

    Returns:
        uv_coord (torch.Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (torch.Tensor): [..., N] linear depth
    """
    assert intrinsic is not None, "intrinsic matrix is required"
    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    if extrinsic is not None:
        points = points @ extrinsic.transpose(-1, -2)
    points = points[..., :3] @ intrinsic.transpose(-2, -1)
    uv_coord = points[..., :2] / points[..., 2:]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth


@batched(2, 2, 2, 2)
def unproject_gl(
        screen_coord: torch.Tensor,
        model: torch.Tensor = None,
        view: torch.Tensor = None,
        perspective: torch.Tensor = None
    ) -> torch.Tensor:
    """
    Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrice)

    Args:
        screen_coord (torch.Tensor): [... N, 3] screen space coordinates, value ranging in [0, 1].
            The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        model (torch.Tensor): [..., 4, 4] model matrix
        view (torch.Tensor): [..., 4, 4] view matrix
        perspective (torch.Tensor): [..., 4, 4] perspective matrix

    Returns:
        points (torch.Tensor): [..., N, 3] 3d points
    """
    assert perspective is not None, "perspective matrix is required"
    ndc_xy = screen_coord * 2 - 1
    clip_coord = torch.cat([ndc_xy, torch.ones_like(ndc_xy[..., :1])], dim=-1)
    transform = perspective
    if view is not None:
        transform = transform @ view
    if model is not None:
        transform = transform @ model
    transform = torch.inverse(transform)
    points = clip_coord @ transform.transpose(-1, -2)
    points = points[..., :3] / points[..., 3:]
    return points
    

@batched(2, 1, 2, 2)
def unproject_cv(
    uv_coord: torch.Tensor,
    depth: torch.Tensor,
    extrinsic: torch.Tensor = None,
    intrinsic: torch.Tensor = None
) -> torch.Tensor:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    Args:
        uv_coord (torch.Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (torch.Tensor): [..., N] depth value
        extrinsic (torch.Tensor): [..., 4, 4] extrinsic matrix
        intrinsic (torch.Tensor): [..., 3, 3] intrinsic matrix

    Returns:
        points (torch.Tensor): [..., N, 3] 3d points
    """
    assert intrinsic is not None, "intrinsic matrix is required"
    points = torch.cat([uv_coord, torch.ones_like(uv_coord[..., :1])], dim=-1)
    points = points @ torch.inverse(intrinsic).transpose(-2, -1)
    points = points * depth[..., None]
    if extrinsic is not None:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        points = (points @ torch.inverse(extrinsic).transpose(-2, -1))[..., :3]
    return points


def euler_axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Code MODIFIED from pytorch3d
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3), XYZ
        convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
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
    # return functools.reduce(torch.matmul, matrices)
    return matrices[2] @ matrices[1] @ matrices[0]


def axis_angle_to_matrix(axis_angle: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

    Args:
        axis_angle (torch.Tensor): shape (..., 3), axis-angle vcetors

    Returns:
        torch.Tensor: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters
    """
    batch_shape = axis_angle.shape[:-1]
    device, dtype = axis_angle.device, axis_angle.dtype

    angle = torch.norm(axis_angle + eps, dim=-1, keepdim=True)
    axis = axis_angle / angle

    cos = torch.cos(angle)[..., None, :]
    sin = torch.sin(angle)[..., None, :]

    rx, ry, rz = torch.split(axis, 1, dim=-1)
    zeros = torch.zeros((*batch_shape, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1).view((*batch_shape, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)
    return rot_mat


def matrix_to_axis_angle(rot_mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert a batch of 3x3 rotation matrices to axis-angle representation (rotation vector)

    Args:
        rot_mat (torch.Tensor): shape (..., 3, 3), the rotation matrices to convert

    Returns:
        torch.Tensor: shape (..., 3), the axis-angle vectors corresponding to the given rotation matrices
    """
    quat = matrix_to_quaternion(rot_mat)
    axis_angle = quaternion_to_axis_angle(quat, eps=eps)
    return axis_angle


def quaternion_to_axis_angle(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert a batch of quaternions (w, x, y, z) to axis-angle representation (rotation vector)

    Args:
        quaternion (torch.Tensor): shape (..., 4), the quaternions to convert

    Returns:
        torch.Tensor: shape (..., 3), the axis-angle vectors corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    norm = torch.norm(quaternion[..., 1:], dim=-1, keepdim=True)
    axis = quaternion[..., 1:] / norm.clamp(min=eps)
    angle = 2 * torch.atan2(norm, quaternion[..., 0:1])
    return angle * axis


def axis_angle_to_quaternion(axis_angle: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert axis-angle representation (rotation vector) to quaternion (w, x, y, z)

    Args:
        axis_angle (torch.Tensor): shape (..., 3), axis-angle vcetors

    Returns:
        torch.Tensor: shape (..., 4) The quaternions for the given axis-angle parameters
    """
    axis = F.normalize(axis_angle, dim=-1, eps=eps)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    quat = torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=-1)
    return quat


def matrix_to_quaternion(rot_mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

    Args:
        rot_mat (torch.Tensor): shape (..., 3, 3), the rotation matrices to convert

    Returns:
        torch.Tensor: shape (..., 4), the quaternions corresponding to the given rotation matrices
    """
    # Extract the diagonal and off-diagonal elements of the rotation matrix
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rot_mat.flatten(-2).unbind(dim=-1)

    diag = torch.diagonal(rot_mat, dim1=-2, dim2=-1)
    M = torch.tensor([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=rot_mat.dtype, device=rot_mat.device)
    wxyz = (1 + diag @ M.transpose(-1, -2)).clamp_(0).sqrt().mul(0.5)
    _, max_idx = wxyz.max(dim=-1)
    xw = torch.sign(m21 - m12)
    yw = torch.sign(m02 - m20)
    zw = torch.sign(m10 - m01)
    yz = torch.sign(m21 + m12)
    xz = torch.sign(m02 + m20)
    xy = torch.sign(m01 + m10)
    ones = torch.ones_like(xw)
    sign = torch.where(
        max_idx[..., None] == 0,
        torch.stack([ones, xw, yw, zw], dim=-1),
        torch.where(
            max_idx[..., None] == 1,
            torch.stack([xw, ones, xy, xz], dim=-1),
            torch.where(
                max_idx[..., None] == 2,
                torch.stack([yw, xy, ones, yz], dim=-1),
                torch.stack([zw, xz, yz, ones], dim=-1)
            )
        )
    )
    quat = sign * wxyz
    quat = F.normalize(quat, dim=-1, eps=eps)
    return quat


def quaternion_to_matrix(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices
    
    Args:
        quaternion (torch.Tensor): shape (..., 4), the quaternions to convert
    
    Returns:
        torch.Tensor: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    quaternion = F.normalize(quaternion, dim=-1, eps=eps)
    w, x, y, z = quaternion.unbind(dim=-1)
    zeros = torch.zeros_like(w)
    I = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    xyz = quaternion[..., 1:]
    A = xyz[..., :, None] * xyz[..., None, :] - I * (xyz ** 2).sum(dim=-1)[..., None, None]
    B = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=-1).unflatten(-1, (3, 3))
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def slerp(rot_mat_1: torch.Tensor, rot_mat_2: torch.Tensor, t: Union[Number, torch.Tensor]) -> torch.Tensor:
    """Spherical linear interpolation between two rotation matrices

    Args:
        rot_mat_1 (torch.Tensor): shape (..., 3, 3), the first rotation matrix
        rot_mat_2 (torch.Tensor): shape (..., 3, 3), the second rotation matrix
        t (torch.Tensor): scalar or shape (...,), the interpolation factor

    Returns:
        torch.Tensor: shape (..., 3, 3), the interpolated rotation matrix
    """
    assert rot_mat_1.shape == rot_mat_2.shape
    assert rot_mat_1.shape[-2:] == (3, 3)
    rot_vec_1 = matrix_to_axis_angle(rot_mat_1)
    rot_vec_2 = matrix_to_axis_angle(rot_mat_2)
    if isinstance(t, Number):
        t = torch.tensor(t, dtype=rot_mat_1.dtype, device=rot_mat_1.device)
    rot_vec = (1 - t[..., None]) * rot_vec_1 + t[..., None] * rot_vec_2
    rot_mat = axis_angle_to_matrix(rot_vec)
    return rot_mat


def interpolate_extrinsics(ext1: torch.Tensor, ext2: torch.Tensor, t: Union[Number, torch.Tensor]) -> torch.Tensor:
    """Interpolate extrinsics between two camera poses. Linear interpolation for translation, spherical linear interpolation for rotation.

    Args:
        ext1 (torch.Tensor): shape (..., 4, 4), the first camera pose
        ext2 (torch.Tensor): shape (..., 4, 4), the second camera pose
        t (torch.Tensor): scalar or shape (...,), the interpolation factor

    Returns:
        torch.Tensor: shape (..., 4, 4), the interpolated camera pose
    """
    assert ext1.shape[-2:] == (4, 4) and ext2.shape[-2:] == (4, 4)
    pos1 = ext1[..., :3, 3] @ ext1[..., :3, :3]
    pos2 = ext2[..., :3, 3] @ ext2[..., :3, :3]
    if isinstance(t, Number):
        t = torch.tensor(t, dtype=ext1.dtype, device=ext1.device)
    pos = (1 - t[..., None]) * pos1 + t[..., None] * pos2
    rot = slerp(ext1[..., :3, :3], ext2[..., :3, :3], t)
    ext = torch.cat([rot, rot @ pos[..., None]], dim=-1)
    ext = torch.cat([ext, torch.tensor([0, 0, 0, 1], dtype=ext.dtype, device=ext.device).expand_as(ext[..., :1, :])], dim=-2)
    return ext


def interpolate_view(view1: torch.Tensor, view2: torch.Tensor, t: Union[Number, torch.Tensor]):
    """Interpolate view matrices between two camera poses. Linear interpolation for translation, spherical linear interpolation for rotation.

    Args:
        ext1 (torch.Tensor): shape (..., 4, 4), the first camera pose
        ext2 (torch.Tensor): shape (..., 4, 4), the second camera pose
        t (torch.Tensor): scalar or shape (...,), the interpolation factor

    Returns:
        torch.Tensor: shape (..., 4, 4), the interpolated camera pose
    """
    return interpolate_extrinsics(view1, view2, t)