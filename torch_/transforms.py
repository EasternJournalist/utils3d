import torch

from ..numpy_.utils import (
    perspective_from_fov as __perspective_from_fov, 
    perspective_from_fov_xy as __perspective_from_fov_xy,
)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
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
    Code from pytorch3d
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

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
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    """Calculates the rotation matrices for a batch of rotation vectors. (code from SMPLX)

    Args:
        rot_vecs (torch.Tensor): shape (..., 3), axis-angle vetors

    Returns:
        torch.Tensor: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters
    """
    batch_shape = rot_vecs.shape[:-1]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=-1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle)[..., None, :]
    sin = torch.sin(angle)[..., None, :]

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=-1)
    K = torch.zeros((*batch_shape, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((*batch_shape, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((*batch_shape, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def perspective_from_fov(fov: float, width: int, height: int, near: float, far: float) -> torch.Tensor:
    return torch.from_numpy(__perspective_from_fov(fov, width, height, near, far))

def perspective_from_fov_xy(fov_x: float, fov_y: float, near: float, far: float) -> torch.Tensor:
    return torch.from_numpy(__perspective_from_fov_xy(fov_x, fov_y, near, far))

def perspective_to_intrinsic(perspective: torch.Tensor) -> torch.Tensor:
    """OpenGL convention perspective matrix to OpenCV convention intrinsic

    Args:
        perspective (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix

    Returns:
        torch.Tensor: shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
    """
    fx, fy = perspective[..., 0, 0], perspective[..., 1, 1]
    cx, cy = perspective[..., 0, 2], perspective[..., 1, 2]
    zero = torch.zeros_like(fx)
    one = torch.full_like(fx, -1)

    matrix = [
        [0.5 * fx,     zero, -0.5 * cx + 0.5],
        [    zero, 0.5 * fy,  0.5 * cy + 0.5],
        [    zero,     zero,             one]]
    return torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)

def intrinsic_to_perspective(intrinsic: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """OpenGL convention perspective matrix to OpenCV convention intrinsic

    Args:
        intrinsic (torch.Tensor): shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        torch.Tensor: shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix
    """
    fx, fy = intrinsic[..., 0, 0], intrinsic[..., 1, 1]
    cx, cy = intrinsic[..., 0, 2], intrinsic[..., 1, 2]
    zero = torch.zeros_like(fx)
    negone = torch.full_like(fx, -1)
    a = torch.full_like(fx, (near + far) / (near - far))
    b = torch.full_like(fx, 2. * near * far / (near - far))

    matrix = [
        [2 * fx,   zero,  -2 * cx + 1, zero],
        [  zero, 2 * fy,   2 * cy - 1, zero],
        [  zero,   zero,            a,    b],
        [  zero,   zero,       negone, zero]]
    return torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)

def extrinsic_to_view(extrinsic: torch.Tensor) -> torch.Tensor:
    """OpenCV convention camera extrinsic to OpenGL convention view matrix

    Args:
        extrinsic (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenCV convention camera extrinsic

    Returns:
        torch.Tensor: shape (4, 4) or (..., 4, 4) OpenGL convention view matrix
    """
    return torch.inverse(extrinsic) @ torch.diag(torch.tensor([1, -1, -1, 1])).to(extrinsic)

def view_to_extrinsic(view: torch.Tensor) -> torch.Tensor:
    """OpenCV convention camera extrinsic to OpenGL convention view matrix

    Args:
        view (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix

    Returns:
        torch.Tensor: shape (4, 4) or (..., 4, 4) OpenCV convention camera extrinsic
    """
    return torch.diag(torch.tensor([1, -1, -1, 1])).to(view) @ torch.inverse(view)

def camera_cv_to_gl(extrinsic: torch.Tensor, intrinsic: torch.Tensor, near: float, far: float):
    """Convert OpenCV convention camera extrinsic & intrinsic to OpenGL convention view matrix and perspective matrix

    Args:
        extrinsic (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenCV convention camera extrinsic
        intrinsic (torch.Tensor): shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
        near (float): near plane to clip
        far (float): far plane to clip

    Returns:
        view (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix
        perspective (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix
    """
    return extrinsic_to_view(extrinsic), intrinsic_to_perspective(intrinsic, near, far)

def camera_gl_to_cv(view, perspective):
    """Convert OpenGL convention view matrix & perspective matrix to OpenCV convention camera extrinsic & intrinsic 

    Args:
        view (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix
        perspective (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix

    Returns:
        view (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenCV convention camera extrinsic
        perspective (torch.Tensor): shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
    """
    return view_to_extrinsic(view), perspective_to_intrinsic(perspective)

def normalize_intrinsic(intrinsic: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """normalize camera intrinsic
    Args:
        intrinsic (torch.Tensor): shape (..., 3, 3) 
        width (int): image width
        height (int): image height

    Returns:
        (torch.Tensor): shape (..., 3, 3), same as input intrinsic. Normalized intrinsic(s)
    """
    return intrinsic * torch.tensor([1 / width, 1 / height, 1])[:, None].to(intrinsic)

def crop_intrinsic(intrinsic: torch.Tensor, width: int, height: int, left: int, top: int, crop_width: int, crop_height: int):
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
    s = torch.tensor([
        [width / crop_width, 0, -left / crop_width], 
        [0, height / crop_height,  -top / crop_height], 
        [0., 0., 1.]]).to(intrinsic)
    return s @ intrinsic

def projection(vertices: torch.Tensor, model_matrix: torch.Tensor = None, view_matrix: torch.Tensor = None, projection_matrix: torch.Tensor = None) -> torch.Tensor:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrice)

    Args:
        vertices (torch.Tensor): 3D vertices positions of shape (batch, n, 3)
        model_matrix (torch.Tensor): row major model to world matrix of shape (batch, 4, 4)
        view_matrix (torch.Tensor): camera to world matrix of shape (batch, 4, 4)
        projection_matrix (torch.Tensor): projection matrix of shape (batch, 4, 4)

    Returns:
        scr_coord (torch.Tensor): vertex screen space coordinates of shape (batch, n, 3), value ranging in [0, 1]. 
            first two channels are screen space uv, where the origin (0., 0.) is corresponding to the bottom-left corner of the screen.
        linear_depth (torch.Tensor): 
    """
    if model_matrix is None: model_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]
    if view_matrix is None: view_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]
    if projection_matrix is None: projection_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]

    vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    clip_coord = vertices @ (projection_matrix @ torch.inverse(view_matrix) @ model_matrix).transpose(dim0=-2, dim1=-1)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord, linear_depth

def projection_ndc(vertices: torch.Tensor, model_matrix: torch.Tensor = None, view_matrix: torch.Tensor = None, projection_matrix: torch.Tensor = None) -> torch.Tensor:
    """Very similar to projection(), but return NDC space coordinates instead of screen space coordinates.
    The only difference is scr_coord = ndc_coord * 0.5 + 0.5.

    Args:
        vertices (torch.Tensor): 3D vertices positions of shape (batch, n, 3)
        model_matrix (torch.Tensor): row major model to world matrix of shape (batch, 4, 4)
        view_matrix (torch.Tensor): camera to world matrix of shape (batch, 4, 4)
        projection_matrix (torch.Tensor): projection matrix of shape (batch, 4, 4)

    Returns:
        ndc_coord (torch.Tensor): NDC space coordinates of shape (batch, n, 3), value ranging in [-1, 1]. Point (-1, -1, -1) is corresponding to the left & bottom & nearest.
    """
    if model_matrix is None: model_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]
    if view_matrix is None: view_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]
    if projection_matrix is None: projection_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]

    vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    clip_coord = vertices @ (projection_matrix @ torch.inverse(view_matrix) @ model_matrix).transpose(dim0=-2, dim1=-1)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    linear_depth = clip_coord[..., 3]
    return ndc_coord, linear_depth