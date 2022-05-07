import torch
from typing import Tuple

from ..numpy_.utils import (
    perspective_from_fov as __perspective_from_fov, 
    perspective_from_fov_xy as __perspective_from_fov_xy,
    image_uv as __image_uv,
    image_mesh as __image_mesh,
    to_linear_depth as __to_linear_depth,
    to_depth_buffer as __to_depth_buffer,
)

def to_linear_depth(depth_buffer: torch.Tensor) -> torch.Tensor:
    return __to_linear_depth(depth_buffer)

def to_depth_buffer(linear_depth: torch.Tensor) -> torch.Tensor:
    return __to_depth_buffer(linear_depth)

def triangulate(faces: torch.Tensor) -> torch.Tensor:
    assert len(faces.shape) == 2
    if faces.shape[1] == 3:
        return faces
    n = faces.shape[1]
    loop_indice = torch.stack([
        torch.zeros(n - 2, dtype=torch.int64), 
        torch.arange(1, n - 1, 1, dtype=torch.int64), 
        torch.arange(2, n, 1, dtype=torch.int64)
    ], dim=1)
    return faces[:, loop_indice].reshape(-1, 3)

def compute_face_normal(vertices: torch.Tensor, faces: torch.Tensor):
    """Compute face normals of a triangular mesh

    Args:
        vertices (np.ndarray):  3-dimensional vertices of shape (..., N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)

    Returns:
        normals (np.ndarray): face normals of shape (..., T, 3)
    """
    normal = torch.cross(torch.index_select(vertices, dim=-2, index=faces[:, 1]) - torch.index_select(vertices, dim=-2, index=faces[:, 0]), torch.index_select(vertices, dim=-2, index=faces[:, 2]) - torch.index_select(vertices, dim=-2, index=faces[:, 0]))
    normal = torch.nan_to_num(normal / torch.norm(normal, p=2, dim=-1, keepdim=True))
    return normal

def compute_vertex_normal(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute vertex normals as mean of adjacent face normals

    Args:
        vertices (np.ndarray): 3-dimensional vertices of shape (..., N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)

    Returns:
        normals (np.ndarray): vertex normals of shape (..., N, 3)
    """
    face_normal = compute_face_normal(vertices, faces) # (..., T, 3)
    face_normal = face_normal[..., None, :].repeat(*[1] * (len(vertices.shape) - 1), 3, 1).view(*face_normal.shape[:-2], -1, 3) # (..., T * 3, 3)
    vertex_normal = torch.index_add(torch.zeros_like(vertices), dim=-2, index=faces.view(-1), source=face_normal)
    vertex_normal = torch.nan_to_num(vertex_normal / torch.norm(vertex_normal, p=2, dim=-1, keepdim=True))
    return vertex_normal

def perspective_from_image(fov: float, width: int, height: int, near: float, far: float) -> torch.Tensor:
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
    return torch.inverse(extrinsic) @ torch.diag([1, -1, -1, 1]).to(extrinsic)

def view_to_extrinsic(view: torch.Tensor) -> torch.Tensor:
    """OpenCV convention camera extrinsic to OpenGL convention view matrix

    Args:
        view (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix

    Returns:
        torch.Tensor: shape (4, 4) or (..., 4, 4) OpenCV convention camera extrinsic
    """
    return torch.diag([1, -1, -1, 1]).to(view) @ torch.inverse(view)

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

def image_uv(width: int, height: int):
    return torch.from_numpy(__image_uv(width, height))

def image_mesh(width: int, height: int, mask: torch.Tensor = None):
    uv, faces = __image_mesh(width, height, mask.cpu().numpy() if mask is not None else None)
    uv, faces = torch.from_numpy(uv), torch.from_numpy(faces)
    if mask is not None:
        uv, faces= uv.to(mask.device), faces.to(mask.device)
    return uv, faces

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

