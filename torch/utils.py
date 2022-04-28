import torch
from typing import Tuple

from ..numpy.utils import (
    perspective_from_image as __perspective_from_image, 
    perspective_from_fov_xy as __perspective_from_fov_xy,
    image_uv as __image_uv
)

def perspective_from_image(fov: float, width: int, height: int, near: float, far: float) -> torch.Tensor:
    return torch.from_numpy(__perspective_from_image(fov, width, height, near, far))

def perspective_from_fov_xy(fov_x: float, fov_y: float, near: float, far: float) -> torch.Tensor:
    return torch.from_numpy(__perspective_from_fov_xy(fov_x, fov_y, near, far))

def image_uv(width: int, height: int):
    return torch.from_numpy(__image_uv(width, height))

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