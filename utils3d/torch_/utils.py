import torch
from typing import *

from ..numpy_.utils import (
    image_uv as __image_uv,
    image_mesh as __image_mesh,
)
from . import transforms
from . import mesh


__all__ = [
    'image_uv',
    'image_mesh',
    'chessboard',
    'depth_edge',
    'image_mesh_from_depth',
    'depth_to_normal'
]


def image_uv(width: int, height: int, left: int = None, top: int = None, right: int = None, bottom: int = None):
    return torch.from_numpy(__image_uv(width, height, left, top, right, bottom))


def image_mesh(width: int, height: int, mask: torch.Tensor = None):
    uv, faces = __image_mesh(width, height, mask.cpu().numpy() if mask is not None else None)
    uv, faces = torch.from_numpy(uv), torch.from_numpy(faces)
    if mask is not None:
        uv, faces= uv.to(mask.device), faces.to(mask.device)
    return uv, faces


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, slope_tol: float = None, intrinsics: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute edge map from depth map. 
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance
        slope_tol (float): slope tolerance, in radians
        intrinsics (torch.Tensor): shape (..., 3, 3), intrinsic matrix, used to compute slope tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    diff_x = (depth[:, :-1, :] - depth[:, 1:, :]).abs()
    diff_x = torch.cat([
        diff_x[:, :1, :],
        torch.maximum(diff_x[:, :-1, :], diff_x[:, 1:, :]),
        diff_x[:, -1:, :],
    ], dim=-2)
    diff_y = (depth[:, :, :-1] - depth[:, :, 1:]).abs()
    diff_y = torch.cat([
        diff_y[:, :, :1],
        torch.maximum(diff_y[:, :, :-1], diff_y[:, :, 1:]),
        diff_y[:, :, -1:],
    ], dim=-1)
    diff = torch.maximum(diff_x, diff_y)

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    if slope_tol is not None:
        pixel_width, pixel_height = (torch.inverse(intrinsics[..., :2, :2]) @ torch.tensor([1 / depth.shape[-1], 1 / depth.shape[-2]], dtype=depth.dtype, device=depth.device)).unbind(dim=-1)
        pixel_width, pixel_height = pixel_width[..., None, None], pixel_height[..., None, None]
        tan_slope = torch.maximum(diff_x / (pixel_width * depth), diff_y / (pixel_height * depth))
        edge |= tan_slope > torch.tan(torch.tensor(slope_tol, dtype=depth.dtype, device=depth.device))
    return edge


def image_mesh_from_depth(
    depth: torch.Tensor,
    extrinsic: torch.Tensor = None,
    intrinsic: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = depth.shape
    image_uv, image_mesh = image_mesh(width, height)
    image_mesh = image_mesh.reshape(-1, 4)
    depth = depth.reshape(-1)
    pts = transforms.unproject_cv(image_uv, depth, extrinsic, intrinsic)
    image_mesh = mesh.triangulate(image_mesh, vertices=pts)
    return pts, image_mesh


def depth_to_normal(depth: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        intrinsic (torch.Tensor): shape (..., 3, 3), intrinsic matrix
    Returns:
        normal (torch.Tensor): shape (..., 3, height, width), normal map
    """
    height, width = depth.shape[-2:]
    uv, faces = image_mesh(width, height)
    faces = mesh.triangulate(faces)
    uv = uv.reshape(-1, 2).to(depth)
    depth = depth.flatten(-2)
    pts = transforms.unproject_cv(uv, depth, intrinsic=intrinsic, extrinsic=transforms.view_to_extrinsic(torch.eye(4).to(depth)))
    normal = mesh.compute_vertex_normal(pts, faces.to(pts.device))
    return normal.reshape(*depth.shape[:-1], height, width, 3).permute(0, 3, 1, 2)


def chessboard(width: int, height: int, grid_size: int, color_a: torch.Tensor, color_b: torch.Tensor) -> torch.Tensor:
    """get a chessboard image

    Args:
        width (int): image width
        height (int): image height
        grid_size (int): size of chessboard grid
        color_a (torch.Tensor): shape (chanenls,), color of the grid at the top-left corner
        color_b (torch.Tensor): shape (chanenls,), color in complementary grids

    Returns:
        image (torch.Tensor): shape (height, width, channels), chessboard image
    """
    x = torch.div(torch.arange(width), grid_size, rounding_mode='floor')
    y = torch.div(torch.arange(height), grid_size, rounding_mode='floor')
    mask = ((x[None, :] + y[:, None]) % 2).to(color_a)
    image = (1 - mask[..., None]) * color_a + mask[..., None] * color_b
    return image