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
    'image_mesh_with_depth',
]


def image_uv(width: int, height: int, left: int = None, top: int = None, right: int = None, bottom: int = None):
    return torch.from_numpy(__image_uv(width, height, left, top, right, bottom))


def image_mesh(width: int, height: int, mask: torch.Tensor = None):
    uv, faces = __image_mesh(width, height, mask.cpu().numpy() if mask is not None else None)
    uv, faces = torch.from_numpy(uv), torch.from_numpy(faces)
    if mask is not None:
        uv, faces= uv.to(mask.device), faces.to(mask.device)
    return uv, faces


def image_mesh_with_depth(
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