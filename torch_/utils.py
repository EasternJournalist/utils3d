import torch
from typing import Tuple, Union
from numbers import Number

from ..numpy_.utils import (
    perspective_from_fov as __perspective_from_fov, 
    perspective_from_fov_xy as __perspective_from_fov_xy,
    image_uv as __image_uv,
    image_mesh as __image_mesh,
    to_linear_depth as __to_linear_depth,
    to_depth_buffer as __to_depth_buffer,
    chessboard as __chessboard
)

def to_linear_depth(depth_buffer: torch.Tensor) -> torch.Tensor:
    return __to_linear_depth(depth_buffer)

def to_depth_buffer(linear_depth: torch.Tensor) -> torch.Tensor:
    return __to_depth_buffer(linear_depth)



def image_uv(width: int, height: int):
    return torch.from_numpy(__image_uv(width, height))

def image_mesh(width: int, height: int, mask: torch.Tensor = None):
    uv, faces = __image_mesh(width, height, mask.cpu().numpy() if mask is not None else None)
    uv, faces = torch.from_numpy(uv), torch.from_numpy(faces)
    if mask is not None:
        uv, faces= uv.to(mask.device), faces.to(mask.device)
    return uv, faces



def chessboard(width: int, height: int, grid_size: int, color_a: torch.Tensor, color_b: torch.Tensor) -> torch.Tensor:
    """get a chessboard image

    Args:
        width (int): image width
        height (int): image height
        grid_size (int): size of chessboard grid
        color_a (torch.Tensor): shape (chanenls,), color of the grid at the top-left corner
        color_b (torch.Tensor): shape (chanenls,), color in complementary grids

    Returns:
        image (np.ndarray): shape (height, width, channels), chessboard image
    """
    x = torch.div(torch.arange(width), grid_size, rounding_mode='floor')
    y = torch.div(torch.arange(height), grid_size, rounding_mode='floor')
    mask = ((x[None, :] + y[:, None]) % 2).to(color_a)
    image = (1 - mask[..., None]) * color_a + mask[..., None] * color_b
    return image