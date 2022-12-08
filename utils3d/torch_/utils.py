import torch

from ..numpy_.utils import (
    image_uv as __image_uv,
    image_mesh as __image_mesh,
    to_linear_depth as __to_linear_depth,
    to_screen_depth as __to_screen_depth,
)

def to_linear_depth(depth_buffer: torch.Tensor, near: float, far: float) -> torch.Tensor:
    return __to_linear_depth(depth_buffer, near, far)

def to_depth_buffer(linear_depth: torch.Tensor) -> torch.Tensor:
    return to_screen_depth(linear_depth)

def view_look_at(eye: torch.Tensor, look_at: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Return a view matrix looking at something

    Args:
        eye (torch.Tensor): shape (3,) eye position
        look_at (torch.Tensor): shape (3,) the point to look at
        up (torch.Tensor): shape (3,) head up direction (y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        view: shape (4, 4), view matrix
    """
    z = eye - look_at
    z = z / z.norm(keepdim=True)
    y = up - torch.sum(up * z, dim=-1, keepdim=True) * z
    y = y / y.norm(keepdim=True)
    x = torch.cross(y, z)
    return torch.cat([torch.stack([x, y, z, eye], axis=-1), torch.tensor([[0., 0., 0., 1.]])], axis=-2)

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