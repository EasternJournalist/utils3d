from typing import *

import torch
import nvdiffrast.torch as dr

from . import utils, transforms, mesh


__all__ = [
    'RastContext',
    'rasterize_vertex_attr', 
    'warp_image_by_depth'
]


class RastContext:
    """
    Create a rasterization context. Nothing but a wrapper of nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext.
    """
    def __init__(self, nvd_ctx: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext] = None, *, backend: Literal['cuda', 'gl'] = 'gl',  device: Union[str, torch.device] = None):
        if nvd_ctx is not None:
            self.nvd_ctx = nvd_ctx
            return 
        
        if backend == 'gl':
            self.nvd_ctx = dr.RasterizeGLContext(device=device)
        elif backend == 'cuda':
            self.nvd_ctx = dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f'Unknown backend: {backend}')


def rasterize_vertex_attr(
    ctx: RastContext,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr: torch.Tensor,
    width: int,
    height: int,
    model: torch.Tensor = None,
    view: torch.Tensor = None,
    perspective: torch.Tensor = None,
    antialiasing: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize a mesh with vertex attributes.

    Args:
        ctx (GLContext): rasterizer context
        vertices (np.ndarray): (B, N, 2 or 3 or 4)
        faces (torch.Tensor): (T, 3)
        attr (torch.Tensor): (B, N, C)
        width (int): width of the output image
        height (int): height of the output image
        model (torch.Tensor, optional): ([B,] 4, 4) model matrix. Defaults to None (identity).
        view (torch.Tensor, optional): ([B,] 4, 4) view matrix. Defaults to None (identity).
        perspective (torch.Tensor, optional): ([B,] 4, 4) perspective matrix. Defaults to None (identity).
        cull_backface (bool, optional): whether to cull backface. Defaults to True.
        ssaa (int, optional): super sampling anti-aliasing. Defaults to 1.

    Returns:
        image: (torch.Tensor): (B, C, H, W)
        depth: (torch.Tensor): (B, H, W) screen space depth, ranging from 0 to 1
    """
    assert vertices.ndim == 3
    assert faces.ndim == 2

    if vertices.shape[-1] == 2:
        vertices = torch.cat([vertices, torch.zeros_like(vertices[..., :1]), torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] == 3:
        vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] == 4:
        pass
    else:
        raise ValueError(f'Wrong shape of vertices: {vertices.shape}')
    
    mvp = perspective if perspective is not None else torch.eye(4).to(vertices)
    if view is not None:
        mvp = mvp @ view
    if model is not None:
        mvp = mvp @ model
    
    pos_clip = vertices @ mvp.transpose(-1, -2)
    
    rast_out, rast_db = dr.rasterize(ctx.nvd_ctx, pos_clip, faces, resolution=[height, width], grad_db=True)
    image, image_dr = dr.interpolate(attr, rast_out, faces, rast_db)
    if antialiasing:
        image = dr.antialias(image, rast_out, pos_clip, faces)
    image = image.flip(1).permute(0, 3, 1, 2)
    
    depth = rast_out[..., 2]
    return image, depth


def warp_image_by_depth(
    ctx: RastContext,
    depth: torch.Tensor,
    image: torch.Tensor = None,
    width: int = None,
    height: int = None,
    *,
    extrinsic_src: torch.Tensor = None,
    extrinsic_tgt: torch.Tensor = None,
    intrinsic_src: torch.Tensor = None,
    intrinsic_tgt: torch.Tensor = None,
    near: float = 0.1,
    far: float = 100.0,
    antialiasing: bool = True,
    backslash: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp image by depth. You may provide either view and perspective (OpenGL convention), or extrinsic and intrinsic (OpenCV convention).
    NOTE: if batch size is 1, image mesh will be triangulated aware of the depth, yielding less distorted results.
    Otherwise, image mesh will be triangulated simply for batch rendering.

    Args:
        ctx (Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]): rasterization context
        depth (torch.Tensor): [B, H, W] linear depth
        image (torch.Tensor): [B, C, H, W]. None to use image space uv. Defaults to None.
        width (int, optional): width of the output image. None to use the same as depth. Defaults to None.
        height (int, optional): height of the output image. Defaults the same as depth..
        extrinsic_src (torch.Tensor, optional): extrinsic matrix for source. None to use identity. Defaults to None.
        extrinsic_tgt (torch.Tensor, optional): extrinsic matrix for target. None to use identity. Defaults to None.
        intrinsic_src (torch.Tensor, optional): intrinsic matrix for source. None to use the same as target. Defaults to None.
        intrinsic_tgt (torch.Tensor, optional): intrinsic matrix for target. None to use the same as source. Defaults to None.
        near (float, optional): near plane. Defaults to 0.1. 
        far (float, optional): far plane. Defaults to 100.0.
    """
    assert depth.ndim == 3
    batch_size = depth.shape[0]

    if width is None:
        width = depth.shape[-1]
    if height is None:
        height = depth.shape[-2]
    if image is not None:
        assert image.shape[-2:] == depth.shape[-2:], f'Shape of image {image.shape} does not match shape of depth {depth.shape}'

    if extrinsic_src is None:
        extrinsic_src = torch.eye(4).to(depth)
    if extrinsic_tgt is None:
        extrinsic_tgt = torch.eye(4).to(depth)
    if intrinsic_src is None:
        intrinsic_src = intrinsic_tgt
    if intrinsic_tgt is None:
        intrinsic_tgt = intrinsic_src
    
    assert all(x is not None for x in [extrinsic_src, extrinsic_tgt, intrinsic_src, intrinsic_tgt]), "Make sure you have provided all the necessary camera parameters."

    view_tgt = transforms.extrinsic_to_view(extrinsic_tgt)
    perspective_tgt = transforms.intrinsic_to_perspective(intrinsic_tgt, near=near, far=far)
        
    uv, faces = utils.image_mesh(width=depth.shape[-1], height=depth.shape[-2])
    uv, faces = uv.to(depth.device), faces.to(depth.device)
    pts = transforms.unproject_cv(
        uv,
        depth.flatten(-2, -1),
        extrinsic_src,
        intrinsic_src,
    )

    # triangulate
    if batch_size == 1:
        faces = mesh.triangulate(faces, vertices=pts[0])
    else:
        faces = mesh.triangulate(faces, backslash=backslash)

    # rasterize attributes
    if image is not None:
        attr = image.permute(0, 2, 3, 1).flatten(1, 2)
    else:
        attr = uv.expand(batch_size, -1, -1)

    return rasterize_vertex_attr(
        ctx,
        pts,
        faces,
        attr,
        width,
        height,
        view=view_tgt,
        perspective=perspective_tgt,
        antialiasing=antialiasing
    )

