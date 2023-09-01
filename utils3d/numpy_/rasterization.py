from typing import *
import numpy as np
from ..rasterization_ import GLContext
from . import *


def rasterize_vertex_attr(
    ctx: GLContext,
    vertices: np.ndarray,
    faces: np.ndarray,
    attr: np.ndarray,
    width: int,
    height: int,
    model: np.ndarray = None,
    view: np.ndarray = None,
    perspective: np.ndarray = None,
    cull_backface: bool = True,
    ssaa: int = 1,
) -> np.ndarray:
    """
    Rasterize a mesh with uv coordinates.

    Args:
        ctx (GLContext): rasterizer context
        vertices (np.ndarray): [N, 3]
        faces (np.ndarray): [T, 3]
        attr (np.ndarray): [N, C]
        width (int): width of the output image
        height (int): height of the output image
        model (np.ndarray, optional): model matrix. Defaults to None (identity).
        view (np.ndarray, optional): view matrix. Defaults to None (identity).
        perspective (np.ndarray, optional): perspective matrix. Defaults to None (identity).
        cull_backface (bool, optional): whether to cull backface. Defaults to True.
        ssaa (int, optional): super sampling anti-aliasing. Defaults to 1.

    Returns:
        (np.ndarray): [H, W, C]
    """
    mvp = perspective if perspective is not None else np.eye(4)
    if view is not None:
        mvp = mvp @ view
    if model is not None:
        mvp = mvp @ model
    return ctx.rasterize_vertex_attr(
        vertices, faces, attr, width, height, mvp=mvp, cull_backface=cull_backface, ssaa=ssaa
    )


def warp_image_by_depth(
    ctx: GLContext,
    depth: np.ndarray,
    image: np.ndarray = None,
    width: int = None,
    height: int = None,
    *,
    extrinsic_src: np.ndarray = None,
    extrinsic_tgt: np.ndarray = None,
    intrinsic_src: np.ndarray = None,
    intrinsic_tgt: np.ndarray = None,
    near: float = 0.1,
    far: float = 100.0,
    cull_backface: bool = True,
    ssaa: int = 1,
) -> np.ndarray:
    """
    Warp image by depth map.

    Args:
        ctx (GLContext): rasterizer context
        depth (np.ndarray): [H, W]
        image (np.ndarray, optional): [H, W, C]. The image to warp. Defaults to None (use uv coordinates).
        width (int, optional): width of the output image. None to use depth map width. Defaults to None.
        height (int, optional): height of the output image. None to use depth map height. Defaults to None.
        extrinsic_src (np.ndarray, optional): extrinsic matrix of the source camera. Defaults to None (identity).
        extrinsic_tgt (np.ndarray, optional): extrinsic matrix of the target camera. Defaults to None (identity).
        intrinsic_src (np.ndarray, optional): intrinsic matrix of the source camera. Defaults to None (use the same as intrinsic_tgt).
        intrinsic_tgt (np.ndarray, optional): intrinsic matrix of the target camera. Defaults to None (use the same as intrinsic_src).
        cull_backface (bool, optional): whether to cull backface. Defaults to True.
        ssaa (int, optional): super sampling anti-aliasing. Defaults to 1.
    """
    assert depth.ndim == 2

    if width is None:
        width = depth.shape[1]
    if height is None:
        height = depth.shape[0]
    if image is not None:
        assert image.shape[-2:] == depth.shape[-2:], f'Shape of image {image.shape} does not match shape of depth {depth.shape}'

    # set up default camera parameters
    if extrinsic_src is None:
        extrinsic_src = np.eye(4)
    if extrinsic_tgt is None:
        extrinsic_tgt = np.eye(4)
    if intrinsic_src is None:
        intrinsic_src = intrinsic_tgt
    if intrinsic_tgt is None:
        intrinsic_tgt = intrinsic_src
    
    assert all(x is not None for x in [extrinsic_src, extrinsic_tgt, intrinsic_src, intrinsic_tgt]), "Make sure you have provided all the necessary camera parameters."

    # check shapes
    assert extrinsic_src.shape == (4, 4) and extrinsic_tgt.shape == (4, 4)
    assert intrinsic_src.shape == (3, 3) and intrinsic_tgt.shape == (3, 3) 

    # convert to view and perspective matrices
    view_tgt = transforms.extrinsic_to_view(extrinsic_tgt)
    perspective_tgt = transforms.intrinsic_to_perspective(intrinsic_tgt, near=near, far=far)

    # unproject depth map
    uv, faces = utils.image_mesh(width=depth.shape[-1], height=depth.shape[-2])
    pts = transforms.unproject_cv(uv, depth.reshape(-1), extrinsic_src, intrinsic_src)
    faces = mesh.triangulate(faces, vertices=pts)

    # rasterize attributes
    if image is not None:
        attr = image.reshape(-1, image.shape[-1])
    else:
        attr = uv

    return rasterize_vertex_attr(
        ctx,
        pts,
        faces,
        attr,
        width,
        height,
        view=view_tgt,
        perspective=perspective_tgt,
        cull_backface=cull_backface,
        ssaa=ssaa,
    )