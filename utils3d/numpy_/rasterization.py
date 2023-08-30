from typing import *
import numpy as np
from ..rasterization_ import GLContext
from . import *


def rasterize_attr(
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
        model (np.ndarray, optional): model matrix. Defaults to None.
        view (np.ndarray, optional): view matrix. Defaults to None.
        perspective (np.ndarray, optional): perspective matrix. Defaults to None.
        cull_backface (bool, optional): whether to cull backface. Defaults to True.
        ssaa (int, optional): super sampling anti-aliasing. Defaults to 1.

    Returns:
        (np.ndarray): [H, W, C]
    """
    assert perspective is not None, 'perspective must be specified'
    mvp = perspective
    if view is not None:
        mvp = mvp @ view
    if model is not None:
        mvp = mvp @ model
    return ctx.rasterize_vertex_attr(
        vertices, faces, attr, width, height, mvp=mvp, cull_backface=cull_backface, ssaa=ssaa
    )


def warp_image_uv(
        ctx: GLContext,
        depth: np.ndarray,
        width: int = None,
        height: int = None,
        view: np.ndarray = None,
        perspective: np.ndarray = None,
        view_src: np.ndarray = None,
        view_tgt: np.ndarray = None,
        perspective_src: np.ndarray = None,
        perspective_tgt: np.ndarray = None,
        cull_backface: bool = True,
        ssaa: int = 1,
    ) -> np.ndarray:
    """
    Warp image uv map according to depth map.

    Args:
        ctx (GLContext): rasterizer context
        depth (np.ndarray): [H, W]
        width (int, optional): width of the output image. None to use depth map width. Defaults to None.
        height (int, optional): height of the output image. None to use depth map height. Defaults to None.
        view (np.ndarray, optional): view difference matrix between source and target.
            If specified, view_src and view_tgt will be ignored. Defaults to None.
        perspective (np.ndarray, optional): projection matrix for both source and target.
            If specified, perspective_src and perspective_tgt will be ignored. Defaults to None.
        view_src (np.ndarray, optional): view matrix for source. Defaults to None.
        view_tgt (np.ndarray, optional): view matrix for target. Defaults to None.
        perspective_src (np.ndarray, optional): projection matrix for source. Defaults to None.
        perspective_tgt (np.ndarray, optional): projection matrix for target. Defaults to None.
        cull_backface (bool, optional): whether to cull backface. Defaults to True.
        ssaa (int, optional): super sampling anti-aliasing. Defaults to 1.
    """
    assert depth.ndim == 2

    if width is None:
        width = depth.shape[1]
    if height is None:
        height = depth.shape[0]

    if view is not None:
        assert view.shape == (4, 4), f'Wrong shape of view: {view.shape}. Should be (4, 4)'
        if view_src is not None or view_tgt is not None:
            print('utils3d.rasterization_.warp_image_uv:')
            print('\tWarning: view_src and view_tgt will be ignored if view is specified')
        view_src = None
        view_tgt = view
    if perspective is not None:
        assert perspective.shape == (4, 4), f'Wrong shape of perspective: {perspective.shape}. Should be (4, 4)'
        if perspective_src is not None or perspective_tgt is not None:
            print('utils3d.rasterization_.warp_image_uv:')
            print('\tWarning: perspective_src and perspective_tgt will be ignored if perspective is specified')
        perspective_src = perspective
        perspective_tgt = perspective

    if view_src is None:
        view_src = np.eye(4)
    if view_tgt is None:
        view_tgt = np.eye(4)
    
    assert perspective_src is not None and perspective_tgt is not None, 'perspective_src and perspective_tgt must be specified'

    # check shapes
    assert view_src.shape == (4, 4), f'Wrong shape of view_src: {view_src.shape}. Should be (4, 4)'
    assert view_tgt.shape == (4, 4), f'Wrong shape of view_tgt: {view_tgt.shape}. Should be (4, 4)'
    assert perspective_src.shape == (4, 4), f'Wrong shape of perspective_src: {perspective_src.shape}. Should be (4, 4)'
    assert perspective_tgt.shape == (4, 4), f'Wrong shape of perspective_tgt: {perspective_tgt.shape}. Should be (4, 4)'

    # unproject depth map
    image_uv, image_mesh = utils.image_mesh(width, height)
    pts = transforms.unproject_cv(
        image_uv,
        depth.reshape(-1),
        transforms.view_to_extrinsic(view_src),
        transforms.perspective_to_intrinsic(perspective_src),
    )
    image_uv = image_uv.reshape(-1, 2)
    image_mesh = image_mesh.reshape(-1, 4)
    pts = pts.reshape(-1, 3)
    image_mesh = mesh.triangulate(image_mesh, vertices=pts)

    return rasterize_attr(
        ctx,
        pts,
        image_mesh,
        image_uv,
        width,
        height,
        view=view_tgt,
        perspective=perspective_tgt,
        cull_backface=cull_backface,
        ssaa=ssaa,
    )


def warp_image_attr(
        ctx: GLContext,
        attr: np.ndarray,
        depth: np.ndarray,
        width: int = None,
        height: int = None,
        view: np.ndarray = None,
        perspective: np.ndarray = None,
        view_src: np.ndarray = None,
        view_tgt: np.ndarray = None,
        perspective_src: np.ndarray = None,
        perspective_tgt: np.ndarray = None,
        cull_backface: bool = True,
        ssaa: int = 1,
    ) -> np.ndarray:
    """
    Warp image uv map according to depth map.

    Args:
        ctx (GLContext): rasterizer context
        attr (np.ndarray): [H, W, C]
        depth (np.ndarray): [H, W]
        width (int, optional): width of the output image. None to use depth map width. Defaults to None.
        height (int, optional): height of the output image. None to use depth map height. Defaults to None.
        view (np.ndarray, optional): view difference matrix between source and target.
            If specified, view_src and view_tgt will be ignored. Defaults to None.
        perspective (np.ndarray, optional): projection matrix for both source and target.
            If specified, perspective_src and perspective_tgt will be ignored. Defaults to None.
        view_src (np.ndarray, optional): view matrix for source. Defaults to None.
        view_tgt (np.ndarray, optional): view matrix for target. Defaults to None.
        perspective_src (np.ndarray, optional): projection matrix for source. Defaults to None.
        perspective_tgt (np.ndarray, optional): projection matrix for target. Defaults to None.
        cull_backface (bool, optional): whether to cull backface. Defaults to True.
        ssaa (int, optional): super sampling anti-aliasing. Defaults to 1.
    """
    assert depth.ndim == 2

    if width is None:
        width = depth.shape[1]
    if height is None:
        height = depth.shape[0]

    if view is not None:
        assert view.shape == (4, 4), f'Wrong shape of view: {view.shape}. Should be (4, 4)'
        if view_src is not None or view_tgt is not None:
            print('utils3d.rasterization_.warp_image_uv:')
            print('\tWarning: view_src and view_tgt will be ignored if view is specified')
        view_src = None
        view_tgt = view
    if perspective is not None:
        assert perspective.shape == (4, 4), f'Wrong shape of perspective: {perspective.shape}. Should be (4, 4)'
        if perspective_src is not None or perspective_tgt is not None:
            print('utils3d.rasterization_.warp_image_uv:')
            print('\tWarning: perspective_src and perspective_tgt will be ignored if perspective is specified')
        perspective_src = perspective
        perspective_tgt = perspective

    if view_src is None:
        view_src = np.eye(4)
    if view_tgt is None:
        view_tgt = np.eye(4)
    
    assert perspective_src is not None and perspective_tgt is not None, 'perspective_src and perspective_tgt must be specified'

    # check shapes
    assert view_src.shape == (4, 4), f'Wrong shape of view_src: {view_src.shape}. Should be (4, 4)'
    assert view_tgt.shape == (4, 4), f'Wrong shape of view_tgt: {view_tgt.shape}. Should be (4, 4)'
    assert perspective_src.shape == (4, 4), f'Wrong shape of perspective_src: {perspective_src.shape}. Should be (4, 4)'
    assert perspective_tgt.shape == (4, 4), f'Wrong shape of perspective_tgt: {perspective_tgt.shape}. Should be (4, 4)'

    # unproject depth map
    image_uv, image_mesh = utils.image_mesh(width, height)
    pts = transforms.unproject_cv(
        image_uv,
        depth.reshape(-1),
        transforms.view_to_extrinsic(view_src),
        transforms.perspective_to_intrinsic(perspective_src),
    )
    image_mesh = image_mesh.reshape(-1, 4)
    pts = pts.reshape(-1, 3)
    image_mesh = mesh.triangulate(image_mesh, vertices=pts)

    return rasterize_attr(
        ctx,
        pts,
        image_mesh,
        attr.reshape(-1, attr.shape[-1]),
        width,
        height,
        view=view_tgt,
        perspective=perspective_tgt,
        cull_backface=cull_backface,
        ssaa=ssaa,
    )



