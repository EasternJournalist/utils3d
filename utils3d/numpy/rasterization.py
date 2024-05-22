import os
from typing import *

import numpy as np
import moderngl

from . import transforms, utils, mesh


__all__ = [
    'RastContext',
    'rasterize_vertex_attr',
    'texture',
    'warp_image_by_depth',
]


def map_np_dtype(dtype) -> str:
    if dtype == int:
        return 'i4'
    elif dtype == np.uint8:
        return 'u1'
    elif dtype == np.uint32:
        return 'u2'
    elif dtype == np.float16:
        return 'f2'
    elif dtype == np.float32:
        return 'f4'
    

def one_value(dtype):
    if dtype == 'u1':
        return 255
    elif dtype == 'u2':
        return 65535
    else:
        return 1
    

class RastContext:
    def __init__(self, standalone: bool = True, backend: str = None, **kwargs):
        """
        Create a moderngl context.

        Args:
            standalone (bool, optional): whether to create a standalone context. Defaults to True.
            backend (str, optional): backend to use. Defaults to None.

        Keyword Args:
            See moderngl.create_context
        """
        if backend is None:
            self.mgl_ctx = moderngl.create_context(standalone=standalone, **kwargs)
        else:
            self.mgl_ctx = moderngl.create_context(standalone=standalone, backend=backend, **kwargs)

        self.__prog_src = {}
        self.__prog = {}

    def __del__(self):
        self.mgl_ctx.release()

    def screen_quad(self) -> moderngl.VertexArray:
        self.screen_quad_vbo = self.mgl_ctx.buffer(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype='f4'))
        self.screen_quad_ibo = self.mgl_ctx.buffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))

    def program_vertex_attribute(self, n: int) -> moderngl.Program:
        assert n in [1, 2, 3, 4], 'vertex attribute only supports channels 1, 2, 3, 4'

        if 'vertex_attribute_vsh' not in self.__prog_src:
            with open(os.path.join(os.path.dirname(__file__), 'shaders', 'vertex_attribute.vsh'), 'r') as f:
                self.__prog_src['vertex_attribute_vsh'] = f.read()
        if 'vertex_attribute_fsh' not in self.__prog_src:
            with open(os.path.join(os.path.dirname(__file__), 'shaders', 'vertex_attribute.fsh'), 'r') as f:
                self.__prog_src['vertex_attribute_fsh'] = f.read()
        
        if f'vertex_attribute_{n}' not in self.__prog:
            vsh = self.__prog_src['vertex_attribute_vsh'].replace('vecN', f'vec{n}')
            fsh = self.__prog_src['vertex_attribute_fsh'].replace('vecN', f'vec{n}')
            self.__prog[f'vertex_attribute_{n}'] = self.mgl_ctx.program(vertex_shader=vsh, fragment_shader=fsh)

        return self.__prog[f'vertex_attribute_{n}']

    def program_texture(self, n: int) -> moderngl.Program:
        assert n in [1, 2, 3, 4], 'texture only supports channels 1, 2, 3, 4'

        if 'texture_vsh' not in self.__prog_src:
            with open(os.path.join(os.path.dirname(__file__), 'shaders', 'texture.vsh'), 'r') as f:
                self.__prog_src['texture_vsh'] = f.read()
        if 'texture_fsh' not in self.__prog_src:
            with open(os.path.join(os.path.dirname(__file__), 'shaders', 'texture.fsh'), 'r') as f:
                self.__prog_src['texture_fsh'] = f.read()

        if f'texture_{n}' not in self.__prog:
            vsh = self.__prog_src['texture_vsh'].replace('vecN', f'vec{n}')
            fsh = self.__prog_src['texture_fsh'].replace('vecN', f'vec{n}')
            self.__prog[f'texture_{n}'] = self.mgl_ctx.program(vertex_shader=vsh, fragment_shader=fsh)
            self.__prog[f'texture_{n}']['tex'] = 0
            self.__prog[f'texture_{n}']['uv'] = 1
        
        return self.__prog[f'texture_{n}']

    
def rasterize_vertex_attr(
    ctx: RastContext,
    vertices: np.ndarray,
    faces: np.ndarray,
    attr: np.ndarray,
    width: int,
    height: int,
    model: np.ndarray = None,
    view: np.ndarray = None,
    projection: np.ndarray = None,
    cull_backface: bool = True,
    return_depth: bool = False,
    ssaa: int = 1,
) -> Tuple[np.ndarray, ...]:
    """
    Rasterize vertex attribute.

    Args:
        vertices (np.ndarray): [N, 3]
        faces (np.ndarray): [T, 3]
        attr (np.ndarray): [N, C]
        width (int): width of rendered image
        height (int): height of rendered image
        mvp (np.ndarray): [4, 4] model-view-projection matrix
        cull_backface (bool): whether to cull backface
        ssaa (int): super sampling anti-aliasing

    Returns:
        image (np.ndarray): [H, W, C] rendered image
        depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert attr.ndim == 2 and attr.shape[1] in [1, 2, 3, 4], 'vertex attribute only supports channels 1, 2, 3, 4, but got {}'.format(attr.shape)
    assert vertices.shape[0] == attr.shape[0]
    assert vertices.dtype == np.float32
    assert faces.dtype == np.uint32 or faces.dtype == np.int32
    assert attr.dtype == np.float32

    C = attr.shape[1]
    prog = ctx.program_vertex_attribute(C)

    mvp = projection if projection is not None else np.eye(4, np.float32)
    if view is not None:
        mvp = mvp @ view
    if model is not None:
        mvp = mvp @ model

    # Create buffers
    ibo = ctx.mgl_ctx.buffer(np.ascontiguousarray(faces, dtype='i4'))
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
    vbo_attr = ctx.mgl_ctx.buffer(np.ascontiguousarray(attr, dtype='f4'))
    vao = ctx.mgl_ctx.vertex_array(
        prog,
        [
            (vbo_vertices, '3f', 'i_position'),
            (vbo_attr, f'{C}f', 'i_attr'),
        ],
        ibo,
    )

    # Create framebuffer
    width, height = width * ssaa, height * ssaa
    attr_tex = ctx.mgl_ctx.texture((width, height), C, dtype='f4')
    depth_tex = ctx.mgl_ctx.depth_texture((width, height))
    fbo = ctx.mgl_ctx.framebuffer(
        color_attachments=[attr_tex],
        depth_attachment=depth_tex,
    )

    # Render
    prog['u_mvp'].write(mvp.transpose().copy().astype('f4') if mvp is not None else np.eye(4, 4, dtype='f4'))
    fbo.use()
    fbo.viewport = (0, 0, width, height)
    ctx.mgl_ctx.depth_func = '<'
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    if cull_backface:
        ctx.mgl_ctx.enable(ctx.mgl_ctx.CULL_FACE)
    else:
        ctx.mgl_ctx.disable(ctx.mgl_ctx.CULL_FACE)
    vao.render(moderngl.TRIANGLES)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.DEPTH_TEST)

    # Read
    attr_map = np.zeros((height, width, C), dtype='f4')
    attr_tex.read_into(attr_map)
    if ssaa > 1:
        attr_map = attr_map.reshape(height // ssaa, ssaa, width // ssaa, ssaa, C).mean(axis=(1, 3))
    attr_map = attr_map[::-1, :, :]
    if return_depth:
        depth_map = np.zeros((height, width), dtype='f4')
        depth_tex.read_into(depth_map)
        if ssaa > 1:
            depth_map = depth_map.reshape(height // ssaa, ssaa, width // ssaa, ssaa).mean(axis=(1, 3))
        depth_map = depth_map[::-1, :]
    else:
        depth_map = None

    # Release
    vao.release()
    ibo.release()
    vbo_vertices.release()
    vbo_attr.release()
    fbo.release()
    attr_tex.release()
    depth_tex.release()

    return attr_map, depth_map


def texture(
    ctx: RastContext,
    uv: np.ndarray,
    texture: np.ndarray,
    interpolation: str= 'linear', 
    wrap: str = 'clamp'
) -> np.ndarray:
    """
    Given an UV image, texturing from the texture map
    """
    assert len(texture.shape) == 3 and 1 <= texture.shape[2] <= 4
    assert uv.shape[2] == 2
    height, width = uv.shape[:2]
    texture_dtype = map_np_dtype(texture.dtype)

    # Create VAO
    screen_quad_vbo = ctx.mgl_ctx.buffer(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype='f4'))
    screen_quad_ibo = ctx.mgl_ctx.buffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))
    screen_quad_vao = ctx.mgl_ctx.vertex_array(ctx.program_texture(texture.shape[2]), [(screen_quad_vbo, '2f4', 'in_vert')], index_buffer=screen_quad_ibo, index_element_size=4)

    # Create texture, set filter and bind. TODO: min mag filter, mipmap
    texture_tex = ctx.mgl_ctx.texture((texture.shape[1], texture.shape[0]), texture.shape[2], dtype=texture_dtype, data=texture)
    if interpolation == 'linear':
        texture_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    elif interpolation == 'nearest':
        texture_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    texture_tex.use(location=0)
    texture_uv = ctx.mgl_ctx.texture((width, height), 2, dtype='f4', data=uv.astype('f4', copy=False))
    texture_uv.filter = (moderngl.NEAREST, moderngl.NEAREST)
    texture_uv.use(location=1)

    # Create render buffer and frame buffer
    rb = ctx.mgl_ctx.renderbuffer((uv.shape[1], uv.shape[0]), texture.shape[2], dtype=texture_dtype)
    fbo = ctx.mgl_ctx.framebuffer(color_attachments=[rb])

    # Render
    fbo.use()
    fbo.viewport = (0, 0, width, height)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.BLEND)
    screen_quad_vao.render()

    # Read buffer
    image_buffer = np.frombuffer(fbo.read(components=texture.shape[2], attachment=0, dtype=texture_dtype), dtype=texture_dtype).reshape((height, width, texture.shape[2]))

    # Release
    texture_tex.release()
    rb.release()
    fbo.release()

    return image_buffer


def warp_image_by_depth(
    ctx: RastContext,
    depth: np.ndarray,
    image: np.ndarray = None,
    width: int = None,
    height: int = None,
    *,
    extrinsics_src: np.ndarray = None,
    extrinsics_tgt: np.ndarray = None,
    intrinsics_src: np.ndarray = None,
    intrinsics_tgt: np.ndarray = None,
    near: float = 0.1,
    far: float = 100.0,
    cull_backface: bool = True,
    ssaa: int = 1,
    return_depth: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Warp image by depth map.

    Args:
        ctx (RastContext): rasterizer context
        depth (np.ndarray): [H, W]
        image (np.ndarray, optional): [H, W, C]. The image to warp. Defaults to None (use uv coordinates).
        width (int, optional): width of the output image. None to use depth map width. Defaults to None.
        height (int, optional): height of the output image. None to use depth map height. Defaults to None.
        extrinsics_src (np.ndarray, optional): extrinsics matrix of the source camera. Defaults to None (identity).
        extrinsics_tgt (np.ndarray, optional): extrinsics matrix of the target camera. Defaults to None (identity).
        intrinsics_src (np.ndarray, optional): intrinsics matrix of the source camera. Defaults to None (use the same as intrinsics_tgt).
        intrinsics_tgt (np.ndarray, optional): intrinsics matrix of the target camera. Defaults to None (use the same as intrinsics_src).
        cull_backface (bool, optional): whether to cull backface. Defaults to True.
        ssaa (int, optional): super sampling anti-aliasing. Defaults to 1.
    
    Returns:
        image (np.ndarray): [H, W, C] warped image (or uv coordinates if image is None).
        depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None.
    """
    assert depth.ndim == 2

    if width is None:
        width = depth.shape[1]
    if height is None:
        height = depth.shape[0]
    if image is not None:
        assert image.shape[-2:] == depth.shape[-2:], f'Shape of image {image.shape} does not match shape of depth {depth.shape}'

    # set up default camera parameters
    if extrinsics_src is None:
        extrinsics_src = np.eye(4)
    if extrinsics_tgt is None:
        extrinsics_tgt = np.eye(4)
    if intrinsics_src is None:
        intrinsics_src = intrinsics_tgt
    if intrinsics_tgt is None:
        intrinsics_tgt = intrinsics_src
    
    assert all(x is not None for x in [extrinsics_src, extrinsics_tgt, intrinsics_src, intrinsics_tgt]), "Make sure you have provided all the necessary camera parameters."

    # check shapes
    assert extrinsics_src.shape == (4, 4) and extrinsics_tgt.shape == (4, 4)
    assert intrinsics_src.shape == (3, 3) and intrinsics_tgt.shape == (3, 3) 

    # convert to view and perspective matrices
    view_tgt = transforms.extrinsics_to_view(extrinsics_tgt)
    perspective_tgt = transforms.intrinsics_to_perspective(intrinsics_tgt, near=near, far=far)

    # unproject depth map
    uv, faces = utils.image_mesh(*depth.shape[-2:])
    pts = transforms.unproject_cv(uv, depth.reshape(-1), extrinsics_src, intrinsics_src)
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
        mvp=perspective_tgt @ view_tgt,
        cull_backface=cull_backface,
        ssaa=ssaa,
        return_depth=return_depth,
    )


