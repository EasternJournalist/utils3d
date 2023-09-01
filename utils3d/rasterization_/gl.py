import os
import numpy as np
from types import *
import moderngl


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
    

class GLContext:
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

    def __prog_vertex_attribute(self, n: int) -> moderngl.Program:
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

    def rasterize_vertex_attr(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
            attr: np.ndarray,
            width: int,
            height: int,
            mvp: np.ndarray = None,
            cull_backface: bool = True,
            ssaa: int = 1,
        ) -> np.ndarray:
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
            np.ndarray: [H, W, 2]
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert attr.ndim == 2 and attr.shape[1] in [1, 2, 3, 4], 'vertex attribute only supports channels 1, 2, 3, 4, but got {}'.format(attr.shape)
        assert vertices.shape[0] == attr.shape[0]
        assert vertices.dtype == np.float32
        assert faces.dtype == np.uint32 or faces.dtype == np.int32
        assert attr.dtype == np.float32

        C = attr.shape[1]
        prog = self.__prog_vertex_attribute(C)

        # Create buffers
        ibo = self.mgl_ctx.buffer(np.ascontiguousarray(faces, dtype='i4'))
        vbo_vertices = self.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
        vbo_attr = self.mgl_ctx.buffer(np.ascontiguousarray(attr, dtype='f4'))
        vao = self.mgl_ctx.vertex_array(
            prog,
            [
                (vbo_vertices, '3f', 'i_position'),
                (vbo_attr, f'{C}f', 'i_attr'),
            ],
            ibo,
        )

        # Create framebuffer
        width, height = width * ssaa, height * ssaa
        attr_tex = self.mgl_ctx.texture((width, height), C, dtype='f4')
        depth_tex = self.mgl_ctx.depth_texture((width, height))
        fbo = self.mgl_ctx.framebuffer(
            color_attachments=[attr_tex],
            depth_attachment=depth_tex,
        )

        # Render
        prog['u_mvp'].write(mvp.transpose().copy().astype('f4') if mvp is not None else np.eye(4, 4, dtype='f4'))
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.mgl_ctx.depth_func = '<'
        self.mgl_ctx.enable(self.mgl_ctx.DEPTH_TEST)
        if cull_backface:
            self.mgl_ctx.enable(self.mgl_ctx.CULL_FACE)
        else:
            self.mgl_ctx.disable(self.mgl_ctx.CULL_FACE)
        vao.render(moderngl.TRIANGLES)
        self.mgl_ctx.disable(self.mgl_ctx.DEPTH_TEST)

        # Read
        attr_map = np.zeros((height, width, C), dtype='f4')
        attr_tex.read_into(attr_map)
        if ssaa > 1:
            attr_map = attr_map.reshape(height // ssaa, ssaa, width // ssaa, ssaa, C).mean(axis=(1, 3))
        attr_map = attr_map[::-1, :, :]

        # Release
        vao.release()
        ibo.release()
        vbo_vertices.release()
        vbo_attr.release()
        fbo.release()
        attr_tex.release()
        depth_tex.release()

        return attr_map





