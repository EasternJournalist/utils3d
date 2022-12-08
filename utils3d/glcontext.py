from typing import Tuple
import numpy as np
import moderngl

from .numpy_.utils import image_mesh, image_uv
from .numpy_.mesh import triangulate

__all__ = ['GLContext']

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
        return 65536
    else:
        return 1

class GLContext:
    def __init__(self, standalone: bool = True, backend: str = None):
        if backend is None:
            self.mgl_ctx = moderngl.create_context(standalone=standalone)
        else:
            self.mgl_ctx = moderngl.create_context(standalone=standalone, backend=backend)

        self.__prog_rast_bary = self.mgl_ctx.program(
            vertex_shader='''
            #version 330 core

            in vec4 in_vert;
            in vec3 in_bary;
            out vec3 v_bary;

            uniform mat4 transform;

            void main() {
                gl_Position = transform * in_vert;
                v_bary = in_bary; 
            }
            ''',
            fragment_shader='''
            #version 330 core

            in vec3 v_bary;
            out int tri_id;
            out vec3 f_bary;
            
            void main() {
                tri_id = gl_PrimitiveID;
                f_bary = v_bary;
            }
            '''
        )

        self.__prog_rast_attr = self.mgl_ctx.program(
            vertex_shader='''
            #version 330 core

            in vec4 in_vert;
            in vec4 in_attr;
            out vec4 v_attr;

            uniform mat4 transform;

            void main() {
                gl_Position = transform * in_vert;
                v_attr = in_attr; 
            }
            ''',
            fragment_shader='''
            #version 330 core

            in vec4 v_attr;
            out vec4 f_attr;
            void main() {
                f_attr = v_attr;
            }
            '''
        )

        self.__prog_tex = self.mgl_ctx.program(
            vertex_shader='''
            #version 330 core

            in vec2 in_vert;
            out vec2 scr_coord;

            void main() {
                scr_coord = in_vert * 0.5 + 0.5;
                gl_Position = vec4(in_vert, 0., 1.);
            }
            ''',
            fragment_shader='''
            #version 330

            uniform sampler2D tex;
            uniform sampler2D uv;
            
            in vec2 scr_coord;
            out vec4 tex_color;

            void main() {
                tex_color = texture(tex, texture(uv, scr_coord).xy);
            }
            '''
        )
        self.__prog_tex['tex'] = 0
        self.__prog_tex['uv'] = 1

        self.__prog_flow = self.mgl_ctx.program(
            vertex_shader='''
            #version 330 core

            in vec4 in_vert_src;
            in vec4 in_vert_tgt;
            out vec4 src_pos;
            out vec4 tgt_pos;
            uniform mat4 transform_src;
            uniform mat4 transform_tgt;

            void main() {
                src_pos = transform_src * in_vert_src;
                tgt_pos = transform_tgt * in_vert_tgt; 
                gl_Position = src_pos;
            }
            ''',
            fragment_shader='''
            #version 330 core

            in vec4 src_pos;
            in vec4 tgt_pos;
            out vec4 flow;

            uniform float threshold;
            uniform sampler2D tgt_depth;

            void main() {
                vec3 src_pos_ndc = src_pos.xyz / src_pos.w;
                vec3 src_pos_scr = src_pos_ndc * 0.5 + 0.5;
                vec3 tgt_pos_ndc = tgt_pos.xyz / tgt_pos.w;
                vec3 tgt_pos_scr = tgt_pos_ndc * 0.5 + 0.5;

                float visible = tgt_pos_scr.z < texture(tgt_depth, tgt_pos_scr.xy).x + threshold ? 1 : 0;
                flow = vec4(tgt_pos_scr.xy - src_pos_scr.xy, tgt_pos.w - src_pos.w, visible);
            }
            '''
        )
        self.__prog_flow['tgt_depth'] = 0

        self.__prog_rast_tex = self.mgl_ctx.program(
            vertex_shader='''
            #version 330 core

            in vec4 in_vert;
            in vec2 in_uv;
            out vec2 uv;
            uniform mat4 transform;

            void main() {
                uv = in_uv;
                gl_Position = transform * in_vert;
            }
            ''',
            fragment_shader='''
            #version 330 core

            in vec2 uv;
            out vec4 color;

            uniform sampler2D tex;

            void main() {
                color = texture(tex, uv);
            }
            '''
        )
        self.__prog_rast_tex['tex'] = 0

        self.__program_warp__ = self.mgl_ctx.program(
            vertex_shader='''
            #version 330 core

            in vec2 in_uv;
            out vec2 uv;
            uniform sampler2D pixel_positions;
            uniform mat4 transform;

            void main() {
                uv = in_uv;
                gl_Position = transform * texture(pixel_positions, in_uv);
            }
            ''',
            fragment_shader='''
            #version 330 core

            in vec2 uv;
            out vec4 color;

            uniform sampler2D image;

            void main() {
                color = texture(image, uv);
            }
            '''
        )
        self.__program_warp__['pixel_positions'] = 0
        self.__program_warp__['image'] = 1

        self.screen_quad_vbo = self.mgl_ctx.buffer(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype='f4'))
        self.screen_quad_ibo = self.mgl_ctx.buffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))
        self.screen_quad_vao = self.mgl_ctx.vertex_array(self.__prog_tex, [(self.screen_quad_vbo, '2f4', 'in_vert')], index_buffer=self.screen_quad_ibo, index_element_size=4)

    def rasterize_barycentric(self, 
        width: int, 
        height: int, 
        vertices: np.ndarray, 
        faces: np.ndarray = None,  
        transform: np.ndarray = None, 
        *,
        bary_buffer: np.ndarray = None,
        tri_id_buffer: np.ndarray = None,
        depth_buffer: np.ndarray = None,
        alpha_blend: bool = False,
        cull_backface: bool = False,
        wireframe: bool = False
    ):
        """
        Rasterize triangles onto image with attributes attached to each vertex.\n

        Args:
            width (int): result image width
            height (int): result image height
            vertices (np.ndarray): Vertex positions of shape (N, M), M = 2, 3 or 4. `transform` can be applied convert vertex positions to clip space. If the dimension of vertices coordinate is less than 4, coordinates `z` will be padded with `0` and `w` will be padded with `1`. Note the dtype will always be converted to `float32`.
            faces (np.ndarray): shape (T, 3), triangular faces vertuces
            transform (np.ndarray, optional): row major matrix of shape (4, 4). Transform matrix to multiplicate with vertices and convert them into clip space coordinates. (Usually the Projection * View * Model). Defaults to None, that is to use identity matrix.
            image_buffer (np.ndarray, optional): The initial image to draw on. By default the image is initialized with zeros.
            depth_buffer (np.ndarray, optional): The initial depth to draw on. By default the depth is initialized with ones (infinitely far).
            alpha_blend (bool, optional): whether to enable alpha blend. Defaults to False.
            cull_backface (bool, optional): whether to enable culling backface. Defaults to False.

        Returns:
            bary_buffer (np.ndarray): float32 array of shape (height, width, 3), barycentric coordinates of each pixel (fragment) to the corresponding triangle.
            tri_id_buffer (np.ndarray): int32 array of shape (height, width), triangle indices. 
            depth_buffer (np.ndarray): float32 array of shape (height, width). The depth buffer in screen space ranging from 0 to 1; 0 is the near plane, and 1 is the far plane. If you want the linear depth in view space (z value), you can use 'to_linear_depth'
        """
        assert len(vertices.shape) == 2 
        assert 2 <= vertices.shape[1] <= 4
        assert len(faces.shape) == 2 and faces.shape[1] == 3
        if bary_buffer is not None:
            assert bary_buffer.dtype == np.float32 and len(bary_buffer.shape) == 3 and bary_buffer.shape[:2] == (height, width)
        if tri_id_buffer is not None:
            assert tri_id_buffer.dtype == np.int32 and len(tri_id_buffer.shape) == 2 and tri_id_buffer.shape[:2] == (height, width)
        if depth_buffer is not None:
            assert depth_buffer.dtype == np.float32 and len(depth_buffer.shape) == 2 and depth_buffer.shape[:2] == (height, width)
        
        # Pad vertices
        n_vertices = vertices.shape[0]
        if vertices.shape[1] == 3:
            vertices = np.concatenate([vertices, np.ones((n_vertices, 1))], axis=-1)
        elif vertices.shape[1] == 2:
            vertices = np.concatenate([vertices, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=-1)

        # Flatten vertices and create input barycentric coordinates
        if faces is not None:
            vertices = vertices[faces.ravel()]
        in_bary = np.zeros((vertices.shape[0], 3)).reshape((-1, 3, 3))
        for i in range(3):
            in_bary[:, i, i] = 1

        # Create vertex array
        vbo_vert = self.mgl_ctx.buffer(vertices.astype('f4', copy=False))
        vbo_bary = self.mgl_ctx.buffer(in_bary.astype('f4', copy=False))
        vao = self.mgl_ctx.vertex_array(self.__prog_rast_bary, [(vbo_vert, '4f4', 'in_vert'), (vbo_bary, '3f4', 'in_bary')])

        # Create frame buffer & textrue
        if bary_buffer is None:
            bary_buffer = np.zeros((height, width, 3), dtype='f4')
        if tri_id_buffer is None:
            tri_id_buffer = np.full((height, width), -1, dtype='i4')
        if depth_buffer is None:
            depth_buffer = np.ones((height, width), dtype='f4')
        bary_tex = self.mgl_ctx.texture((width, height), 3, dtype='f4', data=bary_buffer)
        tri_id_tex = self.mgl_ctx.texture((width, height), 1, dtype='i4', data=tri_id_buffer)
        depth_tex = self.mgl_ctx.depth_texture((width, height), data=depth_buffer)
        fbo = self.mgl_ctx.framebuffer(color_attachments=[tri_id_tex, bary_tex], depth_attachment=depth_tex)

        # Set transform matrix
        self.__prog_rast_bary['transform'].write(transform.transpose().copy().astype('f4') if transform is not None else np.eye(4, 4, dtype='f4'))

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.mgl_ctx.depth_func = '<'
        self.mgl_ctx.enable(self.mgl_ctx.DEPTH_TEST)
        if cull_backface:
            self.mgl_ctx.enable(self.mgl_ctx.CULL_FACE)
        else:
            self.mgl_ctx.disable(self.mgl_ctx.CULL_FACE)
        if alpha_blend:
            self.mgl_ctx.enable(self.mgl_ctx.BLEND)
        else:
            self.mgl_ctx.disable(self.mgl_ctx.BLEND)
        self.mgl_ctx.wireframe = wireframe
        vao.render(moderngl.TRIANGLES)
        self.mgl_ctx.disable(self.mgl_ctx.DEPTH_TEST)
        
        # Read result
        bary_tex.read_into(bary_buffer)
        tri_id_tex.read_into(tri_id_buffer)
        depth_tex.read_into(depth_buffer)

        # Release
        vao.release()
        vbo_vert.release()
        fbo.release()
        bary_tex.release()
        tri_id_tex.release()
        depth_tex.release()

        return bary_buffer, tri_id_buffer, depth_buffer

    def rasterize_attribute(self, 
        width: int, 
        height: int, 
        vertices: np.ndarray, 
        attributes: np.ndarray, 
        faces: np.ndarray = None,  
        transform: np.ndarray = None,
        *, 
        image_buffer: np.ndarray = None, 
        depth_buffer: np.ndarray = None,
        alpha_blend: bool = False,
        cull_backface: bool = False,
        wireframe: bool = False
    ):
        """
        Rasterize triangles onto image with attributes attached to each vertex.\n

        Args:
            width (int): result image width
            height (int): result image height
            vertices (np.ndarray): Vertex positions of shape (N, M), M = 2, 3 or 4. `transform` can be applied convert vertex positions to clip space. If the dimension of vertices coordinate is less than 4, coordinates `z` will be padded with `0` and `w` will be padded with `1`. Note the dtype will always be converted to `float32`.
            attributes (np.ndarray): Vertex attrubutes shape shape (N, C), C = 1, 2, 3 or 4. 
            faces (np.ndarray): shape (T, 3), triangular faces vertuces
            transform (np.ndarray, optional): row major matrix of shape (4, 4). Transform matrix to multiplicate with vertices and convert them into clip space coordinates. (Usually the Projection * View * Model). Defaults to None, that is to use identity matrix.
            image_buffer (np.ndarray, optional): The initial image to draw on. By default the image is initialized with zeros.
            depth_buffer (np.ndarray, optional): The initial depth to draw on. By default the depth is initialized with ones (infinitely far).
            alpha_blend (bool, optional): whether to enable alpha blend. Defaults to False.
            cull_backface (bool, optional): whether to enable culling backface. Defaults to False.

        Returns:
            image_buffer (np.ndarray): shape (height, width, 4). The rendering result. The first M channels corresponds to M channels of attributes. If M is less than 4, the extra channels will be filled ones in the rendered areas. This makes it convenient to get the mask from the 4th channel.\n
            depth_buffer (np.ndarray): shape (height, width). The depth buffer in screen space ranging from 0 to 1; 0 is the near plane, and 1 is the far plane. If you want the linear depth in view space (z value), you can use 'to_linear_depth'
        """
        assert len(vertices.shape) == 2 and len(attributes.shape) == 2
        assert vertices.shape[0] == attributes.shape[0]
        assert 2 <= vertices.shape[1] <= 4
        assert 1 <= attributes.shape[1] <= 4
        assert len(faces.shape) == 2 and faces.shape[1] == 3
        if image_buffer is not None:
            assert image_buffer.shape[0] == height and image_buffer.shape[1] == width
        if depth_buffer is not None:
            assert depth_buffer.dtype == np.float32 and len(depth_buffer.shape) == 2 and depth_buffer.shape[0] == height and depth_buffer.shape[1] == width 
        
        # Pad vertices
        n_vertices = vertices.shape[0]
        if vertices.shape[1] == 3:
            vertices = np.concatenate([vertices, np.ones((n_vertices, 1))], axis=-1)
        elif vertices.shape[1] == 2:
            vertices = np.concatenate([vertices, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=-1)

        # Pad attributes
        attr_size = attributes.shape[1]
        attr_dtype = map_np_dtype(attributes.dtype)
        if attr_dtype not in ['f2', 'f4']:
            raise TypeError('attribute dtype unsupported')
        if attr_size < 4:
            attributes = np.concatenate([attributes, np.full((n_vertices, 4 - attr_size), one_value(attr_dtype), dtype=attr_dtype)], axis=-1)

        # Create vertex array
        vbo_vert = self.mgl_ctx.buffer(vertices.astype('f4', copy=False))
        vbo_attr = self.mgl_ctx.buffer(attributes)
        ibo = self.mgl_ctx.buffer(faces.astype('i4', copy=False)) if faces is not None else None
        if ibo is None:
            vao = self.mgl_ctx.vertex_array(self.__prog_rast_attr, [(vbo_vert, '4f4', 'in_vert'), (vbo_attr, f'4{attr_dtype}', 'in_attr')])
        else:
            vao = self.mgl_ctx.vertex_array(self.__prog_rast_attr, [(vbo_vert, '4f4', 'in_vert'), (vbo_attr, f'4{attr_dtype}', 'in_attr')], index_buffer=ibo, index_element_size=4)

        # Create frame buffer & textrue
        if image_buffer is None:
            image_buffer = np.zeros((height, width, 4), dtype=attr_dtype)
        if depth_buffer is None:
            depth_buffer = np.ones((height, width), dtype='f4')
        image_tex = self.mgl_ctx.texture((width, height), 4, dtype=attr_dtype, data=image_buffer)
        depth_tex = self.mgl_ctx.depth_texture((width, height), data=depth_buffer)
        fbo = self.mgl_ctx.framebuffer(color_attachments=[image_tex], depth_attachment=depth_tex)

        # Set transform matrix
        self.__prog_rast_attr['transform'].write(transform.transpose().copy().astype('f4') if transform is not None else np.eye(4, 4, dtype='f4'))

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.mgl_ctx.depth_func = '<'
        self.mgl_ctx.enable(self.mgl_ctx.DEPTH_TEST)
        if cull_backface:
            self.mgl_ctx.enable(self.mgl_ctx.CULL_FACE)
        else:
            self.mgl_ctx.disable(self.mgl_ctx.CULL_FACE)
        if alpha_blend:
            self.mgl_ctx.enable(self.mgl_ctx.BLEND)
        else:
            self.mgl_ctx.disable(self.mgl_ctx.BLEND)
        self.mgl_ctx.wireframe = wireframe
        vao.render(moderngl.TRIANGLES)
        self.mgl_ctx.disable(self.mgl_ctx.DEPTH_TEST)
        
        # Read result
        image_tex.read_into(image_buffer)
        depth_tex.read_into(depth_buffer)

        # Release
        vao.release()
        vbo_vert.release()
        vbo_attr.release()
        if ibo is not None: ibo.release()
        fbo.release()
        image_tex.release()
        depth_tex.release()

        return image_buffer, depth_buffer

    def texture(self, uv: np.ndarray, texture: np.ndarray, interpolation: str= 'linear', wrap: str = 'clamp'):
        """
        Given an UV image, texturing from the texture map
        """
        assert len(texture.shape) == 3 and 1 <= texture.shape[2] <= 4
        assert uv.shape[2] == 2
        height, width = uv.shape[:2]
        texture_dtype = map_np_dtype(texture.dtype)

        # Create texture, set filter and bind. TODO: min mag filter, mipmap
        texture_tex = self.mgl_ctx.texture((texture.shape[1], texture.shape[0]), texture.shape[2], dtype=texture_dtype, data=texture)
        if interpolation == 'linear':
            texture_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        elif interpolation == 'nearest':
            texture_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture_tex.use(location=0)
        texture_uv = self.mgl_ctx.texture((width, height), 2, dtype='f4', data=uv.astype('f4', copy=False))
        texture_uv.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture_uv.use(location=1)

        # Create render buffer and frame buffer
        rb = self.mgl_ctx.renderbuffer((uv.shape[1], uv.shape[0]), 4, dtype=texture_dtype)
        fbo = self.mgl_ctx.framebuffer(color_attachments=[rb])

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.mgl_ctx.disable(self.mgl_ctx.BLEND)
        self.screen_quad_vao.render()

        # Read buffer
        image_buffer = np.frombuffer(fbo.read(components=4, attachment=0, dtype=texture_dtype), dtype=texture_dtype).reshape((height, width, 4))

        # Release
        texture_tex.release()
        rb.release()
        fbo.release()

        return image_buffer[:, :, :texture.shape[2]]

    def rasterize_texture(self, 
        width: int,
        height: int,
        vertices: np.ndarray, 
        vertices_uv: np.ndarray, 
        texture: np.ndarray, 
        faces: np.ndarray = None,
        transform: np.ndarray = None, 
        interpolation: str = 'linear',
        *, 
        image_buffer: np.ndarray = None, 
        depth_buffer: np.ndarray = None,
        alpha_blend: bool = False,
        cull_backface: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rasterize a triagular mesh with texture

        Args:
            width (int): result image width
            height (int): result image height
            vertices (np.ndarray): vertex positions of shape (N, M), `M` = 2, 3 or 4. `transform` can be applied convert vertex positions to clip space. If the dimension of vertices coordinate is less than 4, coordinates `z` will be padded with `0` and `w` will be padded with `1`. Note the dtype will always be converted to `float32`.
            vertices_uv (np.ndarray): vertices UV texture coordinates fo shape (N, 2). Note the dtype will always be converted to `float32`.
            faces (np.ndarray): triangles' vertices faces of shape (T, 3). 
            texture (np.ndarray): The texture image
            transform (np.ndarray, optional): row major matrix of shape (4, 4). Transform matrix to multiplicate with vertices and convert them into clip space coordinates. (Usually the Projection * View * Model). Defaults to None, that is to use identity matrix.
            interpolation (str, optional): texture interpolation method, 'linear' or 'nearest'. Defaults to 'linear'.
            image_buffer (np.ndarray, optional): The initial image to draw on. By default the image is initialized with zeros.
            depth_buffer (np.ndarray, optional): The initial depth to draw on. By default the depth is initialized with ones (infinitely far).
            alpha_blend (bool, optional): whether to enable alpha blend. Defaults to False.
            cull_backface (bool, optional): whether to enable culling backface. Defaults to False.

        Returns:
            image_buffer (np.ndarray): shape (height, width, 4). The rendering result. The first M channels corresponds to M channels of attributes. If M is less than 4, the 3rd channel will be filled with zero and the 4th channel will be filled with 1. This makes it convenient to get the mask from the 4th channel.\n
            depth_buffer (np.ndarray): shape (height, width). The depth buffer in screen space ranging from 0 to 1; 0 is the near plane, and 1 is the far plane. If you want the linear depth in view space (z value), you can use 'to_linear_depth'
        """
        # Check data type
        assert len(vertices.shape) == 2 and len(vertices_uv.shape) == 2 and len(faces.shape) == 2 and len(texture.shape) == 3
        assert vertices.shape[0] == vertices_uv.shape[0]
        assert 2 <= vertices.shape[1] <= 4
        assert 1 <= texture.shape[2] <= 4
        if image_buffer is not None:
            assert image_buffer.shape[0] == height and image_buffer.shape[1] == width
        if depth_buffer is not None:
            assert depth_buffer.dtype == np.float32 and len(depth_buffer.shape) == 2 and depth_buffer.shape[0] == height and depth_buffer.shape[1] == width 

        # Pad vertices
        n_vertices = vertices.shape[0]
        if vertices.shape[1] == 3:
            vertices = np.concatenate([vertices, np.ones((n_vertices, 1))], axis=-1)
        elif vertices.shape[1] == 2:
            vertices = np.concatenate([vertices, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=-1)

        # Pad texture
        tex_dsize = texture.shape[2]
        tex_dtype = map_np_dtype(texture.dtype)
        if not tex_dtype:
            raise TypeError('attribute dtype unsupported')
        if tex_dsize < 4:
            texture = np.concatenate([texture, np.full((*texture.shape[:2], 4 - tex_dsize), one_value(tex_dtype), dtype=tex_dtype)], axis=-1)

        # Create vertex array
        vbo_vert = self.mgl_ctx.buffer(vertices.astype('f4', copy=False))
        vbo_uv = self.mgl_ctx.buffer(vertices_uv.astype('f4', copy=False))
        ibo = self.mgl_ctx.buffer(faces.astype('i4', copy=False)) if faces is not None else None
        if ibo is None:
            vao = self.mgl_ctx.vertex_array(self.__prog_rast_tex, [(vbo_vert, '4f4', 'in_vert'), (vbo_uv, f'2f4', 'in_uv')])
        else:
            vao = self.mgl_ctx.vertex_array(self.__prog_rast_tex, [(vbo_vert, '4f4', 'in_vert'), (vbo_uv, f'2f4', 'in_uv')], index_buffer=ibo, index_element_size=4)

        # Create texture
        texture_tex = self.mgl_ctx.texture((texture.shape[1], texture.shape[0]), 4, dtype=tex_dtype, data=texture)
        if interpolation == 'linear':
            texture_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        elif interpolation == 'nearest':
            texture_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Create frame buffer
        if image_buffer is None:
            image_buffer = np.zeros((height, width, 4), dtype=tex_dtype)
        if depth_buffer is None:
            depth_buffer = np.ones((height, width), dtype='f4')
        image_tex = self.mgl_ctx.texture((width, height), 4, dtype=tex_dtype, data=image_buffer)
        depth_tex = self.mgl_ctx.depth_texture((width, height), data=depth_buffer)
        fbo = self.mgl_ctx.framebuffer(color_attachments=[image_tex], depth_attachment=depth_tex)

        # Set transform matrix
        self.__prog_rast_tex['transform'].write(transform.transpose().copy().astype('f4') if transform is not None else np.eye(4, 4, dtype='f4'))

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.mgl_ctx.depth_func = '<'
        self.mgl_ctx.enable(self.mgl_ctx.DEPTH_TEST)
        if cull_backface:
            self.mgl_ctx.enable(self.mgl_ctx.CULL_FACE)
        else:
            self.mgl_ctx.disable(self.mgl_ctx.CULL_FACE)
        if alpha_blend:
            self.mgl_ctx.enable(self.mgl_ctx.BLEND)
        else:
            self.mgl_ctx.disable(self.mgl_ctx.BLEND)
        texture_tex.use(location=0)
        vao.render(moderngl.TRIANGLES)
        self.mgl_ctx.disable(self.mgl_ctx.DEPTH_TEST)

        # Read result
        image_tex.read_into(image_buffer)
        depth_tex.read_into(depth_buffer)

        # Release
        vao.release()
        vbo_vert.release()
        vbo_uv.release()
        ibo.release()
        texture_tex.release()
        fbo.release()
        image_tex.release()
        depth_tex.release()

        return image_buffer, depth_buffer

    def rasterize_flow(self, 
        width: int, 
        height: int, 
        vertices_source: np.ndarray, 
        vertices_target: np.ndarray, 
        faces: np.ndarray = None,  
        transform_source: np.ndarray = None,
        transform_target: np.ndarray = None, 
        target_depth_buffer: np.ndarray = None,
        threshold: float = 1e-4,  
        *,
        image_buffer: np.ndarray = None, 
        depth_buffer: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render optical flow. 

        Args:
            width (int): image width
            height (int): image height
            vertices_source (np.ndarray): source vertices positions
            vertices_target (np.ndarray): target vertices positions
            faces (np.ndarray, optional): shape (T, 3). Vertex faces of each triangle. `T` is the number of triangles. If the mesh is not trianglular mesh, `trianglate()` can help. Defaults to None.
            transform_source (np.ndarray, optional): row major matrix of shape (4, 4). Transform matrix to multiplicate with vertices and convert them into clip space coordinates. (Usually the Projection * View * Model). Defaults to None, that is to use identity matrix.
            transform_target (np.ndarray, optional): row major matrix of shape (4, 4). Transform matrix to multiplicate with vertices and convert them into clip space coordinates. (Usually the Projection * View * Model). Defaults to None, that is to use identity matrix.
            target_depth_buffer (np.ndarray, optional): _description_. Defaults to None.
            threshold (float, optional): _description_. Defaults to 1e-4.
            image_buffer (np.ndarray, optional): _description_. Defaults to None.
            depth_buffer (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            image_buffer (np.ndarray): shape (height, width, 4). The first two channels are UV flow in image space. The third channel is the depth flow in linear depth space. The last channel is the occulusion mask, where ones indicate visibility in target image.
                The flow map in normalized image space ranging [0, 1]. Note that in OpenGL the origin of screen space (i.e. viewport) is the left bottom corner. 
                If you need to display or save the image through OpenCV or Pillow, you will have to flip the Y axis of both the flow vector and image data array.
                >>> opencv_flow_image = flow_image[::-1]    # Flip image data
                >>> opencv_flow_image[:, :, 1] *= -1        # Flip flow vector
            depth_buffer (np.ndarray): shape (height, width). The depth buffer in screen space ranging from 0 to 1; 0 is the near plane, and 1 is the far plane. If you want the linear depth in view space (z value), you can use 'to_linear_depth'
        """
        assert len(vertices_source.shape) == 2 and len(vertices_target.shape) == 2
        assert vertices_source.shape == vertices_target.shape
        assert 2 <= vertices_source.shape[1] <= 4
        if target_depth_buffer is not None:
            assert target_depth_buffer.dtype == np.float32 and target_depth_buffer.shape[0] == height and target_depth_buffer.shape[1] == width
        if image_buffer is not None:
            assert image_buffer.shape[0] == height and image_buffer.shape[1] == width
        if depth_buffer is not None:
            assert depth_buffer.dtype == np.float32 and len(depth_buffer.shape) == 2 and depth_buffer.shape[0] == height and depth_buffer.shape[1] == width

        # Pad vertices
        n_vertices = vertices_source.shape[0]
        if vertices_source.shape[1] == 3:
            vertices_source = np.concatenate([vertices_source, np.ones((n_vertices, 1))], axis=-1)
            vertices_target = np.concatenate([vertices_target, np.ones((n_vertices, 1))], axis=-1)
        elif vertices_source.shape[1] == 2:
            vertices_source = np.concatenate([vertices_source, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=-1)
            vertices_target = np.concatenate([vertices_target, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=-1)

        # Create vertex array
        vbo_vert_src = self.mgl_ctx.buffer(vertices_source.astype('f4', copy=False))
        vbo_vert_tgt = self.mgl_ctx.buffer(vertices_target.astype('f4', copy=False))
        ibo = self.mgl_ctx.buffer(faces.astype('i4', copy=False)) if faces is not None else None
        if ibo is None:
            vao = self.mgl_ctx.vertex_array(self.__prog_flow, [(vbo_vert_src, '4f4', 'in_vert_src'), (vbo_vert_tgt, '4f4', 'in_vert_tgt')])
        else:
            vao = self.mgl_ctx.vertex_array(self.__prog_flow, [(vbo_vert_src, '4f4', 'in_vert_src'), (vbo_vert_tgt, '4f4', 'in_vert_tgt')], index_buffer=ibo, index_element_size=4)

        # Create texture
        if target_depth_buffer is None:
            tgt_depth_tex = self.mgl_ctx.texture((width, height), 1, dtype='f4', data=np.ones((height, width), dtype='f4'))
        else:
            tgt_depth_tex = self.mgl_ctx.texture((width, height), 1, dtype='f4', data=target_depth_buffer)

        # Create frame buffer
        if image_buffer is None:
            image_buffer = np.zeros((height, width, 4), dtype='f4')
        if depth_buffer is None:
            depth_buffer = np.ones((height, width), dtype='f4')
        image_tex = self.mgl_ctx.texture((width, height), 4, dtype='f4', data=image_buffer)
        depth_tex = self.mgl_ctx.depth_texture((width, height), data=depth_buffer)

        fbo = self.mgl_ctx.framebuffer(color_attachments=[image_tex], depth_attachment=depth_tex)
        
        # Set uniforms and bind texture
        self.__prog_flow['threshold'] = threshold
        self.__prog_flow['transform_src'].write(transform_source.transpose().copy().astype('f4') if transform_source is not None else np.eye(4, 4, dtype='f4'))
        self.__prog_flow['transform_tgt'].write(transform_target.transpose().copy().astype('f4') if transform_target is not None else np.eye(4, 4, dtype='f4'))
        tgt_depth_tex.use(location=0)

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.mgl_ctx.depth_func = '<'
        self.mgl_ctx.enable(self.mgl_ctx.DEPTH_TEST)
        self.mgl_ctx.disable(self.mgl_ctx.CULL_FACE)
        self.mgl_ctx.disable(self.mgl_ctx.BLEND)
        vao.render(moderngl.TRIANGLES)
        self.mgl_ctx.disable(self.mgl_ctx.DEPTH_TEST)

        # Read result
        image_tex.read_into(image_buffer)
        depth_tex.read_into(depth_buffer)

        # Release
        vao.release()
        vbo_vert_src.release()
        vbo_vert_tgt.release()
        ibo.release()
        fbo.release()
        image_tex.release()
        depth_tex.release()
        tgt_depth_tex.release()

        return image_buffer, depth_buffer
    
    def warp_image_3d(self, image: np.ndarray, pixel_positions: np.ndarray, transform: np.ndarray = None, interpolation: str = 'linear', alpha_blend: bool = False):
        assert len(image.shape) == 3 and len(pixel_positions.shape) == 3
        height, width, n_channels = image.shape
        assert image.shape[:2] == pixel_positions.shape[:2]
        assert 1 <= n_channels <= 4
        assert 1 <= pixel_positions.shape[2] <= 4

        image_dtype = map_np_dtype(image.dtype)
        
        # Pad pixel_positions
        if pixel_positions.shape[2] == 3:
            pixel_positions = np.concatenate([pixel_positions, np.ones((height, width, 1))], axis=-1)
        elif pixel_positions.shape[2] == 2:
            pixel_positions = np.concatenate([pixel_positions, np.zeros((height, width, 1)), np.ones((height, width, 1))], axis=-1)

        # Create vertex array
        im_uv, im_faces = image_mesh(width, height)
        vbo_uv = self.mgl_ctx.buffer(im_uv.astype('f4', copy=False))
        ibo = self.mgl_ctx.buffer(triangulate(im_faces).astype('i4', copy=False))
        vao = self.mgl_ctx.vertex_array(self.__program_warp__, [(vbo_uv, '2f4', 'in_uv')], index_buffer=ibo, index_element_size=4)

        # Create texture
        pixel_positions_tex = self.mgl_ctx.texture((width, height), 4, dtype='f4', data=pixel_positions.astype('f4', copy=False))
        pixel_positions_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        image_tex = self.mgl_ctx.texture((width, height), n_channels, dtype=image_dtype, data=image)
        if interpolation == 'linear':
            image_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        elif interpolation == 'nearest':
            image_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Create framebuffer
        rb = self.mgl_ctx.renderbuffer((width, height), 4, dtype=image_dtype)
        depth_tex = self.mgl_ctx.depth_texture((width, height))
        fbo = self.mgl_ctx.framebuffer(color_attachments=[rb], depth_attachment=depth_tex)

        # Set transform matrix
        self.__program_warp__['transform'].write(transform.transpose().copy().astype('f4') if transform is not None else np.eye(4, 4, dtype='f4'))

        # Render
        fbo.use()
        fbo.clear()
        fbo.viewport = (0, 0, width, height)
        self.mgl_ctx.depth_func = '<'
        self.mgl_ctx.enable(self.mgl_ctx.DEPTH_TEST)
        if alpha_blend:
            self.mgl_ctx.enable(self.mgl_ctx.BLEND)
        else:
            self.mgl_ctx.disable(self.mgl_ctx.BLEND)
        pixel_positions_tex.use(location=0)
        image_tex.use(location=1)
        vao.render(moderngl.TRIANGLES)
        self.mgl_ctx.disable(self.mgl_ctx.DEPTH_TEST)
        
        # Read result
        image_buffer = np.frombuffer(fbo.read(components=4, attachment=0, dtype=image_dtype), dtype=image_dtype).reshape((height, width, 4))

        # Release
        vao.release()
        vbo_uv.release()
        ibo.release()
        pixel_positions_tex.release()
        image_tex.release()
        fbo.release()
        rb.release()
        depth_tex.release()

        return image_buffer[:, :, :n_channels]

    def warp_image_by_flow(self, image: np.ndarray, flow: np.ndarray, occlusion_mask: np.ndarray = None, alpha_blend: bool = False) -> np.ndarray:
        """ Warp image by flow map. 

        Args:
            image (np.ndarray): image to warp
            flow (np.ndarray): flow map in image UV space.
            occlusion_mask (np.ndarray, optional): occulsion mask. Zeros or false indicates occluded. Defaults to None.
            alpha_blend (bool, optional): whether use alpha blend. Defaults to False.

        Returns:
            warped_image (np.ndarray): shape (height, width, channels)
        """
        assert image.shape[:2] == flow.shape[:2]
        assert flow.shape[2] == 2
        height, width, n_channels = image.shape
        assert 1 <= n_channels <= 4
        uv = image_uv(width, height)
        flow = flow.astype(np.float32)
        if occlusion_mask is not None:
            pixel_positions = np.concatenate([(uv + flow) * 2 - 1, -occlusion_mask.astype(np.float32).reshape((height, width, 1)) * 1e-2, np.ones((height, width, 1))], axis=-1)
        else:
            pixel_positions = np.concatenate([(uv + flow) * 2 - 1, np.zeros((height, width, 1), dtype=np.float32), np.ones((height, width, 1), dtype=np.float32)], axis=-1)
        return self.warp_image_3d(image, pixel_positions, alpha_blend=alpha_blend)