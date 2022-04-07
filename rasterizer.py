from typing import Literal, Tuple
import numpy as np
import moderngl
import time
from .utils import *

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
    elif dtype == np.float64:
        return 'f8'

class Context:
    def __init__(self):
        # TODO: create context
        try:
            self.__ctx__ = moderngl.create_context()
        except:
            try:
                self.__ctx__ = moderngl.create_standalone_context()
            except:
                raise Exception("Fail to create context")

        self.__program_rasterize__ = self.__ctx__.program(
            vertex_shader='''
            #version 330

            in vec4 in_vert;
            in vec4 in_attr;
            out vec4 v_attr;

            uniform mat4 transform_matrix;

            void main() {
                gl_Position = transform_matrix * in_vert;
                v_attr = in_attr; 
            }
            ''',
            fragment_shader='''
            #version 330

            in vec4 v_attr;
            out vec4 f_attr;
            void main() {
                f_attr = v_attr;
            }
            '''
        )

        self.__program_texture__ = self.__ctx__.program(
            vertex_shader='''
            #version 330

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
        self.__program_texture__['tex'] = 0
        self.__program_texture__['uv'] = 1

        self.__program_flow__ = self.__ctx__.program(
            vertex_shader='''
            #version 330

            in vec4 in_vert_src;
            in vec4 in_vert_tgt;
            out vec4 src_pos;
            out vec4 tgt_pos;
            uniform mat4 transform_matrix_src;
            uniform mat4 transform_matrix_tgt;

            void main() {
                src_pos = transform_matrix_src * in_vert_src;
                tgt_pos = transform_matrix_tgt * in_vert_tgt; 
                gl_Position = src_pos;
            }
            ''',
            fragment_shader='''
            #version 330

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
        self.__program_flow__['tgt_depth'] = 0

        self.__program_rasterize_texture__ = self.__ctx__.program(
            vertex_shader='''
            #version 330

            in vec4 in_vert;
            in vec2 in_uv;
            out vec2 uv;
            uniform mat4 transform_matrix;

            void main() {
                uv = in_uv;
                gl_Position = transform_matrix * in_vert;
            }
            ''',
            fragment_shader='''
            #version 330

            in vec2 uv;
            out vec4 color;

            uniform sampler2D tex;

            void main() {
                flow = texture(tex, uv);
            }
            '''
        )
        self.__program_rasterize_texture__['tex'] = 0

        self.__program_warp__ = self.__ctx__.program(
            vertex_shader='''
            #version 330

            in vec2 in_uv;
            out vec2 uv;
            uniform sampler2D pixel_positions;
            uniform mat4 transform_matrix;

            void main() {
                uv = in_uv;
                gl_Position = transform_matrix * texture(pixel_positions, in_uv);
            }
            ''',
            fragment_shader='''
            #version 330

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

        self.screen_quad_vbo = self.__ctx__.buffer(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype='f4'))
        self.screen_quad_ibo = self.__ctx__.buffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))
        self.screen_quad_vao = self.__ctx__.vertex_array(self.__program_texture__, [(self.screen_quad_vbo, '2f4', 'in_vert')], index_buffer=self.screen_quad_ibo, index_element_size=4)

    def rasterize(self, 
        width: int, 
        height: int, 
        vertices: np.ndarray, 
        attributes: np.ndarray, 
        triangle_indices: np.ndarray = None,  
        transform_matrix: np.ndarray = None, 
        image_buffer: np.ndarray = None, 
        depth_buffer: np.ndarray = None,
        alpha_blend: bool = False,
        cull_backface: bool = False
    ):
        '''
        Rasterize triangles onto image with attributes attached to each vertex.\n
        ## Parameters

        `width` : Image width\n
        `height`: Image height\n
        `vertices`: Numpy array of shape `[N, M]`, `M` = 2, 3 or 4. Vertex positions. `transform_matrix` can be applied convert vertex positions to clip space. 
            If the dimension of vertices coordinate is less than 4, coordinates `z` will be padded with `0` and `w` will be padded with `1`.
            Note the data type will always be converted to `float32`.\n
        `attributes`: Numpy array of shape `[N, L]`, `L` = 1, 2, 3 or 4. Variouse Attribtues attached to each vertex. The output image data type will be the same as that of attributes.\n
        `triangle_indices`: Numpy array of shape `[T, 3]`. Vertex indices of each triangle. `T` is the number of triangles. If the mesh is not trianglular mesh, `trianglate()` can help.
        `transform_matrix`: Numpy array of shape `[4, 4]`. Row major matrix. Identity matrix by default if not provided. Note the data type will always be converted to `float32`.\n
        `image_buffer`: Numpy array of shape [height, width, 4]. The initial image to draw on. By default the image is initialized with zeros.\n
        `depth_buffer`: Numpy array of shape [height, width]. The initial depth to draw on. By default the depth is initialized with ones.\n

        ## Returns

        `image_buffer`: Numpy array of shape `[height, width, 4]`. The rendering result. The first M channels corresponds to M channels of attributes. 
            If M is less than 4, the 3rd channel will be filled with zero and the 4th channel will be filled with 1. This makes it convenient to get the mask from the 4th channel.\n
        `depth_buffer`: Numpy array of shape `[height, width]`. The depth buffer in window space ranging from 0 to 1; 0 is the near plane, and 1 is the far plane. If you want the linear depth in view space (z value), you can use 'to_linear_depth'
        '''
        assert len(vertices.shape) == 2 and len(attributes.shape) == 2
        assert vertices.shape[0] == attributes.shape[0]
        assert 2 <= vertices.shape[1] <= 4
        assert 1 <= attributes.shape[1] <= 4
        if image_buffer is not None:
            assert image_buffer.shape[0] == height and image_buffer.shape[1] == width
        if depth_buffer is not None:
            assert depth_buffer.dtype == np.float32 and len(depth_buffer.shape) == 2 and depth_buffer.shape[0] == height and depth_buffer.shape[1] == width 

        # Pad vertices
        n_vertices = vertices.shape[0]
        if vertices.shape[1] == 3:
            vertices = np.concatenate([vertices, np.ones((n_vertices, 1))], axis=1)
        elif vertices.shape[1] == 2:
            vertices = np.concatenate([vertices, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=1)

        # Pad attributes
        attr_size = attributes.shape[1]
        attr_dtype = map_np_dtype(attributes.dtype)
        if not attr_dtype:
            raise TypeError('attribute dtype unsupported')
        if attr_size < 4:
            attributes = np.concatenate([attributes, np.ones((n_vertices, 4 - attr_size), dtype=attr_dtype)], axis=1)

        # Create vertex array
        vbo_vert = self.__ctx__.buffer(vertices.astype('f4'))
        vbo_attr = self.__ctx__.buffer(attributes)
        ibo = self.__ctx__.buffer(triangle_indices.astype('i4')) if triangle_indices is not None else None
        if ibo is None:
            vao = self.__ctx__.vertex_array(self.__program_rasterize__, [(vbo_vert, '4f4', 'in_vert'), (vbo_attr, f'4{attr_dtype}', 'in_attr')])
        else:
            vao = self.__ctx__.vertex_array(self.__program_rasterize__, [(vbo_vert, '4f4', 'in_vert'), (vbo_attr, f'4{attr_dtype}', 'in_attr')], index_buffer=ibo, index_element_size=4)

        # Create frame buffer & textrue
        if image_buffer is None:
            image_buffer = np.zeros((height, width, 4), dtype=attr_dtype)
        if depth_buffer is None:
            depth_buffer = np.ones((height, width), dtype='f4')
        image_tex = self.__ctx__.texture((width, height), 4, dtype=attr_dtype, data=image_buffer)
        depth_tex = self.__ctx__.depth_texture((width, height), data=depth_buffer)
        fbo = self.__ctx__.framebuffer(color_attachments=[image_tex], depth_attachment=depth_tex)

        # Set transform matrix
        self.__program_rasterize__['transform_matrix'].write(transform_matrix.transpose().copy().astype('f4') if transform_matrix is not None else np.eye(4, 4, dtype='f4'))

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.__ctx__.depth_func = '<'
        self.__ctx__.enable(self.__ctx__.DEPTH_TEST)
        if cull_backface:
            self.__ctx__.enable(self.__ctx__.CULL_FACE)
        else:
            self.__ctx__.disable(self.__ctx__.CULL_FACE)
        if alpha_blend:
            self.__ctx__.enable(self.__ctx__.BLEND)
        else:
            self.__ctx__.disable(self.__ctx__.BLEND)
        vao.render(moderngl.TRIANGLES)
        self.__ctx__.disable(self.__ctx__.DEPTH_TEST)

        # Read result
        image_tex.read_into(image_buffer)
        depth_tex.read_into(depth_buffer)

        # Release
        vao.release()
        vbo_vert.release()
        vbo_attr.release()
        ibo.release()
        fbo.release()
        image_tex.release()
        depth_tex.release()

        return image_buffer, depth_buffer

    def texture(self, uv: np.ndarray, texture: np.ndarray, interpolation: Literal['nearest', 'linear'] = 'linear', wrap: Literal['clamp', 'repeat'] = 'clamp'):
        """
        Given an UV image, texturing from the texture map
        """
        assert len(texture.shape) == 3 and 1 <= texture.shape[2] <= 4
        assert uv.shape[2] == 2
        height, width = uv.shape[:2]
        texture_dtype = map_np_dtype(texture.dtype)

        # Create texture, set filter and bind. TODO: min mag filter, mipmap
        texture_tex = self.__ctx__.texture((texture.shape[1], texture.shape[0]), texture.shape[2], dtype=texture_dtype, data=texture)
        if interpolation == 'linear':
            texture_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        elif interpolation == 'nearest':
            texture_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture_tex.use(location=0)
        texture_uv = self.__ctx__.texture((width, height), 2, dtype='f4', data=uv.astype('f4'))
        texture_uv.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture_uv.use(location=1)
        # Create render buffer and frame buffer
        rb = self.__ctx__.renderbuffer((uv.shape[1], uv.shape[0]), 4, dtype=texture_dtype)
        fbo = self.__ctx__.framebuffer(color_attachments=[rb])

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.__ctx__.disable(self.__ctx__.BLEND)
        self.screen_quad_vao.render()

        # Read buffer
        image_buffer = np.frombuffer(fbo.read(components=4, attachment=0, dtype=texture_dtype), dtype=texture_dtype).reshape((height, width, 4))

        # Release
        texture_tex.release()
        rb.release()
        fbo.release()

        return image_buffer[:, :, :texture.shape[2]]

    def render_flow(self, 
        width: int, 
        height: int, 
        vertices_source: np.ndarray, 
        vertices_target: np.ndarray, 
        triangle_indices: np.ndarray = None,  
        transform_matrix_source: np.ndarray = None,
        transform_matrix_target: np.ndarray = None, 
        target_depth_buffer: np.ndarray = None,
        threshold: float = 1e-4,  
        image_buffer: np.ndarray = None, 
        depth_buffer: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Render optical flow. \n
        ## Parameters

        `width` : Image width\n
        `height`: Image height\n
        `triangle_indices`: Numpy array of shape `[T, 3]`. Vertex indices of each triangle. `T` is the number of triangles. If the mesh is not trianglular mesh, `trianglate()` can help.
        `transform_matrix`: Numpy array of shape `[4, 4]`. Row major matrix. Identity matrix by default if not provided. Note the data type will always be converted to `float32`.\n
        `image_buffer`: Numpy array of shape [height, width, 4]. The initial image to draw on. By default the image is initialized with zeros.\n
        `depth_buffer`: Numpy array of shape [height, width]. The initial depth to draw on. By default the depth is initialized with ones.\n

        ## Returns
        `flow_image`: Numpy array of shape `[height, width, 4]`. The first two channels are UV flow in image space. The third channel is the depth flow in linear depth space. The last channel is the occulusion mask, where ones indicate visibility in target image.
            The flow map in normalized image space ranging [0, 1]. Note that in OpenGL the origin of image space (i.e. viewport) is the left bottom corner. 
            If you need to display or save the image through OpenCV or Pillow, you will have to flip the Y axis of both the flow vector and image data array.
            
            >>> opencv_flow_image = flow_image[::-1]    # Flip image data
            >>> opencv_flow_image[:, :, 1] *= -1        # Flip flow vector

        `depth`: Numpy array of shape `[height, width]`. The depth buffer in window space ranging from 0 to 1; 0 is the near plane, and 1 is the far plane. If you want the linear depth in view space (z value), you can use 'to_linear_depth'

        ''' 
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
            vertices_source = np.concatenate([vertices_source, np.ones((n_vertices, 1))], axis=1)
            vertices_target = np.concatenate([vertices_target, np.ones((n_vertices, 1))], axis=1)
        elif vertices_source.shape[1] == 2:
            vertices_source = np.concatenate([vertices_source, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=1)
            vertices_target = np.concatenate([vertices_target, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=1)

        # Create vertex array
        vbo_vert_src = self.__ctx__.buffer(vertices_source.astype('f4'))
        vbo_vert_tgt = self.__ctx__.buffer(vertices_target.astype('f4'))
        ibo = self.__ctx__.buffer(triangle_indices.astype('i4')) if triangle_indices is not None else None
        if ibo is None:
            vao = self.__ctx__.vertex_array(self.__program_flow__, [(vbo_vert_src, '4f4', 'in_vert_src'), (vbo_vert_tgt, '4f4', 'in_vert_tgt')])
        else:
            vao = self.__ctx__.vertex_array(self.__program_flow__, [(vbo_vert_src, '4f4', 'in_vert_src'), (vbo_vert_tgt, '4f4', 'in_vert_tgt')], index_buffer=ibo, index_element_size=4)

        # Create texture
        if target_depth_buffer is None:
            tgt_depth_tex = self.__ctx__.texture((width, height), 1, dtype='f4', data=np.ones((height, width), dtype='f4'))
        else:
            tgt_depth_tex = self.__ctx__.texture((width, height), 1, dtype='f4', data=target_depth_buffer)

        # Create frame buffer
        if image_buffer is None:
            image_buffer = np.zeros((height, width, 4), dtype='f4')
        if depth_buffer is None:
            depth_buffer = np.ones((height, width), dtype='f4')
        image_tex = self.__ctx__.texture((width, height), 4, dtype='f4', data=image_buffer)
        depth_tex = self.__ctx__.depth_texture((width, height), data=depth_buffer)

        fbo = self.__ctx__.framebuffer(color_attachments=[image_tex], depth_attachment=depth_tex)
        
        # Set uniforms and bind texture
        self.__program_flow__['threshold'] = threshold
        self.__program_flow__['transform_matrix_src'].write(transform_matrix_source.transpose().copy().astype('f4') if transform_matrix_source is not None else np.eye(4, 4, dtype='f4'))
        self.__program_flow__['transform_matrix_tgt'].write(transform_matrix_target.transpose().copy().astype('f4') if transform_matrix_target is not None else np.eye(4, 4, dtype='f4'))
        tgt_depth_tex.use(location=0)

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.__ctx__.depth_func = '<'
        self.__ctx__.enable(self.__ctx__.DEPTH_TEST)
        self.__ctx__.disable(self.__ctx__.CULL_FACE)
        self.__ctx__.disable(self.__ctx__.BLEND)
        vao.render(moderngl.TRIANGLES)
        self.__ctx__.disable(self.__ctx__.DEPTH_TEST)

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
    
    def warp_image_3d(self, image: np.ndarray, pixel_positions: np.ndarray, transform_matrix: np.ndarray = None, alpha_blend: bool = False):
        assert len(image.shape) == 3 and len(pixel_positions.shape) == 3
        height, width, n_channels = image.shape
        assert image.shape[:2] == pixel_positions.shape[:2]
        assert 1 <= n_channels <= 4
        assert 0 <= pixel_positions.shape[1] <= 4

        image_dtype = map_np_dtype(image.dtype)
        
        # Pad pixel_positions
        if pixel_positions.shape[2] == 3:
            pixel_positions = np.concatenate([pixel_positions, np.ones((height, width, 1))], axis=2)
        elif pixel_positions.shape[2] == 2:
            pixel_positions = np.concatenate([pixel_positions, np.zeros((height, width, 1)), np.ones((height, width, 1))], axis=2)

        # Create vertex array
        im_uv, im_indices = image_mesh(width, height)
        vbo_uv = self.__ctx__.buffer(im_uv.astype('f4'))
        ibo = self.__ctx__.buffer(triangulate(im_indices).astype('i4'))
        vao = self.__ctx__.vertex_array(self.__program_warp__, [(vbo_uv, '2f4', 'in_uv')], index_buffer=ibo, index_element_size=4)

        # Create texture
        pixel_positions_tex = self.__ctx__.texture((width, height), 4, dtype='f4', data=pixel_positions.astype('f4'))
        image_tex = self.__ctx__.texture((width, height), n_channels, dtype=image_dtype, data=image)

        # Create framebuffer
        rb = self.__ctx__.renderbuffer((width, height), 4, dtype=image_dtype)
        depth_tex = self.__ctx__.depth_texture((width, height))
        fbo = self.__ctx__.framebuffer(color_attachments=[rb], depth_attachment=depth_tex)

        # Set transform matrix
        self.__program_warp__['transform_matrix'].write(transform_matrix.transpose().copy().astype('f4') if transform_matrix is not None else np.eye(4, 4, dtype='f4'))

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        self.__ctx__.depth_func = '<'
        self.__ctx__.enable(self.__ctx__.DEPTH_TEST)
        if alpha_blend:
            self.__ctx__.enable(self.__ctx__.BLEND)
        else:
            self.__ctx__.disable(self.__ctx__.BLEND)
        pixel_positions_tex.use(location=0)
        image_tex.use(location=1)
        vao.render(moderngl.TRIANGLES)
        self.__ctx__.disable(self.__ctx__.DEPTH_TEST)

        # Read result
        image_buffer = np.frombuffer(fbo.read(components=4, attachment=0, dtype=image_dtype), dtype=image_dtype).reshape((height, width, 4))

        # Release
        vao.release()
        vbo_uv.release()
        ibo.release()
        fbo.release()
        rb.release()
        depth_tex.release()

        return image_buffer[:n_channels]

    def warp_image_by_flow(self, image: np.ndarray, flow: np.ndarray, occlusion_mask: np.ndarray = None, alpha_blend: bool = False):
        assert image.shape[:2] == flow.shape[:2]
        assert flow.shape[2] == 2
        height, width, n_channels = image.shape
        assert 1 <= n_channels <= 4
        pixel_positions = np.concatenate([image_uv(width, height), -occlusion_mask.astype(float), np.ones((height, width, 1))])
        return self.warp_image_3d(image, pixel_positions, alpha_blend=alpha_blend)
        

# camera_matrix = np.array([
#     [1., 0., 0., 0.],
#     [0., 1., 0., 0.],
#     [0., 0., 1., 2.],
#     [0., 0., 0., 1.]
# ])

# from shapes import cube
# vertices, indices = cube()

# ctx = Context()

# start = time.time()
# img_flow, depth = ctx.rasterize_flow(320, 320, 
#     vertices, 
#     vertices + np.array([[1., 1., 0.]]), 
#     triangle_indices=triangulate(indices.reshape((-1, 4))),
#     transform_matrix=perspective_from_image(np.pi / 2., 320, 320, 0.01, 100.) @ np.linalg.inv(camera_matrix)
# )

# #uv = image_uv(10, 10)
# #print(ctx.texture(uv, np.linspace(1, 100, 100).reshape((10, 10, 1)).astype('f4'))[:, :, 0])
# print(time.time() - start)

# from PIL import Image
# Image.fromarray(np.abs(img_flow * 255).astype(np.uint8)).show()