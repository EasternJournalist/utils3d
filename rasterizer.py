from typing import Literal, Tuple
import numpy as np
import moderngl
import time

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
        self.__ctx__ = moderngl.create_standalone_context()
        self.__ctx__.depth_func = '<'
        self.__ctx__.enable(self.__ctx__.DEPTH_TEST)

        self.program_rasterize = self.__ctx__.program(
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

        self.program_texture = self.__ctx__.program(
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
        self.program_texture['tex'] = 0
        self.program_texture['uv'] = 1

        self.program_flow = self.__ctx__.program(
            vertex_shader='''
            #version 330

            in vec4 in_vert_current;
            in vec4 in_vert_next;
            out vec4 current_pos;
            out vec4 next_pos;
            uniform mat4 transform_matrix;

            void main() {
                current_pos = transform_matrix * in_vert_current;
                next_pos = transform_matrix * in_vert_next; 
                gl_Position = current_pos;
            }
            ''',
            fragment_shader='''
            #version 330

            in vec4 current_pos;
            in vec4 next_pos;
            out vec4 flow;

            uniform float threshold;
            uniform sampler2D next_depth;

            void main() {
                vec3 current_pos_ndc = current_pos.xyz / current_pos.w;
                vec3 current_pos_scr = current_pos_ndc * 0.5 + 0.5;
                vec3 next_pos_ndc = next_pos.xyz / next_pos.w;
                vec3 next_pos_scr = next_pos_ndc * 0.5 + 0.5;

                float visible = next_pos_scr.z < texture(next_depth, next_pos_scr.xy).x + threshold ? 1 : 0;
                flow = vec4(next_pos_scr.xy - current_pos_scr.xy, next_pos.w - current_pos.w, visible);
            }
            '''
        )
        self.program_flow['next_depth'] = 0

        self.screen_quad_vbo = self.__ctx__.buffer(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype='f4'))
        self.screen_quad_ibo = self.__ctx__.buffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))
        self.screen_quad_vao = self.__ctx__.vertex_array(self.program_texture, [(self.screen_quad_vbo, '2f4', 'in_vert')], index_buffer=self.screen_quad_ibo, index_element_size=4)

    def rasterize(self, 
        width: int, 
        height: int, 
        vertices: np.ndarray, 
        attributes: np.ndarray, 
        triangle_indices: np.ndarray = None,  
        transform_matrix: np.ndarray = None, 
        image_buffer: np.ndarray = None, 
        depth_buffer: np.ndarray = None
    ):
        '''
            Rasterize triangles onto image with attributes attached to each vertex.\n
            `width` : Image width\n
            `height`: Image height\n
            `vertices`: Numpy array of shape [N, M], M = 2, 3 or 4. Vertex positions. `transform_matrix` can be applied convert vertex positions to clip space. 
                If the dimension of vertices coordinate is less than 4, coordinates `z` will be padded with `0` and `w` will be padded with `1`.
                Note the data type will always be converted to `float32`.\n
            `attributes`: Numpy array of shape [N, L], L = 1, 2, 3 or 4. Variouse Attribtues attached to each vertex. The output image data type will be the same as that of attributes.\n
            `transform_matrix`: Numpy array of shape [4, 4]. Row major matrix. Identity matrix by default if not provided. Note the data type will always be converted to `float32`.\n
            `image_buffer`:\n
            `depth_buffer`: 
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

        # Create vertex buffer & index buffer
        vbo_vert = self.__ctx__.buffer(vertices.astype('f4'))
        vbo_attr = self.__ctx__.buffer(attributes)
        ibo = self.__ctx__.buffer(triangle_indices.astype('i4')) if triangle_indices is not None else None
        
        # Create vertex array
        if ibo is None:
            vao = self.__ctx__.vertex_array(self.program_rasterize, [(vbo_vert, '4f4', 'in_vert'), (vbo_attr, f'4{attr_dtype}', 'in_attr')])
        else:
            vao = self.__ctx__.vertex_array(self.program_rasterize, [(vbo_vert, '4f4', 'in_vert'), (vbo_attr, f'4{attr_dtype}', 'in_attr')], index_buffer=ibo, index_element_size=4)

        # Create image texture & depth texture & frame buffer
        if image_buffer is None:
            image_buffer = np.zeros((height, width, 4), dtype=attr_dtype)
        if depth_buffer is None:
            depth_buffer = np.ones((height, width), dtype='f4')
        image_tex = self.__ctx__.texture((width, height), 4, dtype=attr_dtype, data=image_buffer)
        depth_tex = self.__ctx__.depth_texture((width, height), data=depth_buffer)
        fbo = self.__ctx__.framebuffer(color_attachments=[image_tex], depth_attachment=depth_tex)

        # Set transform matrix
        self.program_rasterize['transform_matrix'].write(transform_matrix.transpose().copy().astype('f4') if transform_matrix is not None else np.eye(4, 4, dtype='f4'))

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)

        vao.render(moderngl.TRIANGLES)

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

    def rasterize_flow(self, 
        width: int, 
        height: int, 
        vertices_current: np.ndarray, 
        vertices_next: np.ndarray, 
        triangle_indices: np.ndarray = None,  
        transform_matrix: np.ndarray = None, 
        next_depth_buffer: np.ndarray = None,
        threshold: float = 1e-4,  
        image_buffer: np.ndarray = None, 
        depth_buffer: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Rasterize triangles onto image with attributes attached to each vertex.\n
        ## Parameters

        `width` : Image width\n
        `height`: Image height\n
        `vertices`: Numpy array of shape [N, M], M = 2, 3 or 4. Vertex positions. `transform_matrix` can be applied convert vertex positions to clip space. 
            If the dimension of vertices coordinate is less than 4, coordinates `z` will be padded with `0` and `w` will be padded with `1`.
            Note the data type will always be converted to `float32`.\n
        `attributes`: Numpy array of shape [N, L], L = 1, 2, 3 or 4. Variouse Attribtues attached to each vertex. The output image data type will be the same as that of attributes.\n
        `transform_matrix`: Numpy array of shape [4, 4]. Row major matrix. Identity matrix by default if not provided. Note the data type will always be converted to `float32`.\n
        `image_buffer`:\n
        `depth_buffer`: 

        ## Returns
        `flow_image`: The flow map in normalized image space ranging [0, 1]. Note that in OpenGL, the origin of image space (i.e. viewport) is the left bottom corner. 
            If you need to display or process the image through OpenCV  Pillow, you will have to flip the Y axis of both the flow vector and image data array.
            
            >>> opencv_flow_image = flow_image[::-1]    # Flip image data
            >>> opencv_flow_image[:, :, 1] *= -1        # Flip flow vector

        `depth`: The depth buffer in window space ranging from 0 to 1; 0 is the near plane, and 1 is the far plane. If you want the linear depth in view space (z value), you can use 'to_linear_depth'

        '''
        assert len(vertices_current.shape) == 2 and len(vertices_next.shape) == 2
        assert vertices_current.shape == vertices_next.shape
        assert 2 <= vertices_current.shape[1] <= 4
        if next_depth_buffer is not None:
            assert next_depth_buffer.dtype == np.float32 and next_depth_buffer.shape[0] == height and next_depth_buffer.shape[1] == width
        if image_buffer is not None:
            assert image_buffer.shape[0] == height and image_buffer.shape[1] == width
        if depth_buffer is not None:
            assert depth_buffer.dtype == np.float32 and len(depth_buffer.shape) == 2 and depth_buffer.shape[0] == height and depth_buffer.shape[1] == width

        # Pad vertices
        n_vertices = vertices_current.shape[0]
        if vertices_current.shape[1] == 3:
            vertices_current = np.concatenate([vertices_current, np.ones((n_vertices, 1))], axis=1)
            vertices_next = np.concatenate([vertices_next, np.ones((n_vertices, 1))], axis=1)
        elif vertices_current.shape[1] == 2:
            vertices_current = np.concatenate([vertices_current, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=1)
            vertices_next = np.concatenate([vertices_next, np.zeros((n_vertices, 1)), np.ones((n_vertices, 1))], axis=1)

        # Create vertex buffer & index buffer
        vbo_vert_current = self.__ctx__.buffer(vertices_current.astype('f4'))
        vbo_vert_next = self.__ctx__.buffer(vertices_next.astype('f4'))
        ibo = self.__ctx__.buffer(triangle_indices.astype('i4')) if triangle_indices is not None else None
        
        # Create vertex array
        if ibo is None:
            vao = self.__ctx__.vertex_array(self.program_flow, [(vbo_vert_current, '4f4', 'in_vert_current'), (vbo_vert_next, '4f4', 'in_vert_next')])
        else:
            vao = self.__ctx__.vertex_array(self.program_flow, [(vbo_vert_current, '4f4', 'in_vert_current'), (vbo_vert_next, '4f4', 'in_vert_next')], index_buffer=ibo, index_element_size=4)

        # Create textures
        if image_buffer is None:
            image_buffer = np.zeros((height, width, 4), dtype='f4')
        if depth_buffer is None:
            depth_buffer = np.ones((height, width), dtype='f4')
        image_tex = self.__ctx__.texture((width, height), 4, dtype='f4', data=image_buffer)
        depth_tex = self.__ctx__.depth_texture((width, height), data=depth_buffer)

        if next_depth_buffer is None:
            next_depth_tex = self.__ctx__.texture((width, height), 1, dtype='f4', data=np.ones((height, width), dtype='f4'))
        else:
            next_depth_tex = self.__ctx__.texture((width, height), 1, dtype='f4', data=next_depth_buffer)

        # Create frame buffer
        fbo = self.__ctx__.framebuffer(color_attachments=[image_tex], depth_attachment=depth_tex)
        
        # Set uniforms and bind texture
        self.program_flow['threshold'] = threshold
        self.program_flow['transform_matrix'].write(transform_matrix.transpose().copy().astype('f4') if transform_matrix is not None else np.eye(4, 4, dtype='f4'))
        next_depth_tex.use(location=0)

        # Render
        fbo.use()
        fbo.viewport = (0, 0, width, height)

        vao.render(moderngl.TRIANGLES)

        # Read result
        image_tex.read_into(image_buffer)
        depth_tex.read_into(depth_buffer)

        # Release
        vao.release()
        vbo_vert_current.release()
        vbo_vert_next.release()
        ibo.release()
        fbo.release()
        image_tex.release()
        depth_tex.release()
        next_depth_tex.release()

        return image_buffer, depth_buffer
    
    def texture(self, uv: np.ndarray, texture: np.ndarray, interpolation: Literal['nearest', 'linear'] = 'linear', wrap: Literal['clamp', 'repeat'] = 'clamp'):
        assert len(texture.shape) == 3 and 1 <= texture.shape[2] <= 4
        height, width = uv.shape[0], uv.shape[1]
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
        self.screen_quad_vao.render()

        # Read buffer
        image_buffer = np.frombuffer(fbo.read(components=4, attachment=0, dtype=texture_dtype), dtype=texture_dtype).reshape((height, width, 4))

        # Release
        texture_tex.release()
        rb.release()
        fbo.release()

        return image_buffer[:, :, :texture.shape[2]]


camera_matrix = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 2.],
    [0., 0., 0., 1.]
])

from shapes import cube
vertices, indices = cube()

ctx = Context()

start = time.time()
img_flow, depth = ctx.rasterize_flow(320, 320, 
    vertices, 
    vertices + np.array([[1., 1., 0.]]), 
    triangle_indices=triangulate(indices.reshape((-1, 4))),
    transform_matrix=perspective_from_image(np.pi / 2., 320, 320, 0.01, 100.) @ np.linalg.inv(camera_matrix)
)

#uv = image_uv(10, 10)
#print(ctx.texture(uv, np.linspace(1, 100, 100).reshape((10, 10, 1)).astype('f4'))[:, :, 0])
print(time.time() - start)

from PIL import Image
Image.fromarray(np.abs(img_flow * 255).astype(np.uint8)).show()