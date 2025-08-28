import os
from typing import *

import numpy as np
import moderngl

from . import transforms, utils, mesh


__all__ = [
    'RastContext',
    'rasterize_triangles',
    'rasterize_triangles_peeling',
    'rasterize_lines',
    'texture',
    'test_rasterization',
]


TRIANGLE_VERTEX_ATTRIBUTE_VS = """
#version 330
uniform mat4 uMatrix;
in vec3 inVert;
in vec4 inAttr;
out vec4 vAttr;
void main() {
    gl_Position = uMatrix * vec4(inVert, 1.0);
    vAttr = inAttr;
}
"""

TRIANGLE_VERTEX_ATTRIBUTE_FS = """
#version 330
in vec4 vAttr;
out vec4 fAttr;
void main() {
    fAttr = vAttr;
}
"""

TRIANGLE_FACE_ATTRIBUTE_VS = """
#version 330
uniform mat4 uMatrix;
in vec3 inVert;
in vec4 inAttr;
flat out vec4 vAttr;
void main() {
    gl_Position = uMatrix * vec4(inVert, 1.0);
    vAttr = inAttr;
}
"""

TRIANGLE_FACE_ATTRIBUTE_VS = """
#version 330
uniform mat4 uMatrix;
in vec3 inVert;
in vec4 inAttr;
flat out vec4 vAttr;
void main() {
    gl_Position = uMatrix * vec4(inVert, 1.0);
    vAttr = inAttr;
}
"""

TRIANGLE_FACE_ATTRIBUTE_FS = """
#version 330
flat in vec4 vAttr;
out vec4 fAttr;
void main() {
    fAttr = vAttr;
}
"""

TRIANGLE_VERTEX_ATTRIBUTE_PEELING_FS = """
#version 330
uniform sampler2D prevDepthTex;
uniform vec2 screenSize;
in vec4 vAttr;
out vec4 fAttr;
void main() {
    vec2 uv = gl_FragCoord.xy / screenSize;
    float prevDepth = texture(prevDepthTex, uv).r;
    float currDepth = gl_FragCoord.z;
    if (currDepth <= prevDepth + 1e-12) {
        discard; 
    }
    fAttr = vAttr;
}
"""

TRIANGLE_FACE_ATTRIBUTE_PEELING_FS = """
#version 330
uniform sampler2D prevDepthTex;
uniform vec2 screenSize;
flat in vec4 vAttr;
out vec4 fAttr;
void main() {
    vec2 uv = gl_FragCoord.xy / screenSize;
    float prevDepth = texture(prevDepthTex, uv).r;
    float currDepth = gl_FragCoord.z;
    if (currDepth <= prevDepth + 1e-12) {
        discard; 
    }
    fAttr = vAttr;
}
"""

TEXTURE_VS = """
#version 330 core
in vec2 inVert;
out vec2 vScrCoord;=
void main() {
    vScrCoord = inVert * 0.5 + 0.5;
    gl_Position = vec4(inVert, 0., 1.);
}
"""

TEXTURE_FS = """
#version 330
uniform sampler2D tex;
uniform sampler2D uv;
in vec2 vScrCoord;
out vec4 texColor;

void main() {
    texColor = vec4(texture(tex, texture(uv, vScrCoord).xy));
}
"""

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
    def __init__(self, *args, **kwargs):
        """
        Create a moderngl context.

        Args:
            See moderngl.create_context
        """
        if len(args) == 0 and len(kwargs) == 0:
            
            try:
                # Default
                self.mgl_ctx = moderngl.create_context()    
            except:
                pass
            if not hasattr(self, 'mgl_ctx'):
                try:
                    # Default standalone
                    self.mgl_ctx = moderngl.create_context(standalone=True)     
                except:
                    pass
            if not hasattr(self, 'mgl_ctx'):
                try:
                    # EGL
                    self.mgl_ctx = moderngl.create_context(standalone=True, backend='egl',)     
                except:
                    pass
            if not hasattr(self, 'mgl_ctx'):
                try:
                    # EGL with more specifications
                    self.mgl_ctx = moderngl.create_context(standalone=True, backend='egl', libgl='libGL.so.1', libegl='libEGL.so.1')    
                except:
                    pass
            if not hasattr(self, 'mgl_ctx'):
                raise RuntimeError(
                    "Failed to create moderngl context with default settings. " 
                    "Please specify context creation settings following https://moderngl.readthedocs.io/en/latest/topics/context.html. "
                    "If running on headless ubuntu server, please refer to https://moderngl.readthedocs.io/en/latest/techniques/headless_ubuntu_18_server.html"
                )
            
        elif len(args) == 1 and isinstance(args[0], moderngl.Context):
            self.mgl_ctx = args[0]
        else:
            self.mgl_ctx = moderngl.create_context(*args, **kwargs)
        self.programs = {}

    def get_program_triangle_vertex_attribute(self,) -> moderngl.Program:
        if f'triangle_vertex_attribute' not in self.programs:
            self.programs[f'triangle_vertex_attribute'] = self.mgl_ctx.program(
                vertex_shader=TRIANGLE_VERTEX_ATTRIBUTE_VS, 
                fragment_shader=TRIANGLE_VERTEX_ATTRIBUTE_FS
            )
        return self.programs[f'triangle_vertex_attribute']

    def get_program_triangle_face_attribute(self,) -> moderngl.Program:
        if f'triangle_face_attribute' not in self.programs:
            self.programs[f'triangle_face_attribute'] = self.mgl_ctx.program(
                vertex_shader=TRIANGLE_FACE_ATTRIBUTE_VS, 
                fragment_shader=TRIANGLE_FACE_ATTRIBUTE_FS
            )
        return self.programs[f'triangle_face_attribute']

    def get_program_triangle_vertex_attribute_peeling(self) -> moderngl.Program:
        if f'triangle_vertex_attribute_peeling' not in self.programs:
            self.programs[f'triangle_vertex_attribute_peeling'] = self.mgl_ctx.program(
                vertex_shader=TRIANGLE_VERTEX_ATTRIBUTE_VS, 
                fragment_shader=TRIANGLE_VERTEX_ATTRIBUTE_PEELING_FS
            )
            self.programs[f'triangle_vertex_attribute_peeling']['prevDepthTex'] = 0
        return self.programs[f'triangle_vertex_attribute_peeling']

    def get_program_triangle_face_attribute_peeling(self) -> moderngl.Program:
        if f'triangle_face_attribute_peeling' not in self.programs:
            self.programs[f'triangle_face_attribute_peeling'] = self.mgl_ctx.program(
                vertex_shader=TRIANGLE_FACE_ATTRIBUTE_VS, 
                fragment_shader=TRIANGLE_FACE_ATTRIBUTE_PEELING_FS
            )
            self.programs[f'triangle_face_attribute_peeling']['prevDepthTex'] = 0
        return self.programs[f'triangle_face_attribute_peeling']

    def program_texture(self) -> moderngl.Program:
        if f'texture' not in self.programs:
            self.programs[f'texture'] = self.mgl_ctx.program(
                vertex_shader=TEXTURE_VS, 
                fragment_shader=TEXTURE_FS
            )
            self.programs[f'texture']['tex'] = 0
            self.programs[f'texture']['uv'] = 1
        
        return self.programs[f'texture']

    
def rasterize_triangles(
    ctx: RastContext,
    width: int,
    height: int,
    *,
    vertices: np.ndarray,
    attributes: np.ndarray,
    attributes_domain: Literal['vertex', 'face'] = 'vertex',
    faces: Optional[np.ndarray] = None,
    transform: np.ndarray = None,
    cull_backface: bool = True,
    return_depth: bool = False,
    background_image: Optional[np.ndarray] = None,
    background_depth: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        ctx (RastContext): rasterization context
        width (int): width of rendered image
        height (int): height of rendered image
        vertices (np.ndarray): [N, 3] or [T, 3, 3]
        attributes (np.ndarray): [N, C], [T, 3, C] for vertex domain or [T, C] for face domain
        attributes_domain (Literal['vertex', 'face']): domain of the attributes
        faces (np.ndarray): [T, 3] or None
        transform (np.ndarray): [4, 4] OpenGL Model-View-Projection transformation matrix. 
        cull_backface (bool): whether to cull backface
        background_image (np.ndarray): [H, W, C] background image
        background_depth (np.ndarray): [H, W] background depth

    Returns:
        image (np.ndarray): [H, W, C] rendered image corresponding to the input attributes
        depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None.
    """
    
    if faces is None:
        assert vertices.ndim == 3 and vertices.shape[1] == vertices.shape[2] == 3, "If faces is None, vertices must be an array with shape (T, 3, 3)"
    else:
        assert faces.ndim == 2 and faces.shape[1] == 3, f"Faces should be a 2D array with shape (T, 3), but got {faces.shape}"
        assert faces.dtype == np.uint32 or faces.dtype == np.int32
        assert vertices.ndim == 2 and vertices.shape[1] == 3

    if attributes_domain == 'vertex':
        assert (attributes.shape[:-1] == vertices.shape[:-1]), f"Attribute shape {attributes.shape} does not match vertex shape {vertices.shape}"
    elif attributes_domain == 'face':
        if faces is None:
            assert attributes.shape[0] == vertices.shape[0], f"Attribute shape {attributes.shape} does not match vertex shape {vertices.shape}"
        else:
            assert attributes.shape[0] == faces.shape[0], f"Attribute shape {attributes.shape} does not match face shape {faces.shape}"
    else:
        raise ValueError(f"Unknown attributes domain: {attributes_domain}")

    assert attributes.shape[-1] in [1, 2, 3, 4], f'Vertex attribute only supports channels 1, 2, 3, 4, but got {attributes.shape[-1]}'
    
    assert vertices.dtype == np.float32
    assert attributes.dtype == np.float32
    if faces is not None:
        assert faces.dtype == np.uint32 or faces.dtype == np.int32

    assert transform is None or transform.shape == (4, 4), f"Transform should be a 4x4 matrix, but got {transform.shape}"
    assert transform is None or transform.dtype == np.float32, f"Transform should be float32, but got {transform.dtype}"
    if background_image is not None:
        assert background_image.ndim == 3 and background_image.shape == (height, width, attributes.shape[-1]), f"Image should be a 3D array with shape (H, W, {attributes.shape[1]}), but got {background_image.shape}"
        assert background_image.dtype == np.float32, f"Image should be float32, but got {background_image.dtype}"
    if background_depth is not None:
        assert background_depth.ndim == 2 and background_depth.shape == (height, width), f"Depth should be a 2D array with shape (H, W), but got {background_depth.shape}"
        assert background_depth.dtype == np.float32, f"Depth should be float32, but got {background_depth.dtype}"

    num_channels = attributes.shape[-1]
    attributes = np.concatenate([attributes, np.zeros((*attributes.shape[:-1], 4 - num_channels,), dtype=attributes.dtype)], axis=-1) if num_channels < 4 else attributes
    if attributes_domain == 'vertex':
        prog = ctx.get_program_triangle_vertex_attribute()
    elif attributes_domain == 'face':
        prog = ctx.get_program_triangle_face_attribute()
        attributes = attributes[:, None, :].repeat(3, axis=1)
        if faces is not None:
            vertices = vertices[faces]
            faces = None
        
    transform = np.eye(4, np.float32) if transform is None else transform

    # Create buffers
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
    vbo_attributes = ctx.mgl_ctx.buffer(np.ascontiguousarray(attributes, dtype='f4'))
    ibo = ctx.mgl_ctx.buffer(np.ascontiguousarray(faces, dtype='i4')) if faces is not None else None
    vao = ctx.mgl_ctx.vertex_array(
        prog,
        [
            (vbo_vertices, '3f', 'inVert'),
            (vbo_attributes, f'4f', 'inAttr'),
        ],
        index_buffer=ibo,
        mode=moderngl.TRIANGLES,
    )

    # Create output textures
    image_tex = ctx.mgl_ctx.texture((width, height), 4, dtype='f4', data=np.ascontiguousarray(background_image[::-1, :, :]) if background_image is not None else None)
    depth_tex = ctx.mgl_ctx.depth_texture((width, height), data=np.ascontiguousarray(background_depth[::-1, :]) if background_depth is not None else None)
    
    # Create framebuffer
    fbo = ctx.mgl_ctx.framebuffer(
        color_attachments=[image_tex],
        depth_attachment=depth_tex,
    )
    fbo.viewport = (0, 0, width, height)
    fbo.use()

    # Set uniforms
    prog['uMatrix'].write(np.ascontiguousarray(transform.transpose().astype('f4')))

    # Set render states
    ctx.mgl_ctx.depth_func = '<'
    if background_depth is None:
        ctx.mgl_ctx.clear(depth=1.0)
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    if cull_backface:
        ctx.mgl_ctx.enable(ctx.mgl_ctx.CULL_FACE)
    else:
        ctx.mgl_ctx.disable(ctx.mgl_ctx.CULL_FACE)
    
    # Render
    vao.render()

    # Read
    image = np.empty((height, width, 4), dtype='f4') 
    image_tex.read_into(image)
    image = np.flip(image, axis=0)
    if return_depth:
        depth = np.empty((height, width), dtype='f4')
        depth_tex.read_into(depth)
        depth = np.flip(depth, axis=0)
    else:
        depth = None

    # Release
    vao.release()
    if ibo is not None:
        ibo.release()
    vbo_vertices.release()
    vbo_attributes.release()
    fbo.release()
    image_tex.release()
    depth_tex.release()

    return image[:, :, :num_channels], depth


def rasterize_triangles_peeling(
    ctx: RastContext,
    width: int,
    height: int,
    *,
    vertices: np.ndarray,
    attributes: np.ndarray,
    attributes_domain: Literal['vertex', 'face'] = 'vertex',
    faces: Optional[np.ndarray] = None,
    transform: np.ndarray = None,
    cull_backface: bool = True,
    return_depth: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray]]:
    """
    Rasterize vertex attribute.

    Args:
        width (int): width of rendered image
        height (int): height of rendered image
        vertices (np.ndarray): [N, 3] or [T, 3, 3]
        attributes (np.ndarray): [N, C] or [T, 3, 3]
        attributes_domain (Literal['vertex', 'face']): domain of the attributes
        faces (np.ndarray): [T, 3] or None
        transform (np.ndarray): [4, 4] OpenGL Model-View-Projection transformation matrix. 
        cull_backface (bool): whether to cull backface

    Returns:
        image (np.ndarray): [H, W, C] rendered image
        depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None.
    """
    
    if faces is None:
        assert vertices.ndim == 3 and vertices.shape[1] == vertices.shape[2] == 3, "If faces is None, vertices must be an array with shape (T, 3, 3)"
    else:
        assert faces.ndim == 2 and faces.shape[1] == 3, f"Faces should be a 2D array with shape (T, 3), but got {faces.shape}"
        assert faces.dtype == np.uint32 or faces.dtype == np.int32
        assert vertices.ndim == 2 and vertices.shape[1] == 3

    if attributes_domain == 'vertex':
        assert (attributes.shape[:-1] == vertices.shape[:-1]), f"Attribute shape {attributes.shape} does not match vertex shape {vertices.shape}"
    elif attributes_domain == 'face':
        if faces is None:
            assert attributes.shape[0] == vertices.shape[0], f"Attribute shape {attributes.shape} does not match vertex shape {vertices.shape}"
        else:
            assert attributes.shape[0] == faces.shape[0], f"Attribute shape {attributes.shape} does not match face shape {faces.shape}"
    else:
        raise ValueError(f"Unknown attributes domain: {attributes_domain}")

    assert attributes.shape[-1] in [1, 2, 3, 4], f'Vertex attribute only supports channels 1, 2, 3, 4, but got {attributes.shape[-1]}'
    
    assert vertices.dtype == np.float32
    assert attributes.dtype == np.float32
    if faces is not None:
        assert faces.dtype == np.uint32 or faces.dtype == np.int32

    assert transform is None or transform.shape == (4, 4), f"Transform should be a 4x4 matrix, but got {transform.shape}"
    assert transform is None or transform.dtype == np.float32, f"Transform should be float32, but got {transform.dtype}"

    num_channels = attributes.shape[-1]
    attributes = np.concatenate([attributes, np.zeros((*attributes.shape[:-1], 4 - num_channels,), dtype=attributes.dtype)], axis=-1) if num_channels < 4 else attributes
    if attributes_domain == 'vertex':
        prog = ctx.get_program_triangle_vertex_attribute_peeling()
    elif attributes_domain == 'face':
        prog = ctx.get_program_triangle_face_attribute_peeling()
        attributes = attributes[:, None, :].repeat(3, axis=1)
        if faces is not None:
            vertices = vertices[faces]
            faces = None
        
    transform = np.eye(4, np.float32) if transform is None else transform

    # Create buffers
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
    vbo_attributes = ctx.mgl_ctx.buffer(np.ascontiguousarray(attributes, dtype='f4'))
    ibo = ctx.mgl_ctx.buffer(np.ascontiguousarray(faces, dtype='i4')) if faces is not None else None
    vao = ctx.mgl_ctx.vertex_array(
        prog,
        [
            (vbo_vertices, '3f', 'inVert'),
            (vbo_attributes, f'4f', 'inAttr'),
        ],
        index_buffer=ibo,
        mode=moderngl.TRIANGLES,
    )

    # Create textures
    image_tex = ctx.mgl_ctx.texture((width, height), 4, dtype='f4', data=None)
    depth_tex_a = ctx.mgl_ctx.depth_texture((width, height), data=None)
    depth_tex_b = ctx.mgl_ctx.depth_texture((width, height), data=None)

    # Create and use frame buffer
    fbo_curr = ctx.mgl_ctx.framebuffer(
        color_attachments=[image_tex],
        depth_attachment=depth_tex_a,
    )
    fbo_curr.viewport = (0, 0, width, height)
    fbo_prev = ctx.mgl_ctx.framebuffer(
        color_attachments=[image_tex],
        depth_attachment=depth_tex_b,
    )
    fbo_prev.viewport = (0, 0, width, height)

    # Set uniforms
    prog['uMatrix'].write(np.ascontiguousarray(transform.transpose().astype('f4')))
    prog['screenSize'].value = (width, height)

    # Rendering settings
    ctx.mgl_ctx.depth_func = '<'
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    if cull_backface:
        ctx.mgl_ctx.enable(ctx.mgl_ctx.CULL_FACE)
    else:
        ctx.mgl_ctx.disable(ctx.mgl_ctx.CULL_FACE)

    # Initialize prev depth texture = 0
    fbo_prev.use()
    ctx.mgl_ctx.clear(depth=0.0)

    try:
        while True:
            # Bind frame buffer & prev depth texture
            fbo_curr.use()
            fbo_prev.depth_attachment.use(location=0) 
            ctx.mgl_ctx.clear()
            
            # Render
            vao.render()

            # Read
            image = np.empty((height, width, 4), dtype='f4') 
            image_tex.read_into(image)
            image = np.flip(image, axis=0)
            if return_depth:
                depth = np.empty((height, width), dtype='f4')
                fbo_curr.depth_attachment.read_into(depth)
                depth = np.flip(depth, axis=0)
            else:
                depth = None

            yield image[:, :, :num_channels], depth

            # Swap curr and prev fbos
            fbo_curr, fbo_prev = fbo_prev, fbo_curr
            
    finally:
        # Release
        vao.release()
        if ibo is not None:
            ibo.release()
        vbo_vertices.release()
        vbo_attributes.release()
        fbo_curr.release()
        fbo_prev.release()
        image_tex.release()
        depth_tex_a.release()
        depth_tex_b.release()


def rasterize_lines(
    ctx: RastContext,
    width: int,
    height: int,
    *,
    vertices: np.ndarray,
    lines: np.ndarray,
    attr: np.ndarray,
    transform: np.ndarray = None,
    line_width: float = 1.0,
    return_depth: bool = False,
    image: np.ndarray = None,
    depth: np.ndarray = None
) -> Tuple[np.ndarray, ...]:
    """
    Rasterize vertex attribute.

    Args:
        width (int): width of rendered image
        height (int): height of rendered image
        vertices (np.ndarray): [N, 3]
        faces (np.ndarray): [T, 3]
        attr (np.ndarray): [N, C]
        transform (np.ndarray): [4, 4] model-view-projection matrix
        line_width (float): width of line. Defaults to 1.0. NOTE: Values other than 1.0 may not work across all platforms.
        cull_backface (bool): whether to cull backface

    Returns:
        image (np.ndarray): [H, W, C] rendered image
        depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert lines.ndim == 2 and lines.shape[1] == 2, f"Lines should be a 2D array with shape (T, 2), but got {lines.shape}"
    assert attr.ndim == 2 and attr.shape[1] in [1, 2, 3, 4], f'Vertex attribute only supports channels 1, 2, 3, 4, but got {attr.shape}'
    assert vertices.shape[0] == attr.shape[0]
    assert vertices.dtype == np.float32
    assert lines.dtype == np.uint32 or lines.dtype == np.int32
    assert attr.dtype == np.float32, "Attribute should be float32"

    C = attr.shape[1]
    prog = ctx.get_program_triangle_vertex_attribute(C)

    transform = transform if transform is not None else np.eye(4, np.float32)

    # Create buffers
    ibo = ctx.mgl_ctx.buffer(np.ascontiguousarray(lines, dtype='i4'))
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
    vbo_attr = ctx.mgl_ctx.buffer(np.ascontiguousarray(attr, dtype='f4'))
    vao = ctx.mgl_ctx.vertex_array(
        prog,
        [
            (vbo_vertices, '3f', 'inVert'),
            (vbo_attr, f'{C}f', 'inAttr'),
        ],
        ibo,
        mode=moderngl.LINES,
    )

    # Create textures
    image_tex = ctx.mgl_ctx.texture((width, height), C, dtype='f4', data=np.ascontiguousarray(image[::-1, :, :]) if image is not None else None)
    depth_tex = ctx.mgl_ctx.depth_texture((width, height), data=np.ascontiguousarray(depth[::-1, :]) if depth is not None else None)
    
    # Create framebuffer
    fbo = ctx.mgl_ctx.framebuffer(
        color_attachments=[image_tex],
        depth_attachment=depth_tex,
    )
    fbo.viewport = (0, 0, width, height)
    fbo.use()

    # Set uniforms
    prog['uMatrix'].write(transform.transpose().copy().astype('f4'))

    if depth is None:
        ctx.mgl_ctx.clear(depth=1.0)
    ctx.mgl_ctx.depth_func = '<'
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    ctx.mgl_ctx.line_width = line_width

    # Render
    vao.render()

    # Read
    image = np.zeros((height, width, C), dtype='f4')
    image_tex.read_into(image)
    image = np.flip(image, axis=0)
    if return_depth:
        depth = np.zeros((height, width), dtype='f4')
        depth_tex.read_into(depth)
        depth = np.flip(depth, axis=0)
    else:
        depth = None

    # Release
    vao.release()
    ibo.release()
    vbo_vertices.release()
    vbo_attr.release()
    fbo.release()
    image_tex.release()
    depth_tex.release()

    return image, depth


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
    texture_tex = ctx.mgl_ctx.texture((texture.shape[1], texture.shape[0]), texture.shape[2], dtype=texture_dtype, data=np.ascontiguousarray(texture))
    if interpolation == 'linear':
        texture_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    elif interpolation == 'nearest':
        texture_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    texture_tex.use(location=0)
    texture_uv = ctx.mgl_ctx.texture((width, height), 2, dtype='f4', data=np.ascontiguousarray(uv.astype('f4', copy=False)))
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


# def warp_image_by_depth(
#     ctx: RastContext,
#     src_depth: np.ndarray,
#     src_image: np.ndarray = None,
#     width: int = None,
#     height: int = None,
#     *,
#     extrinsics_src: np.ndarray = None,
#     extrinsics_tgt: np.ndarray = None,
#     intrinsics_src: np.ndarray = None,
#     intrinsics_tgt: np.ndarray = None,
#     near: float = 0.1,
#     far: float = 100.0,
#     cull_backface: bool = True,
#     ssaa: int = 1,
#     return_depth: bool = False,
# ) -> Tuple[np.ndarray, ...]:
#     """
#     Warp image by depth map.

#     Args:
#         ctx (RastContext): rasterizer context
#         src_depth (np.ndarray): [H, W]
#         src_image (np.ndarray, optional): [H, W, C]. The image to warp. Defaults to None (use uv coordinates).
#         width (int, optional): width of the output image. None to use depth map width. Defaults to None.
#         height (int, optional): height of the output image. None to use depth map height. Defaults to None.
#         extrinsics_src (np.ndarray, optional): extrinsics matrix of the source camera. Defaults to None (identity).
#         extrinsics_tgt (np.ndarray, optional): extrinsics matrix of the target camera. Defaults to None (identity).
#         intrinsics_src (np.ndarray, optional): intrinsics matrix of the source camera. Defaults to None (use the same as intrinsics_tgt).
#         intrinsics_tgt (np.ndarray, optional): intrinsics matrix of the target camera. Defaults to None (use the same as intrinsics_src).
#         cull_backface (bool, optional): whether to cull backface. Defaults to True.
#         ssaa (int, optional): super sampling anti-aliasing. Defaults to 1.
    
#     Returns:
#         tgt_image (np.ndarray): [H, W, C] warped image (or uv coordinates if image is None).
#         tgt_depth (np.ndarray): [H, W] screen space depth, ranging from 0 to 1. If return_depth is False, it is None.
#     """
#     assert src_depth.ndim == 2

#     if width is None:
#         width = src_depth.shape[1]
#     if height is None:
#         height = src_depth.shape[0]
#     if src_image is not None:
#         assert src_image.shape[-2:] == src_depth.shape[-2:], f'Shape of source image {src_image.shape} does not match shape of source depth {src_depth.shape}'

#     # set up default camera parameters
#     extrinsics_src = np.eye(4) if extrinsics_src is None else extrinsics_src
#     extrinsics_tgt = np.eye(4) if extrinsics_tgt is None else extrinsics_tgt
#     intrinsics_src = intrinsics_tgt if intrinsics_src is None else intrinsics_src
#     intrinsics_tgt = intrinsics_src if intrinsics_tgt is None else intrinsics_tgt
    
#     assert all(x is not None for x in [extrinsics_src, extrinsics_tgt, intrinsics_src, intrinsics_tgt]), "Make sure you have provided all the necessary camera parameters."

#     # check shapes
#     assert extrinsics_src.shape == (4, 4) and extrinsics_tgt.shape == (4, 4)
#     assert intrinsics_src.shape == (3, 3) and intrinsics_tgt.shape == (3, 3) 

#     # convert to view and perspective matrices
#     view_tgt = transforms.extrinsics_to_view(extrinsics_tgt)
#     perspective_tgt = transforms.intrinsics_to_perspective(intrinsics_tgt, near=near, far=far)

#     # unproject depth map
#     uv, faces = utils.image_mesh(*src_depth.shape[-2:])
#     pts = transforms.unproject_cv(uv, src_depth.reshape(-1), extrinsics_src, intrinsics_src)
#     faces = mesh.triangulate(faces, vertices=pts)

#     # rasterize attributes
#     if src_image is not None:
#         attr = src_image.reshape(-1, src_image.shape[-1])
#     else:
#         attr = uv

#     tgt_image, tgt_depth = rasterize_triangles(
#         ctx,
#         width * ssaa,
#         height * ssaa,
#         vertices=pts,
#         faces=faces,
#         attr=attr,
#         transform=perspective_tgt @ view_tgt,
#         cull_backface=cull_backface,
#         return_depth=return_depth,
#     )

#     if ssaa > 1:
#         tgt_image = tgt_image.reshape(height, ssaa, width, ssaa, -1).mean(axis=(1, 3))
#         tgt_depth = tgt_depth.reshape(height, ssaa, width, ssaa, -1).mean(axis=(1, 3)) if return_depth else None

#     return tgt_image, tgt_depth


def test_rasterization(ctx: RastContext):
    """
    Test if rasterization works. It will render a cube with random colors and save it as a CHECKME.png file.
    """
    from .mesh import cube
    vertices, faces = cube(tri=True)
    attributes = np.random.rand(len(vertices), 3).astype(np.float32)
    perspective = transforms.perspective(np.deg2rad(60), 1, 0.01, 100)
    view = transforms.view_look_at(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    
    image, depth = rasterize_triangles(
        ctx, 
        512, 512, 
        vertices=vertices, 
        attributes=attributes, 
        faces=faces, 
        transform=(perspective @ view).astype(np.float32), 
        cull_backface=False,
        return_depth=True,
    )   
    import cv2
    cv2.imwrite('CHECKME.png', cv2.cvtColor((image.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("An image has been saved as ./CHECKME.png")

