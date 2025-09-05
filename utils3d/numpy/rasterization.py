import os
import time
from typing import *
from contextlib import contextmanager

import numpy as np
import moderngl


__all__ = [
    'RastContext',
    'rasterize_triangles',
    'rasterize_triangles_peeling',
    'rasterize_lines',
    'rasterize_point_cloud',
    'sample_texture',
    'test_rasterization',
]


@contextmanager
def timeit():
    start_t = time.time()
    yield
    end_t = time.time()
    print(f"Elapsed time: {end_t - start_t:.4f} seconds")


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

    RASTERIZE_TRIANGLES_VS = """
    #version 430
    uniform mat4 uViewMat;
    uniform mat4 uProjectionMat;
    in vec3 inVert;
    in vec4 inAttr;
    {return_interpolation} in vec2 inUV;
    {flat} out vec4 vAttr;
    out vec3 vViewPos;
    {return_interpolation} out vec2 vUV;
    void main() {
        vec4 viewPos = uViewMat * vec4(inVert, 1.0);
        vAttr = inAttr;
        vViewPos = viewPos.xyz;
        gl_Position = uProjectionMat * viewPos;
        {return_interpolation} vUV = inUV;
    }
    """

    RASTERIZE_TRIANGLES_FS = """
    #version 430
    {flat} in vec4 vAttr;
    in vec2 vUV;
    in vec3 vViewPos;
    out vec4 outAttr;
    {return_interpolation} out int outID;
    {return_interpolation} out vec2 outUV;
    void main() {
        float currDepth = log2(-vViewPos.z) / 64.f + 0.5f;
        outAttr = vAttr;
        {return_interpolation} outID = gl_PrimitiveID;
        {return_interpolation} outUV = vUV;
        gl_FragDepth = currDepth;
    }
    """

    RASTERIZE_TRIANGLES_PEELING_FS = """
    #version 430
    uniform sampler2D prevBufferDepthMap;
    uniform vec2 uScreenSize;
    {flat} in vec4 vAttr;
    in vec2 vUV;
    in vec3 vViewPos;
    out vec4 outAttr;
    {return_interpolation} out int outID;
    {return_interpolation} out vec2 outUV;
    void main() {
        float currDepth = log2(-vViewPos.z) / 64.f + 0.5f;
        float prevDepth = texture(prevBufferDepthMap, gl_FragCoord.xy / uScreenSize).r;
        if (currDepth <= prevDepth) {
            discard; 
        }
        outAttr = vAttr;
        {return_interpolation} outID = gl_PrimitiveID;
        {return_interpolation} outUV = vUV;
        gl_FragDepth = currDepth;
    }
    """

    RASTERIZE_POINT_CLOUD_VS = """
    #version 430
    uniform mat4 uViewMat;
    uniform mat4 uProjectionMat;
    in vec3 inVert;
    in vec4 inAttr;
    in float inPointSize;
    {return_point_id} out int vPointID;
    out vec3 vViewPos;
    out vec4 vAttr;
    out float vPointSize;
    void main() {
        vec4 viewPos = uViewMat * vec4(inVert, 1.0);
        vAttr = inAttr;
        vViewPos = viewPos.xyz;
        vPointSize = inPointSize;
        {return_point_id} vPointID = gl_VertexID;
        gl_Position = uProjectionMat * viewPos;
    }
    """

    RASTERIZE_POINT_CLOUD_GS = """
    #version 430 core
    layout(points) in;
    layout(triangle_strip, max_vertices = {num_point_shape_vertices}) out;
    in vec3 vViewPos[];
    in vec4 vAttr[];
    in float vPointSize[];
    {return_point_id} in int vPointID[];
    flat out vec4 gAttr;
    out vec3 gViewPos;
    flat out vec3 gPointViewPos;
    flat out vec2 gPointCenter2D;
    flat out float gPointSize;
    {return_point_id} flat out int gPointID;
    uniform mat4 uViewMat;
    uniform mat4 uProjectionMat;
    uniform vec2 uScreenSize;
    uniform bool uIsPointSize3D;
    const vec3 triangle[3] = {vec3(0., 0.5, 0.0), vec3(-0.433013, -0.25, 0.0), vec3(0.433013,  -0.25, 0.0)};
    const vec3 square[4] = {vec3(-0.5, -0.5, 0.0), vec3(0.5, -0.5, 0.0), vec3(-0.5,  0.5, 0.0), vec3( 0.5,  0.5, 0.0)};
    const vec3 pentagon[5] = {vec3(0.0, 0.5, 0.0),  vec3(-0.475528, 0.154508, 0.0),  vec3(0.475528, 0.154508, 0.0),  vec3(-0.293893, -0.404508, 0.0),  vec3(0.293893, -0.404508, 0.0)};
    const vec3 hexagon[6] = {vec3(0.0, 0.5, 0.0), vec3(-0.433013, 0.25, 0.0), vec3(0.433013, 0.25, 0.0), vec3(-0.433013, -0.25, 0.0), vec3(0.433013, -0.25, 0.0), vec3(0.0, -0.5, 0.0)};
    const vec3 circle[4] = {vec3(-0.5, -0.5, 0.0), vec3(0.5, -0.5, 0.0), vec3(-0.5,  0.5, 0.0), vec3( 0.5,  0.5, 0.0)};
    void main() {
        mat4 invProjection = inverse(uProjectionMat);
        gAttr = vAttr[0];
        {return_point_id} gPointID = vPointID[0];
        gPointViewPos = vViewPos[0];
        gPointCenter2D = (0.5 * gl_in[0].gl_Position.xy / gl_in[0].gl_Position.w + 0.5) * uScreenSize;
        gPointSize = vPointSize[0];
        for (int i = 0; i < {point_shape}.length(); ++i) {
            vec3 viewPos = uIsPointSize3D ? gPointViewPos + gPointSize * {point_shape}[i] : gPointViewPos;
            vec4 clipCoord = uProjectionMat * vec4(viewPos, 1.0);
            if (!uIsPointSize3D) clipCoord.xy += 2 * gPointSize * clipCoord.w / uScreenSize * {point_shape}[i].xy;
            gViewPos = (invProjection * clipCoord).xyz;
            gl_Position = clipCoord;
            EmitVertex();
        }
        EndPrimitive();
    }
    """

    RASTERIZE_POINT_CLOUD_FS = """
    #version 430 core
    flat  in vec4 gAttr;
    in vec3 gViewPos;
    flat in vec3 gPointViewPos;
    flat in vec2 gPointCenter2D;
    flat in float gPointSize;
    {return_point_id} flat in int gPointID;
    out vec4 outAttr;
    {return_point_id} out int outPointID;
    uniform bool uIsPointSize3D;
    void main() {
        outAttr = gAttr;
        outPointID = gPointID;
        gl_FragDepth = log2(-gViewPos.z) / 64.f + 0.5f;
        {circle_begin} 
        float distSquare;
        if (uIsPointSize3D) {
            vec3 x = gViewPos - gPointViewPos;
            distSquare = dot(x, x);
        }
        else {
            vec2 x = gl_FragCoord.xy - gPointCenter2D;
            distSquare = dot(x, x);   
        }
        if (distSquare > 0.25 * gPointSize * gPointSize)
            discard;
        {circle_end}
    }
    """

    FULL_SCREEN_VS = """
    #version 430 core
    out vec2 screenXY;
    void main() {
        screenXY = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(screenXY * 2.0 - 1.0, 0.0, 1.0);
    }
    """

    CLEAR_TEXTURE_FS = """
    #version 430
    uniform {type} uCleanColor;
    in vec2 screenXY;
    out {type} outColor;
    void main() {
        outColor = uCleanColor;
    }
    """

    SAMPLE_TEXTURE_FS = """
    #version 430
    uniform sampler2D texMap;
    uniform sampler2D uvMap;
    in vec2 screenXY;
    out vec4 outColor;
    void main() {
        outColor = vec4(texture(texMap, texture(uvMap, screenXY).xy));
    }
    """

    DEPTH_LINEAR_TO_BUFFER_FS = """
    #version 430
    uniform sampler2D linearDepthMap;
    in vec2 screenXY;
    out float outBufferDepth;
    void main() {
        float d = texture(linearDepthMap, screenXY).r;
        outBufferDepth = min(1.f, log2(d) / 64.f + 0.5f);
    }
    """

    DEPTH_BUFFER_TO_LINEAR_FS = """
    #version 430
    uniform sampler2D bufferDepthMap;
    in vec2 screenXY;
    out float outLinearDepth;
    void main() {
        float d = texture(bufferDepthMap, screenXY).r;
        outLinearDepth = d == 1.f ? 1.f / 0.f : exp2((d - 0.5f) * 64.f);
    }
    """

    def __init__(self, *args, **kwargs):
        """
        Create a moderngl context.

        ## Parameters
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
        self.shared_objects = {}

    def get_program_triangles(self, flat: bool = False, return_interpolation: bool = False) -> moderngl.Program:
        program_name = f'rasterize_triangles(flat={flat}, return_interpolation={return_interpolation})'
        if program_name not in self.programs:
            self.programs[program_name] = self.mgl_ctx.program(
                vertex_shader=RastContext.RASTERIZE_TRIANGLES_VS.replace('{flat}', 'flat' if flat else '').replace('{return_interpolation}', '' if return_interpolation else '//'),
                fragment_shader=RastContext.RASTERIZE_TRIANGLES_FS.replace('{flat}', 'flat' if flat else '').replace('{return_interpolation}', '' if return_interpolation else '//')
            )
        return self.programs[program_name]

    def get_program_point_cloud(self, return_point_id: bool = False, point_shape: Literal['triangle', 'square', 'pentagon', 'hexagon', 'circle'] = 'square') -> moderngl.Program:
        program_name = f'rasterize_point_cloud(return_point_id={return_point_id}, point_shape={point_shape})'
        num_point_shape_vertices = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6, 'circle': 4}.get(point_shape)
        if program_name not in self.programs:
            self.programs[program_name] = self.mgl_ctx.program(
                vertex_shader=RastContext.RASTERIZE_POINT_CLOUD_VS.replace('{return_point_id}', '' if return_point_id else '//'),
                geometry_shader=RastContext.RASTERIZE_POINT_CLOUD_GS\
                    .replace('{return_point_id}', '' if return_point_id else '//')\
                    .replace('{point_shape}', point_shape)\
                    .replace('{num_point_shape_vertices}', str(num_point_shape_vertices)),
                fragment_shader=RastContext.RASTERIZE_POINT_CLOUD_FS\
                    .replace('{return_point_id}', '' if return_point_id else '//')\
                    .replace('{circle_begin}', '' if point_shape == 'circle' else '/*')\
                    .replace('{circle_end}', '' if point_shape == 'circle' else '*/')
            )
        return self.programs[program_name]
    
    def get_program_triangles_peeling(self, flat: bool = False, return_interpolation: bool = False) -> moderngl.Program:
        program_name = f'rasterize_triangles_peeling(flat={flat}, return_interpolation={return_interpolation})'
        if program_name not in self.programs:
            self.programs[program_name] = self.mgl_ctx.program(
                vertex_shader=RastContext.RASTERIZE_TRIANGLES_VS.replace('{flat}', 'flat' if flat else '').replace('{return_interpolation}', '' if return_interpolation else '//'),
                fragment_shader=RastContext.RASTERIZE_TRIANGLES_PEELING_FS.replace('{flat}', 'flat' if flat else '').replace('{return_interpolation}', '' if return_interpolation else '//')
            )
            self.programs[program_name]['prevBufferDepthMap'] = 0
        return self.programs[program_name]

    def get_program_depth_buffer_to_linear(self) -> moderngl.Program:
        program_name = f'depth_buffer_to_linear'
        if program_name not in self.programs:
            self.programs[program_name] = self.mgl_ctx.program(
                vertex_shader=RastContext.FULL_SCREEN_VS,
                fragment_shader=RastContext.DEPTH_BUFFER_TO_LINEAR_FS
            )
            self.programs[program_name]['bufferDepthMap'] = 0
        return self.programs[program_name]

    def get_program_depth_linear_to_buffer(self) -> moderngl.Program:
        program_name = f'depth_linear_to_buffer'
        if program_name not in self.programs:
            self.programs[program_name] = self.mgl_ctx.program(
                vertex_shader=RastContext.FULL_SCREEN_VS,
                fragment_shader=RastContext.DEPTH_LINEAR_TO_BUFFER_FS
            )
            self.programs[program_name]['linearDepthMap'] = 0
        return self.programs[program_name]

    def get_program_sample_texture(self) -> moderngl.Program:
        program_name = 'sample_texture'
        if program_name not in self.programs:
            self.programs[program_name] = self.mgl_ctx.program(
                vertex_shader=RastContext.FULL_SCREEN_VS,
                fragment_shader=RastContext.SAMPLE_TEXTURE_FS
            )
            self.programs[program_name]['texMap'] = 0
            self.programs[program_name]['uvMap'] = 1
        return self.programs[program_name]

    def get_program_clear_texture(self, dtype: Literal['i4', 'f4']) -> moderngl.Program:
        dtype_to_vec_type = {
            'i4': 'ivec4',
            'f4': 'vec4'
        }
        program_name = f'clear_texture_{dtype}'
        if program_name not in self.programs:
            self.programs[program_name] = self.mgl_ctx.program(
                vertex_shader=RastContext.FULL_SCREEN_VS,
                fragment_shader=RastContext.CLEAR_TEXTURE_FS.replace('{type}', dtype_to_vec_type[dtype])
            )
        return self.programs[program_name]

    def __del__(self):
        try:
            self.mgl_ctx.release()
            for prog_name, prog in self.programs.items():
                prog.release()
            for obj_name, obj in self.shared_objects.items():
                obj.release()
        except:
            pass


def run_full_screen_program(
    ctx: RastContext, 
    prog: moderngl.Program, 
    in_tex: Union[moderngl.Texture, List[moderngl.Texture]], 
    out_tex_or_rbo: Union[Union[moderngl.Texture, moderngl.Renderbuffer], List[Union[moderngl.Texture, moderngl.Renderbuffer]]],
    **uniforms
) -> moderngl.Texture:
    vao = ctx.mgl_ctx.vertex_array(prog, [])
    if isinstance(in_tex, list):
        for i, tex in enumerate(in_tex):
            tex.use(location=i)
    else:
        in_tex.use(location=0)
    for k, v in uniforms.items():   
        prog[k] = v
    if isinstance(out_tex_or_rbo, list):
        fbo = ctx.mgl_ctx.framebuffer(color_attachments=out_tex_or_rbo)
    else:
        fbo = ctx.mgl_ctx.framebuffer(color_attachments=[out_tex_or_rbo])
    fbo.use()
    ctx.mgl_ctx.disable(ctx.mgl_ctx.DEPTH_TEST)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.CULL_FACE)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.BLEND)
    vao.render(moderngl.TRIANGLES, vertices=3)
    vao.release()
    fbo.release()


def clear_texture(ctx: RastContext, tex: moderngl.Texture, value: Union[Tuple[float], Tuple[int], float, int]):
    if tex.depth:
        fbo = ctx.mgl_ctx.framebuffer(color_attachments=[], depth_attachment=tex)
        fbo.clear(depth=value)
        fbo.release()
    elif tex.dtype == 'f4':
        fbo = ctx.mgl_ctx.framebuffer(color_attachments=[tex])
        fbo.clear(color=value)
        fbo.release()
    elif tex.dtype == 'i4':
        fbo = ctx.mgl_ctx.framebuffer(color_attachments=[tex])
        prog = ctx.get_program_clear_texture(tex.dtype)
        vao = ctx.mgl_ctx.vertex_array(prog, [])
        prog['uCleanColor'].value = value + (0,) * (4 - len(value))
        fbo.use()
        vao.render(moderngl.TRIANGLES, vertices=3)
        fbo.release()
        vao.release()


def rasterize_triangles(
    ctx: RastContext,
    width: int,
    height: int,
    *,
    vertices: np.ndarray,
    attributes: Optional[np.ndarray] = None,
    attributes_domain: Optional[Literal['vertex', 'face']] = 'vertex',
    faces: Optional[np.ndarray] = None,
    view: np.ndarray = None,
    projection: np.ndarray = None,
    cull_backface: bool = False,
    return_depth: bool = False,
    return_interpolation: bool = False,
    background_image: Optional[np.ndarray] = None,
    background_depth: Optional[np.ndarray] = None,
    background_interpolation_id: Optional[np.ndarray] = None,
    background_interpolation_uv: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Rasterize triangles.

    ## Parameters
        ctx (RastContext): rasterization context
        width (int): width of rendered image
        height (int): height of rendered image
        vertices (np.ndarray): (N, 3) or (T, 3, 3)
        faces (Optional[np.ndarray]): (T, 3) or None. If `None`, the vertices must be an array with shape (T, 3, 3)
        attributes (np.ndarray): (N, C), (T, 3, C) for vertex domain or (T, C) for face domain
        attributes_domain (Literal['vertex', 'face']): domain of the attributes
        view (np.ndarray): (4, 4) View matrix (world to camera).
        projection (np.ndarray): (4, 4) Projection matrix (camera to clip space).
        cull_backface (bool): whether to cull backface
        background_image (np.ndarray): (H, W, C) background image
        background_depth (np.ndarray): (H, W) background depth
        background_interpolation_id (np.ndarray): (H, W) background triangle ID map
        background_interpolation_uv (np.ndarray): (H, W, 2) background triangle UV (first two channels of barycentric coordinates)

    ## Returns
        A dictionary containing
        
        if attributes is not None
        - `image` (np.ndarray): (H, W, C) float32 rendered image corresponding to the input attributes

        if return_depth is True
        - `depth` (np.ndarray): (H, W) float32 camera space linear depth, ranging from 0 to 1.
        
        if return_interpolation is True
        - `interpolation_id` (np.ndarray): (H, W) int32 triangle ID map
        - `interpolation_uv` (np.ndarray): (H, W, 2) float32 triangle UV (first two channels of barycentric coordinates)
    """
    if faces is None:
        assert vertices.ndim == 3 and vertices.shape[1] == vertices.shape[2] == 3, "If faces is None, vertices must be an array with shape (T, 3, 3)"
    else:
        assert faces.ndim == 2 and faces.shape[1] == 3, f"Faces should be a 2D array with shape (T, 3), but got {faces.shape}"
        assert vertices.ndim == 2 and vertices.shape[1] == 3

    assert vertices.dtype == np.float32

    if attributes is not None:
        assert attributes.dtype == np.float32
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

    assert view is None or (view.shape == (4, 4) and view.dtype == np.float32), f"View should be a 4x4 float32 matrix, but got {view.shape} {view.dtype}"
    assert projection is None or (projection.shape == (4, 4) and projection.dtype == np.float32), f"Projection should be a 4x4 float32 matrix, but got {projection.shape} {projection.dtype}"

    if background_image is not None:
        assert background_image.ndim == 3 and background_image.shape == (height, width, attributes.shape[-1]), f"Image should be a float32 array with shape (H, W, {attributes.shape[1]}), but got {background_image.shape} {background_image.dtype}"
    if background_depth is not None:
        assert background_depth.dtype == np.float32 and background_depth.ndim == 2 and background_depth.shape == (height, width), f"Depth should be a float32 array with shape (H, W), but got {background_depth.shape} {background_depth.dtype}"
    if background_interpolation_id is not None:
        assert background_interpolation_id.dtype == np.int32 and background_interpolation_id.ndim == 2 and background_interpolation_id.shape == (height, width), f"Interpolation ID should be a int32 array with shape (H, W), but got {background_interpolation_id.shape} {background_interpolation_id.dtype}"
    if background_interpolation_uv is not None:
        assert background_interpolation_uv.dtype == np.float32 and background_interpolation_uv.ndim == 3 and background_interpolation_uv.shape == (height, width, 2), f"Interpolation UV should be a float32 array with shape (H, W, 2), but got {background_interpolation_uv.shape} {background_interpolation_uv.dtype}"

    if faces is not None:
        vertices = vertices[faces]
    if attributes is not None:
        num_channels = attributes.shape[-1]
        attributes = np.concatenate([attributes, np.zeros((*attributes.shape[:-1], 4 - num_channels,), dtype=attributes.dtype)], axis=-1) if num_channels < 4 else attributes
        if attributes_domain == 'vertex':
            if faces is not None:
                attributes = attributes[faces]
        elif attributes_domain == 'face':
            attributes = attributes[:, None, :].repeat(3, axis=1)
    if view is None:
        view = np.eye(4, np.float32)
    if projection is None:
        projection = np.eye(4, np.float32) 

    # Get program
    prog = ctx.get_program_triangles(flat=attributes_domain == 'face', return_interpolation=return_interpolation)

    # Create buffers
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
    if attributes is not None:
        vbo_attributes = ctx.mgl_ctx.buffer(np.ascontiguousarray(attributes, dtype='f4'))
    if return_interpolation:
        vbo_uv = ctx.mgl_ctx.buffer(np.ascontiguousarray((np.array([[1., 0.], [0., 1.], [0., 0.]], dtype=np.float32).reshape(1, 3, 2).repeat(vertices.shape[0], axis=0))))
    
    vao = ctx.mgl_ctx.vertex_array(
        prog,
        list(filter(lambda x: x is not None, [
            (vbo_vertices, '3f', 'inVert'),
            (vbo_attributes, f'4f', 'inAttr') if attributes is not None else None,
            (vbo_uv, '2f', 'inUV') if return_interpolation else None,
        ])),
        mode=moderngl.TRIANGLES,
    )

    # Create textures
    image_tex = ctx.mgl_ctx.texture((width, height), 4, dtype='f4', data=np.ascontiguousarray(background_image[::-1, :, :]) if background_image is not None else None)
    buffer_depth_tex = ctx.mgl_ctx.depth_texture((width, height))
    if background_depth is not None:
        linear_depth_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='f4', data=np.ascontiguousarray(background_depth[::-1, :]) if background_depth is not None else None)
        run_full_screen_program(ctx, ctx.get_program_depth_linear_to_buffer(), linear_depth_tex, buffer_depth_tex)
    else:
        if return_depth:
            linear_depth_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='f4')
        else:
            linear_depth_tex = None
        clear_texture(ctx, buffer_depth_tex, value=1.0)

    if return_interpolation:
        interpolation_id_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='i4', data=np.ascontiguousarray(background_interpolation_id[::-1, :]) if background_interpolation_id is not None else None)
        interpolation_uv_tex = ctx.mgl_ctx.texture((width, height), 2, dtype='f4', data=np.ascontiguousarray(background_interpolation_uv[::-1, :, :]) if background_interpolation_uv is not None else None)
    
    clear_texture(ctx, image_tex, value=(0.0, 0.0, 0.0, 1.0))
    if return_interpolation:
        clear_texture(ctx, interpolation_id_tex, value=(-1,))
        clear_texture(ctx, interpolation_uv_tex, value=(0.0, 0.0))

    # Create framebuffer
    if return_interpolation:
        fbo = ctx.mgl_ctx.framebuffer(
            color_attachments=[image_tex, interpolation_id_tex, interpolation_uv_tex],
            depth_attachment=buffer_depth_tex,
        )
    else:
        fbo = ctx.mgl_ctx.framebuffer(
            color_attachments=[image_tex],
            depth_attachment=buffer_depth_tex,
        )
    fbo.viewport = (0, 0, width, height)

    # Set uniforms
    prog['uViewMat'].write(np.ascontiguousarray(view.transpose().astype('f4')))
    prog['uProjectionMat'].write(np.ascontiguousarray(projection.transpose().astype('f4')))

    # Set render states
    ctx.mgl_ctx.depth_func = '<'
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    if cull_backface:
        ctx.mgl_ctx.enable(ctx.mgl_ctx.CULL_FACE)
    else:
        ctx.mgl_ctx.disable(ctx.mgl_ctx.CULL_FACE)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.BLEND)
    
    # Render
    fbo.use()
    vao.render()

    # Read
    if attributes is not None:
        image = np.frombuffer(image_tex.read(), dtype='f4').reshape((height, width, 4))
        image = np.flip(image, axis=0)
        image = image[:, :, :num_channels]
    else:
        image = None
    if return_depth:
        run_full_screen_program(ctx, ctx.get_program_depth_buffer_to_linear(), buffer_depth_tex, linear_depth_tex)
        depth = np.frombuffer(linear_depth_tex.read(), dtype='f4').reshape((height, width))
        depth = np.flip(depth, axis=0)
    else:
        depth = None
    if return_interpolation:
        interpolation_id = np.frombuffer(interpolation_id_tex.read(), dtype='i4').reshape((height, width))
        interpolation_id = np.flip(interpolation_id, axis=0)
        interpolation_uv = np.frombuffer(interpolation_uv_tex.read(), dtype='f4').reshape((height, width, 2))
        interpolation_uv = np.flip(interpolation_uv, axis=0)
    else:
        interpolation_id = None
        interpolation_uv = None

    # Release
    vao.release()
    vbo_vertices.release()
    if attributes is not None:
        vbo_attributes.release()
    if return_interpolation:
        vbo_uv.release()
    fbo.release()
    image_tex.release()
    buffer_depth_tex.release()
    if background_depth is not None or return_depth:
        linear_depth_tex.release()
    if return_interpolation:
        interpolation_id_tex.release()
        interpolation_uv_tex.release()

    output = {
        "image": image,
        "depth": depth,
        "interpolation_id": interpolation_id,
        "interpolation_uv": interpolation_uv
    }

    output = {k: v for k, v in output.items() if v is not None}

    return output


@contextmanager
def rasterize_triangles_peeling(
    ctx: RastContext,
    width: int,
    height: int,
    *,
    vertices: np.ndarray,
    attributes: np.ndarray,
    attributes_domain: Literal['vertex', 'face'] = 'vertex',
    faces: Optional[np.ndarray] = None,
    view: np.ndarray = None,
    projection: np.ndarray = None,
    cull_backface: bool = False,
    return_depth: bool = False,
    return_interpolation: bool = False,
) -> Iterator[Iterator[Dict[str, np.ndarray]]]:
    """
    Rasterize triangles with depth peeling.

    ## Parameters
        ctx (RastContext): rasterization context
        width (int): width of rendered image
        height (int): height of rendered image
        vertices (np.ndarray): (N, 3) or (T, 3, 3)
        faces (Optional[np.ndarray]): (T, 3) or None. If `None`, the vertices must be an array with shape (T, 3, 3)
        attributes (np.ndarray): (N, C), (T, 3, C) for vertex domain or (T, C) for face domain
        attributes_domain (Literal['vertex', 'face']): domain of the attributes
        view (np.ndarray): (4, 4) View matrix (world to camera).
        projection (np.ndarray): (4, 4) Projection matrix (camera to clip space).
        cull_backface (bool): whether to cull backface
    ## Returns
        A context manager of generator of dictionary containing
        
        if attributes is not None
        - `image` (np.ndarray): (H, W, C) float32 rendered image corresponding to the input attributes

        if return_depth is True
        - `depth` (np.ndarray): (H, W) float32 camera space linear depth, ranging from 0 to 1.
        
        if return_interpolation is True
        - `interpolation_id` (np.ndarray): (H, W) int32 triangle ID map
        - `interpolation_uv` (np.ndarray): (H, W, 2) float32 triangle UV (first two channels of barycentric coordinates)
    
    ## Example Usage
    ```
    with rasterize_triangles_peeling(
        ctx, 
        512, 512, 
        vertices=vertices, 
        faces=faces, 
        attributes=attributes,
        view=view,
        projection=projection
    ) as peeler:
        for i, layer_output in zip(range(3, peeler)):
            print(f"Layer {i}:")
            for key, value in layer_output.items():
                print(f"  {key}: {value.shape}")
    ```
    """
    if faces is None:
        assert vertices.ndim == 3 and vertices.shape[1] == vertices.shape[2] == 3, "If faces is None, vertices must be an array with shape (T, 3, 3)"
    else:
        assert faces.ndim == 2 and faces.shape[1] == 3, f"Faces should be a 2D array with shape (T, 3), but got {faces.shape}"
        assert vertices.ndim == 2 and vertices.shape[1] == 3

    assert vertices.dtype == np.float32

    if attributes is not None:
        assert attributes.dtype == np.float32
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

    assert view is None or (view.shape == (4, 4) and view.dtype == np.float32), f"View should be a 4x4 float32 matrix, but got {view.shape} {view.dtype}"
    assert projection is None or (projection.shape == (4, 4) and projection.dtype == np.float32), f"Projection should be a 4x4 float32 matrix, but got {projection.shape} {projection.dtype}"

    if faces is not None:
        vertices = vertices[faces]
    if attributes is not None:
        num_channels = attributes.shape[-1]
        attributes = np.concatenate([attributes, np.zeros((*attributes.shape[:-1], 4 - num_channels,), dtype=attributes.dtype)], axis=-1) if num_channels < 4 else attributes
        if attributes_domain == 'vertex':
            if faces is not None:
                attributes = attributes[faces]
        elif attributes_domain == 'face':
            attributes = attributes[:, None, :].repeat(3, axis=1)
    if view is None:
        view = np.eye(4, np.float32)
    if projection is None:
        projection = np.eye(4, np.float32) 
    
    # Get program
    prog = ctx.get_program_triangles_peeling(flat=attributes_domain == 'face', return_interpolation=return_interpolation)

    # Create buffers
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
    if attributes is not None:
        vbo_attributes = ctx.mgl_ctx.buffer(np.ascontiguousarray(attributes, dtype='f4'))
    if return_interpolation:
        vbo_uv = ctx.mgl_ctx.buffer(np.ascontiguousarray((np.array([[1., 0.], [0., 1.], [0., 0.]], dtype=np.float32).reshape(1, 3, 2).repeat(vertices.shape[0], axis=0))))
    
    vao = ctx.mgl_ctx.vertex_array(
        prog,
        list(filter(lambda x: x is not None, [
            (vbo_vertices, '3f', 'inVert'),
            (vbo_attributes, f'4f', 'inAttr') if attributes is not None else None,
            (vbo_uv, '2f', 'inUV') if return_interpolation else None,
        ])),
        mode=moderngl.TRIANGLES,
    )

    # Create textures
    image_tex = ctx.mgl_ctx.texture((width, height), 4, dtype='f4')
    buffer_depth_tex_a = ctx.mgl_ctx.depth_texture((width, height))
    buffer_depth_tex_b = ctx.mgl_ctx.depth_texture((width, height))
    if return_depth:
        linear_depth_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='f4')
    if return_interpolation:
        interpolation_id_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='i4')
        interpolation_uv_tex = ctx.mgl_ctx.texture((width, height), 2, dtype='f4')

    # Create frame buffers
    if return_interpolation:
        color_attachments=[image_tex, interpolation_id_tex, interpolation_uv_tex]
    else:
        color_attachments=[image_tex]
    fbo_curr = ctx.mgl_ctx.framebuffer(
        color_attachments=color_attachments,
        depth_attachment=buffer_depth_tex_a,
    )
    fbo_curr.viewport = (0, 0, width, height)
    fbo_prev = ctx.mgl_ctx.framebuffer(
        color_attachments=color_attachments,
        depth_attachment=buffer_depth_tex_b,
    )
    fbo_prev.viewport = (0, 0, width, height)

    # Set uniforms
    prog['uViewMat'].write(np.ascontiguousarray(view.transpose().astype('f4')))
    prog['uProjectionMat'].write(np.ascontiguousarray(projection.transpose().astype('f4')))
    prog['uScreenSize'].value = (width, height)

    # Rendering settings
    ctx.mgl_ctx.depth_func = '<'
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    if cull_backface:
        ctx.mgl_ctx.enable(ctx.mgl_ctx.CULL_FACE)
    else:
        ctx.mgl_ctx.disable(ctx.mgl_ctx.CULL_FACE)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.BLEND)

    # Initialize prev depth texture = 0
    clear_texture(ctx, fbo_prev.depth_attachment, value=0.0)
    
    def generator():
        nonlocal fbo_curr, fbo_prev
        while True:
            # Clear
            clear_texture(ctx, image_tex, value=(0.0, 0.0, 0.0, 1.0))
            clear_texture(ctx, fbo_curr.depth_attachment, value=1.0)
            if return_interpolation:
                clear_texture(ctx, interpolation_id_tex, value=(-1,))
                clear_texture(ctx, interpolation_uv_tex, value=(0.0, 0.0))

            # Render
            fbo_prev.depth_attachment.use(location=0) 
            fbo_curr.use()
            vao.render()

            # Read
            if attributes is not None:
                image = np.frombuffer(image_tex.read(), dtype='f4').reshape((height, width, 4))
                image = np.flip(image, axis=0)
                image = image[:, :, :num_channels]
            else:
                image = None
            if return_depth:
                run_full_screen_program(ctx, ctx.get_program_depth_buffer_to_linear(), fbo_curr.depth_attachment, linear_depth_tex)
                depth = np.frombuffer(linear_depth_tex.read(), dtype='f4').reshape((height, width))
                depth = np.flip(depth, axis=0)
            else:
                depth = None
            if return_interpolation:
                interpolation_id = np.frombuffer(interpolation_id_tex.read(), dtype='i4').reshape((height, width))
                interpolation_id = np.flip(interpolation_id, axis=0)
                interpolation_uv = np.frombuffer(interpolation_uv_tex.read(), dtype='f4').reshape((height, width, 2))
                interpolation_uv = np.flip(interpolation_uv, axis=0)
            else:
                interpolation_id = None
                interpolation_uv = None

            # Yield
            output = {
                "image": image,
                "depth": depth,
                "interpolation_id": interpolation_id,
                "interpolation_uv": interpolation_uv
            }
            output = {k: v for k, v in output.items() if v is not None}
            yield output

            # Swap curr and prev fbos
            fbo_curr, fbo_prev = fbo_prev, fbo_curr
    
    try:
        yield generator()
    finally:
        # Release
        vao.release()
        vbo_vertices.release()
        vbo_attributes.release()
        fbo_curr.release()
        fbo_prev.release()
        image_tex.release()
        buffer_depth_tex_a.release()
        buffer_depth_tex_b.release()
        if return_depth:
            linear_depth_tex.release()
        if return_interpolation:
            interpolation_id_tex.release()
            interpolation_uv_tex.release()


def rasterize_lines(
    ctx: RastContext,
    width: int,
    height: int,
    *,
    vertices: np.ndarray,
    lines: np.ndarray,
    attributes: Optional[np.ndarray],
    attributes_domain: Literal['vertex', 'line'] = 'vertex',
    view: Optional[np.ndarray] = None,
    projection: Optional[np.ndarray] = None,
    line_width: float = 1.0,
    return_depth: bool = False,
    return_interpolation: bool = False,
    background_image: Optional[np.ndarray] = None,
    background_depth: Optional[np.ndarray] = None,
    background_interpolation_id: Optional[np.ndarray] = None,
    background_interpolation_uv: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, ...]:
    """
    Rasterize lines.

    ## Parameters
        ctx (RastContext): rasterization context
        width (int): width of rendered image
        height (int): height of rendered image
        vertices (np.ndarray): (N, 3) or (T, 3, 3)
        faces (Optional[np.ndarray]): (T, 3) or None. If `None`, the vertices must be an array with shape (T, 3, 3)
        attributes (np.ndarray): (N, C), (T, 3, C) for vertex domain or (T, C) for face domain
        attributes_domain (Literal['vertex', 'face']): domain of the attributes
        view (np.ndarray): (4, 4) View matrix (world to camera).
        projection (np.ndarray): (4, 4) Projection matrix (camera to clip space).
        cull_backface (bool): whether to cull backface
        background_image (np.ndarray): (H, W, C) background image
        background_depth (np.ndarray): (H, W) background depth
        background_interpolation_id (np.ndarray): (H, W) background triangle ID map
        background_interpolation_uv (np.ndarray): (H, W, 2) background triangle UV (first two channels of barycentric coordinates)

    ## Returns
        A dictionary containing
        
        if attributes is not None
        - `image` (np.ndarray): (H, W, C) float32 rendered image corresponding to the input attributes

        if return_depth is True
        - `depth` (np.ndarray): (H, W) float32 camera space linear depth, ranging from 0 to 1.
        
        if return_interpolation is True
        - `interpolation_id` (np.ndarray): (H, W) int32 triangle ID map
        - `interpolation_uv` (np.ndarray): (H, W, 2) float32 triangle UV (first two channels of barycentric coordinates)
    """
    if lines is None:
        assert vertices.ndim == 3 and vertices.shape[1] == 2 and vertices.shape[2] == 3, "If lines is None, vertices must be an array with shape (T, 2, 3)"
    else:
        assert lines.ndim == 2 and lines.shape[1] == 2, f"Lines should be a int32 or uint32 array with shape (T, 2), but got {lines.shape} {lines.dtype}"
        assert lines.dtype == np.uint32 or lines.dtype == np.int32
        assert vertices.ndim == 2 and vertices.shape[1] == 3

    assert vertices.dtype == np.float32

    if attributes is not None:
        assert attributes.dtype == np.float32
        if attributes_domain == 'vertex':
            assert (attributes.shape[:-1] == vertices.shape[:-1]), f"Attribute shape {attributes.shape} does not match vertex shape {vertices.shape}"
        elif attributes_domain == 'line':
            if lines is None:
                assert attributes.shape[0] == vertices.shape[0], f"Attribute shape {attributes.shape} does not match vertex shape {vertices.shape}"
            else:
                assert attributes.shape[0] == lines.shape[0], f"Attribute shape {attributes.shape} does not match line shape {lines.shape}"
        else:
            raise ValueError(f"Unknown attributes domain: {attributes_domain}")

        assert attributes.shape[-1] in [1, 2, 3, 4], f'Vertex attribute only supports channels 1, 2, 3, 4, but got {attributes.shape[-1]}'

    assert view is None or (view.shape == (4, 4) and view.dtype == np.float32), f"View should be a 4x4 float32 matrix, but got {view.shape} {view.dtype}"
    assert projection is None or (projection.shape == (4, 4) and projection.dtype == np.float32), f"Projection should be a 4x4 float32 matrix, but got {projection.shape} {projection.dtype}"

    if background_image is not None:
        assert background_image.ndim == 3 and background_image.shape == (height, width, attributes.shape[-1]), f"Image should be a float32 array with shape (H, W, {attributes.shape[1]}), but got {background_image.shape} {background_image.dtype}"
    if background_depth is not None:
        assert background_depth.dtype == np.float32 and background_depth.ndim == 2 and background_depth.shape == (height, width), f"Depth should be a float32 array with shape (H, W), but got {background_depth.shape} {background_depth.dtype}"
    if background_interpolation_id is not None:
        assert background_interpolation_id.dtype == np.int32 and background_interpolation_id.ndim == 2 and background_interpolation_id.shape == (height, width), f"Interpolation ID should be a int32 array with shape (H, W), but got {background_interpolation_id.shape} {background_interpolation_id.dtype}"
    if background_interpolation_uv is not None:
        assert background_interpolation_uv.dtype == np.float32 and background_interpolation_uv.ndim == 3 and background_interpolation_uv.shape == (height, width, 2), f"Interpolation UV should be a float32 array with shape (H, W, 2), but got {background_interpolation_uv.shape} {background_interpolation_uv.dtype}"

    if lines is not None:
        vertices = vertices[lines]
    if attributes is not None:
        num_channels = attributes.shape[-1]
        attributes = np.concatenate([attributes, np.zeros((*attributes.shape[:-1], 4 - num_channels,), dtype=attributes.dtype)], axis=-1) if num_channels < 4 else attributes
        if attributes_domain == 'vertex':
            if lines is not None:
                attributes = attributes[lines]
        elif attributes_domain == 'line':
            attributes = attributes[:, None, :].repeat(2, axis=1)
    if view is None:
        view = np.eye(4, np.float32)
    if projection is None:
        projection = np.eye(4, np.float32) 

    # Get program
    prog = ctx.get_program_triangles(flat=attributes_domain == 'line', return_interpolation=return_interpolation)

    # Create buffers
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(vertices, dtype='f4'))
    if attributes is not None:
        vbo_attributes = ctx.mgl_ctx.buffer(np.ascontiguousarray(attributes, dtype='f4'))
    if return_interpolation:
        vbo_uv = ctx.mgl_ctx.buffer(np.ascontiguousarray((np.array([[1., 0.], [0., 1.]], dtype=np.float32).reshape(1, 2, 2).repeat(vertices.shape[0], axis=0))))
    
    vao = ctx.mgl_ctx.vertex_array(
        prog,
        list(filter(lambda x: x is not None, [
            (vbo_vertices, '3f', 'inVert'),
            (vbo_attributes, f'4f', 'inAttr') if attributes is not None else None,
            (vbo_uv, '2f', 'inUV') if return_interpolation else None,
        ])),
        mode=moderngl.LINES,
    )

    # Create textures
    image_tex = ctx.mgl_ctx.texture((width, height), 4, dtype='f4', data=np.ascontiguousarray(background_image[::-1, :, :]) if background_image is not None else None)
    buffer_depth_tex = ctx.mgl_ctx.depth_texture((width, height))
    if background_depth is not None:
        linear_depth_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='f4', data=np.ascontiguousarray(background_depth[::-1, :]) if background_depth is not None else None)
        run_full_screen_program(ctx, ctx.get_program_depth_linear_to_buffer(), linear_depth_tex, buffer_depth_tex)
    else:
        if return_depth:
            linear_depth_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='f4')
        else:
            linear_depth_tex = None
        clear_texture(ctx, buffer_depth_tex, value=1.0)

    if return_interpolation:
        interpolation_id_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='i4', data=np.ascontiguousarray(background_interpolation_id[::-1, :]) if background_interpolation_id is not None else None)
        interpolation_uv_tex = ctx.mgl_ctx.texture((width, height), 2, dtype='f4', data=np.ascontiguousarray(background_interpolation_uv[::-1, :, :]) if background_interpolation_uv is not None else None)
    
    clear_texture(ctx, image_tex, value=(0.0, 0.0, 0.0, 1.0))
    if return_interpolation:
        clear_texture(ctx, interpolation_id_tex, value=(-1,))
        clear_texture(ctx, interpolation_uv_tex, value=(0.0, 0.0))

    # Create framebuffer
    if return_interpolation:
        fbo = ctx.mgl_ctx.framebuffer(
            color_attachments=[image_tex, interpolation_id_tex, interpolation_uv_tex],
            depth_attachment=buffer_depth_tex,
        )
    else:
        fbo = ctx.mgl_ctx.framebuffer(
            color_attachments=[image_tex],
            depth_attachment=buffer_depth_tex,
        )
    fbo.viewport = (0, 0, width, height)

    # Set uniforms
    prog['uViewMat'].write(np.ascontiguousarray(view.transpose().astype('f4')))
    prog['uProjectionMat'].write(np.ascontiguousarray(projection.transpose().astype('f4')))

    # Set render states
    ctx.mgl_ctx.depth_func = '<'
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.BLEND)
    ctx.mgl_ctx.line_width = line_width

    # Render
    fbo.use()
    vao.render()

    # Read
    if attributes is not None:
        image = np.frombuffer(image_tex.read(), dtype='f4').reshape((height, width, 4))
        image = np.flip(image, axis=0)
        image = image[:, :, :num_channels]
    else:
        image = None
    if return_depth:
        run_full_screen_program(ctx, ctx.get_program_depth_buffer_to_linear(), buffer_depth_tex, linear_depth_tex)
        depth = np.frombuffer(linear_depth_tex.read(), dtype='f4').reshape((height, width))
        depth = np.flip(depth, axis=0)
    else:
        depth = None
    if return_interpolation:
        interpolation_id = np.frombuffer(interpolation_id_tex.read(), dtype='i4').reshape((height, width))
        interpolation_id = np.flip(interpolation_id, axis=0)
        interpolation_uv = np.frombuffer(interpolation_uv_tex.read(), dtype='f4').reshape((height, width, 2))
        interpolation_uv = np.flip(interpolation_uv, axis=0)
    else:
        interpolation_id = None
        interpolation_uv = None

    # Release
    vao.release()
    vbo_vertices.release()
    if attributes is not None:
        vbo_attributes.release()
    if return_interpolation:
        vbo_uv.release()
    fbo.release()
    image_tex.release()
    buffer_depth_tex.release()
    if background_depth is not None or return_depth:
        linear_depth_tex.release()
    if return_interpolation:
        interpolation_id_tex.release()
        interpolation_uv_tex.release()

    output = {
        "image": image,
        "depth": depth,
        "interpolation_id": interpolation_id,
        "interpolation_uv": interpolation_uv
    }

    output = {k: v for k, v in output.items() if v is not None}

    return output


def rasterize_point_cloud(
    ctx: RastContext,
    width: int,
    height: int,
    *,
    points: np.ndarray,
    point_sizes: Union[float, np.ndarray] = 10,
    point_size_in: Literal['2d', '3d'] = '2d',
    point_shape: Literal['triangle', 'square', 'pentagon', 'hexagon', 'circle'] = 'square',
    attributes: Optional[np.ndarray] = None,
    view: np.ndarray = None,
    projection: np.ndarray = None,
    return_depth: bool = False,
    return_point_id: bool = False,
    background_image: Optional[np.ndarray] = None,
    background_depth: Optional[np.ndarray] = None,
    background_point_id: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Rasterize point cloud.

    ## Parameters
        ctx (RastContext): rasterization context
        width (int): width of rendered image
        height (int): height of rendered image
        points (np.ndarray): (N, 3)
        point_sizes (np.ndarray): (N,) or float
        point_size_in: Literal['2d', '3d'] = '2d'. Whether the point sizes are in 2D (screen space measured in pixels) or 3D (world space measured in scene units).
        point_shape: Literal['triangle', 'square', 'pentagon', 'hexagon', 'circle'] = 'square'. The visual shape of the points.
        attributes (np.ndarray): (N, C)
        view (np.ndarray): (4, 4) View matrix (world to camera).
        projection (np.ndarray): (4, 4) Projection matrix (camera to clip space).
        cull_backface (bool): whether to cull backface,
        return_depth (bool): whether to return depth map
        return_point_id (bool): whether to return point ID map
        background_image (np.ndarray): (H, W, C) background image
        background_depth (np.ndarray): (H, W) background depth
        background_point_id (np.ndarray): (H, W) background point ID map

    ## Returns
        A dictionary containing
        
        if attributes is not None
        - `image` (np.ndarray): (H, W, C) float32 rendered image corresponding to the input attributes

        if return_depth is True
        - `depth` (np.ndarray): (H, W) float32 camera space linear depth, ranging from 0 to 1.
        
        if return_point_id is True
        - `point_id` (np.ndarray): (H, W) int32 point ID map
    """
    assert points.ndim == 2 and points.shape[1] == 3 and points.dtype == np.float32, f"Points should be a float32 array with shape (N, 3), but got {points.shape} {points.dtype}"

    if attributes is not None:
        assert attributes.dtype == np.float32
        assert (attributes.shape[:-1] == points.shape[:-1]), f"Attribute shape {attributes.shape} does not match point shape {points.shape}"
        assert attributes.shape[-1] in [1, 2, 3, 4], f'Vertex attribute only supports channels 1, 2, 3, 4, but got {attributes.shape[-1]}'

    assert view is None or (view.shape == (4, 4) and view.dtype == np.float32), f"View should be a 4x4 float32 matrix, but got {view.shape} {view.dtype}"
    assert projection is None or (projection.shape == (4, 4) and projection.dtype == np.float32), f"Projection should be a 4x4 float32 matrix, but got {projection.shape} {projection.dtype}"

    if background_image is not None:
        assert background_image.ndim == 3 and background_image.shape == (height, width, attributes.shape[-1]), f"Image should be a float32 array with shape (H, W, {attributes.shape[1]}), but got {background_image.shape} {background_image.dtype}"
    if background_depth is not None:
        assert background_depth.dtype == np.float32 and background_depth.ndim == 2 and background_depth.shape == (height, width), f"Depth should be a float32 array with shape (H, W), but got {background_depth.shape} {background_depth.dtype}"
    if background_point_id is not None:
        assert background_point_id.dtype == np.int32 and background_point_id.ndim == 2 and background_point_id.shape == (height, width), f"Point ID should be a int32 array with shape (H, W), but got {background_point_id.shape} {background_point_id.dtype}"

    if not isinstance(point_sizes, np.ndarray):
        point_sizes = np.full((points.shape[0],), point_sizes, dtype=np.float32)

    if attributes is not None:
        num_channels = attributes.shape[-1]
        attributes = np.concatenate([attributes, np.zeros((*attributes.shape[:-1], 4 - num_channels,), dtype=attributes.dtype)], axis=-1) if num_channels < 4 else attributes

    if view is None:
        view = np.eye(4, np.float32)
    if projection is None:
        projection = np.eye(4, np.float32) 

    # Get program
    prog = ctx.get_program_point_cloud(return_point_id=return_point_id, point_shape=point_shape)
    
    # Create buffers
    vbo_vertices = ctx.mgl_ctx.buffer(np.ascontiguousarray(points, dtype='f4'))
    if attributes is not None:
        vbo_attributes = ctx.mgl_ctx.buffer(np.ascontiguousarray(attributes, dtype='f4'))
    vbo_point_sizes = ctx.mgl_ctx.buffer(np.ascontiguousarray(point_sizes, dtype='f4'))

    vao = ctx.mgl_ctx.vertex_array(
        prog,
        list(filter(lambda x: x is not None, [
            (vbo_vertices, '3f', 'inVert'),
            (vbo_attributes, f'4f', 'inAttr') if attributes is not None else None,
            (vbo_point_sizes, '1f', 'inPointSize') if point_sizes is not None else None,
        ]))
    )

    # Create textures
    image_tex = ctx.mgl_ctx.texture((width, height), 4, dtype='f4', data=np.ascontiguousarray(background_image[::-1, :, :]) if background_image is not None else None)
    buffer_depth_tex = ctx.mgl_ctx.depth_texture((width, height))
    if background_depth is not None:
        linear_depth_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='f4', data=np.ascontiguousarray(background_depth[::-1, :]) if background_depth is not None else None)
        run_full_screen_program(ctx, ctx.get_program_depth_linear_to_buffer(), linear_depth_tex, buffer_depth_tex)
    else:
        if return_depth:
            linear_depth_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='f4')
        else:
            linear_depth_tex = None
        clear_texture(ctx, buffer_depth_tex, value=1.0)

    if return_point_id:
        point_id_tex = ctx.mgl_ctx.texture((width, height), 1, dtype='i4', data=np.ascontiguousarray(background_point_id[::-1, :]) if background_point_id is not None else None)

    clear_texture(ctx, image_tex, value=(0.0, 0.0, 0.0, 1.0))
    if return_point_id:
        clear_texture(ctx, point_id_tex, value=(-1,))

    # Create framebuffer
    if return_point_id:
        fbo = ctx.mgl_ctx.framebuffer(
            color_attachments=[image_tex, point_id_tex],
            depth_attachment=buffer_depth_tex,
        )
    else:
        fbo = ctx.mgl_ctx.framebuffer(
            color_attachments=[image_tex],
            depth_attachment=buffer_depth_tex,
        )
    fbo.viewport = (0, 0, width, height)

    # Set uniforms
    prog['uViewMat'].write(np.ascontiguousarray(view.transpose().astype('f4')))
    prog['uProjectionMat'].write(np.ascontiguousarray(projection.transpose().astype('f4')))
    prog['uScreenSize'].value = (width, height)
    prog['uIsPointSize3D'].value = point_size_in == '3d'

    # Set render states
    ctx.mgl_ctx.depth_func = '<'
    ctx.mgl_ctx.enable(ctx.mgl_ctx.DEPTH_TEST)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.CULL_FACE)
    ctx.mgl_ctx.disable(ctx.mgl_ctx.BLEND)
    
    # Render
    fbo.use()
    vao.render(mode=moderngl.POINTS)

    # Read
    if attributes is not None:
        image = np.frombuffer(image_tex.read(), dtype='f4').reshape((height, width, 4))
        image = np.flip(image, axis=0)
        image = image[:, :, :num_channels]
    else:
        image = None
    if return_depth:
        run_full_screen_program(ctx, ctx.get_program_depth_buffer_to_linear(), buffer_depth_tex, linear_depth_tex)
        depth = np.frombuffer(linear_depth_tex.read(), dtype='f4').reshape((height, width))
        depth = np.flip(depth, axis=0)
    else:
        depth = None
    if return_point_id:
        point_id = np.frombuffer(point_id_tex.read(), dtype='i4').reshape((height, width))
        point_id = np.flip(point_id, axis=0)
    else:
        point_id = None

    # Release
    vao.release()
    vbo_vertices.release()
    if attributes is not None:
        vbo_attributes.release()
    vbo_point_sizes.release()
    fbo.release()
    image_tex.release()
    buffer_depth_tex.release()
    if background_depth is not None or return_depth:
        linear_depth_tex.release()
    if return_point_id:
        point_id_tex.release()

    output = {
        "image": image,
        "depth": depth,
        "point_id": point_id
    }

    output = {k: v for k, v in output.items() if v is not None}

    return output


def sample_texture(
    ctx: RastContext,
    uv_map: np.ndarray,
    texture_map: np.ndarray,
    interpolation: Literal['linear', 'nearest'] = 'linear',
    mipmap_level: Union[int, Tuple[int, int]] = 0,
    repeat: Union[bool, Tuple[bool, bool]] = False,
    anisotropic: float = 1.0
) -> np.ndarray:
    """
    Sample from a texture map with a UV map.
    """
    assert len(texture_map.shape) == 3 and 1 <= texture_map.shape[2] <= 4
    assert uv_map.shape[2] == 2
    height, width = uv_map.shape[:2]
    texture_dtype = map_np_dtype(texture_map.dtype)

    # Create texture
    texture_tex = ctx.mgl_ctx.texture((texture_map.shape[1], texture_map.shape[0]), texture_map.shape[2], dtype=texture_dtype, data=np.ascontiguousarray(texture_map))
    if isinstance(mipmap_level, tuple):
        base_mipmap_level, max_mipmap_level = mipmap_level
    else:
        base_mipmap_level, max_mipmap_level = 0, mipmap_level
    if base_mipmap_level > 0 or max_mipmap_level > 0:
        use_mipmap = True
        texture_tex.build_mipmaps(base_mipmap_level, max_mipmap_level)
    else:
        use_mipmap = False
    if interpolation == 'linear':
        texture_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR if use_mipmap else moderngl.LINEAR, moderngl.LINEAR)
    elif interpolation == 'nearest':
        texture_tex.filter = (moderngl.NEAREST_MIPMAP_NEAREST if use_mipmap else moderngl.NEAREST, moderngl.NEAREST)
    if isinstance(repeat, tuple):
        texture_tex.repeat_x, texture_tex.repeat_y = repeat
    else:
        texture_tex.repeat_x = texture_tex.repeat_y = repeat
    texture_tex.anisotropy = anisotropic

    uv_tex = ctx.mgl_ctx.texture((width, height), 2, dtype='f4', data=np.ascontiguousarray(uv_map.astype('f4', copy=False)))
    uv_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

    # Create render buffer and frame buffer
    output_tex = ctx.mgl_ctx.texture((uv_map.shape[1], uv_map.shape[0]), texture_map.shape[2], dtype=texture_dtype)

    # Render
    run_full_screen_program(ctx, ctx.get_program_sample_texture(), [texture_tex, uv_tex], output_tex)

    # Read buffer
    image_buffer = np.frombuffer(output_tex.read(), dtype=texture_dtype).reshape((height, width, texture_tex.shape[2]))

    # Release
    texture_tex.release()
    output_tex.release()
    uv_tex.release()

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

#     ## Parameters
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
    
#     ## Returns
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
#     faces = mesh.triangulate_mesh(faces, vertices=pts)

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
    from .transforms import perspective_from_fov, view_look_at

    vertices, faces = cube(tri=True)
    attributes = np.random.rand(len(vertices), 3).astype(np.float32)
    projection = perspective_from_fov(fov_x=np.deg2rad(60), aspect_ratio=1, near=1e-8, far=100000)
    view = view_look_at([1, 2, 2], [0, 0, 0], [0, 1, 0])
    out = rasterize_triangles(
        ctx, 
        512, 512, 
        vertices=vertices, 
        attributes=attributes, 
        faces=faces, 
        view=view, 
        projection=projection,
        cull_backface=True,
        return_depth=True,
        return_interpolation=False,
    )
    image = out['image']
    image = np.concatenate([image, np.zeros((*image.shape[:-1], 1), dtype=image.dtype)], axis=-1)
    import cv2
    cv2.imwrite('CHECKME.png', cv2.cvtColor((image.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("An image has been saved as ./CHECKME.png")


# def test_rasterization(ctx: RastContext):
#     """
#     Test if rasterization works. It will render a cube with random colors and save it as a CHECKME.png file.
#     """
#     from .mesh import cube
#     from .transforms import perspective_from_fov, view_look_at, project, unproject

#     vertices, faces = cube(tri=True)
#     attributes = np.random.rand(len(vertices), 3).astype(np.float32)
#     projection = perspective_from_fov(fov_x=np.deg2rad(60), aspect_ratio=1, near=0.1, far=100000)
#     view = view_look_at([1, 2, 2], [0, 0, 0], [0, 1, 0])
#     for _ in range(20):
#         with timeit():
#             out = rasterize_point_cloud(
#                 ctx, 
#                 512, 512, 
#                 points=vertices, 
#                 point_sizes=1,
#                 point_size_in='3d',
#                 point_shape='circle',
#                 attributes=attributes, 
#                 view=view, 
#                 projection=projection,
#                 return_depth=True,
#                 return_point_id=True,
#             )
#     image = out['image']
#     print(np.unique(image))
#     image = np.concatenate([image, np.zeros((*image.shape[:2], 4 - image.shape[2]), dtype=image.dtype)], axis=-1)
#     import cv2
#     cv2.imwrite('CHECKME.png', cv2.cvtColor((image.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
#     print("An image has been saved as ./CHECKME.png")