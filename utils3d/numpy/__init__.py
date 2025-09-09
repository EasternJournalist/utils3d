"""
3D utility functions workings with NumPy.
"""
import importlib
import itertools
import numpy
from typing import TYPE_CHECKING


__modules_all__ = {
    'utils': [
        'sliding_window',
        'max_pool_1d',
        'max_pool_2d',
        'max_pool_nd',
        'lookup',
        'segment_roll',
        'csr_matrix_from_indices',
    ],
    'transforms': [
        'perspective_from_fov',
        'perspective_from_window',
        'intrinsics_from_fov',
        'intrinsics_from_focal_center',
        'fov_to_focal',
        'focal_to_fov',
        'intrinsics_to_fov',
        'view_look_at',
        'extrinsics_look_at',
        'perspective_to_intrinsics',
        'perspective_to_near_far',
        'intrinsics_to_perspective',
        'extrinsics_to_view',
        'view_to_extrinsics',
        'normalize_intrinsics',
        'crop_intrinsics',
        'pixel_to_uv',
        'pixel_to_ndc',
        'uv_to_pixel',
        'depth_linear_to_buffer',
        'depth_buffer_to_linear',
        'unproject_cv',
        'unproject_gl',
        'project_cv',
        'project_gl',
        'project',
        'unproject',
        'screen_coord_to_view_coord',
        'quaternion_to_matrix',
        'axis_angle_to_matrix',
        'matrix_to_quaternion',
        'extrinsics_to_essential',
        'euler_axis_angle_rotation',
        'euler_angles_to_matrix',
        'skew_symmetric',
        'rotation_matrix_from_vectors',
        'ray_intersection',
        'make_affine_matrix',
        'lerp',
        'slerp',
        'slerp_rotation_matrix',
        'interpolate_se3_matrix',
        'piecewise_lerp',
        'piecewise_interpolate_se3_matrix',
        'transform',
        'angle_between'
    ],
    'mesh':[
        'triangulate_mesh',
        'compute_face_corner_angles',
        'compute_face_corner_normals',
        'compute_face_corner_tangents',
        'compute_face_normals',
        'compute_face_tangents',
        'compute_vertex_normals',
        'remove_corrupted_faces',
        'merge_duplicate_vertices',
        'remove_unused_vertices',
        'subdivide_mesh',
        'get_mesh_edges',
        'flatten_mesh_indices',
        'create_cube_mesh',
        'create_icosahedron_mesh',
        'create_square_mesh',
        'create_camera_frustum_mesh',
        'merge_meshes',
        'calc_quad_candidates',
        'calc_quad_distortion',
        'calc_quad_direction',
        'calc_quad_smoothness',
        'solve_quad',
        'solve_quad_qp',
        'tri_to_quad'
    ],
    'maps': [
        'uv_map',
        'pixel_coord_map',
        'screen_coord_map',
        'build_mesh_from_map',
        'build_mesh_from_depth_map',
        'depth_map_edge',
        'depth_map_aliasing',
        'normal_map_edge',
        'point_map_to_normal_map',
        'depth_map_to_point_map',
        'depth_map_to_normal_map',
        'chessboard',
        'masked_nearest_resize',
        'masked_area_resize',
        'colorize_depth_map',
        'colorize_normal_map'
    ],
    'rasterization': [
        'RastContext',
        'rasterize_triangles',
        'rasterize_triangles_peeling',
        'rasterize_lines',
        'rasterize_point_cloud',
        'sample_texture',
        'test_rasterization',
    ],
}


__all__ = list(itertools.chain(*__modules_all__.values()))

def __getattr__(name):
    try:
        return globals()[name]
    except KeyError:
        pass

    try:
        module_name = next(m for m in __modules_all__ if name in __modules_all__[m])
    except StopIteration:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = importlib.import_module(f'.{module_name}', __name__)
    for key in __modules_all__[module_name]:
        globals()[key] = getattr(module, key)
        
    return globals()[name]


if TYPE_CHECKING:
    from .utils import *
    from .transforms import *
    from .mesh import *
    from .maps import *
    from .rasterization import *