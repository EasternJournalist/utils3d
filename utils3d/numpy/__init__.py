"""
3D utility functions workings with NumPy.
"""
import importlib
import itertools
import numpy
from typing import TYPE_CHECKING


__modules_all__ = {
    'utils': [
        'sliding_window_1d',
        'sliding_window_nd',
        'sliding_window_2d',
        'max_pool_1d',
        'max_pool_2d',
        'max_pool_nd',
        'lookup',
    ],
    'transforms': [
        'perspective_from_fov',
        'perspective_from_window',
        'intrinsics_from_focal_center',
        'intrinsics_from_fov',
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
        'make_se3_matrix',
        'slerp_quaternion',
        'slerp_vector',
        'lerp',
        'lerp_se3_matrix',
        'piecewise_lerp',
        'piecewise_lerp_se3_matrix',
        'transform',
        'angle_between'
    ],
    'mesh':[
        'triangulate_mesh',
        'compute_face_normal',
        'compute_face_angle',
        'compute_vertex_normal',
        'compute_vertex_normal_weighted',
        'remove_corrupted_faces',
        'merge_duplicate_vertices',
        'remove_unused_vertices',
        'subdivide_mesh',
        'mesh_relations',
        'flatten_mesh_indices',
        'cube',
        'icosahedron',
        'square',
        'camera_frustum',
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
        'depth_map_edge',
        'normal_map_edge',
        'depth_map_aliasing',
        'screen_coord_map',
        'uv_map',
        'pixel_center_coord_map',
        'pixel_coord_map',
        'build_mesh_from_map',
        'build_mesh_from_depth_map',
        'point_map_to_normal_map',
        'depth_map_to_point_map',
        'depth_map_to_normal_map',
        'chessboard',
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