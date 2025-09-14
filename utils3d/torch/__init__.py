import importlib
import itertools
import torch
from typing import TYPE_CHECKING

__modules_all__ = {
    'utils': [
        'sliding_window',
        'masked_min',
        'masked_max',
        'lookup',
        'segment_roll',
        'csr_matrix_from_dense_indices',
        'csr_eliminate_zeros',
        'split_groups_by_labels'
    ],
    'transforms': [
        'perspective_from_fov',
        'perspective_from_window', 
        'intrinsics_from_fov',
        'intrinsics_from_focal_center',
        'focal_to_fov',
        'fov_to_focal',
        'intrinsics_to_fov',
        'view_look_at',
        'extrinsics_look_at',
        'perspective_to_intrinsics',
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
        'project_gl',
        'project_cv',
        'unproject_gl',
        'unproject_cv',
        'project',
        'unproject',
        'skew_symmetric',
        'rotation_matrix_from_vectors',
        'euler_axis_angle_rotation',
        'euler_angles_to_matrix',
        'matrix_to_euler_angles',
        'matrix_to_quaternion',
        'quaternion_to_matrix',
        'matrix_to_axis_angle',
        'axis_angle_to_matrix',
        'axis_angle_to_quaternion',
        'quaternion_to_axis_angle',
        'make_affine_matrix',
        'lerp',
        'slerp',
        'slerp_rotation_matrix',
        'interpolate_se3_matrix',
        'extrinsics_to_essential',
        'rotation_matrix_2d',
        'rotate_2d',
        'translate_2d',
        'scale_2d',
        'transform',
        'angle_between'
    ],
    'mesh': [
        'triangulate_mesh',
        'compute_face_corner_angles',
        'compute_face_corner_normals',
        'compute_face_corner_tangents',
        'compute_face_normals',
        'compute_face_tangents',
        'mesh_edges',
        'mesh_half_edges',
        'mesh_dual_graph',
        'mesh_connected_components',
        'graph_connected_components',
        'compute_boundaries',
        'remove_unused_vertices',
        'remove_corrupted_faces',
        'remove_isolated_pieces',
        'merge_duplicate_vertices',
        'subdivide_mesh',
        'compute_mesh_laplacian',
        'laplacian_smooth_mesh',
        'taubin_smooth_mesh',
        'laplacian_hc_smooth_mesh',
    ],
    "maps": [
        'uv_map',
        'pixel_coord_map',
        'screen_coord_map',
        'build_mesh_from_map',
        'build_mesh_from_depth_map',
        'depth_map_edge',
        'depth_map_aliasing',
        'point_map_to_normal_map',
        'depth_map_to_point_map',
        'depth_map_to_normal_map',
        'chessboard',
        'bounding_rect_from_mask',
        'masked_nearest_resize',
        'masked_area_resize', 
    ],
    'rasterization': [
        'RastContext',
        'rasterize_triangles', 
        'rasterize_triangles_peeling',
        'sample_texture',
        'texture_composite',
        'warp_image_by_depth',
        'warp_image_by_forward_flow',
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
    from .transforms import *
    from .mesh import *
    from .utils import *
    from .maps import *
    from .rasterization import *