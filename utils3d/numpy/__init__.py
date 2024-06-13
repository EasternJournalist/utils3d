import importlib
import itertools
import numpy


__modules_all__ = {
    'mesh':[
        'triangulate',
        'compute_face_normal',
        'compute_face_angle',
        'compute_vertex_normal',
        'compute_vertex_normal_weighted',
        'remove_corrupted_faces',
        'merge_duplicate_vertices',
        'remove_unreferenced_vertices',
        'subdivide_mesh_simple',
        'mesh_relations'
    ],
    'quadmesh': [
        'calc_quad_candidates',
        'calc_quad_distortion',
        'calc_quad_direction',
        'calc_quad_smoothness',
        'sovle_quad',
        'sovle_quad_qp',
        'tri_to_quad'
    ],
    'utils': [
        'sliding_window_1d',
        'sliding_window_nd',
        'sliding_window_2d',
        'max_pool_1d',
        'max_pool_2d',
        'max_pool_nd',
        'depth_edge',
        'depth_aliasing',
        'interpolate',
        'image_scrcoord',
        'image_uv',
        'image_pixel_center',
        'image_mesh',
        'image_mesh_from_depth',
        'chessboard',
        'cube',
        'camera_frustum',
        'to4x4'
    ],
    'transforms': [
        'perspective',
        'perspective_from_fov',
        'perspective_from_fov_xy',
        'intrinsics_from_focal_center',
        'intrinsics_from_fov',
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
        'project_depth',
        'linearize_depth',
        'unproject_cv',
        'unproject_gl',
        'project_cv',
        'project_gl',
        'quaternion_to_matrix',
        'matrix_to_quaternion',
        'extrinsics_to_essential',
        'euler_axis_angle_rotation',
        'euler_angles_to_matrix',
        'skew_symmetric',
        'rotation_matrix_from_vectors',
        'ray_intersection',
    ],
    'rasterization': [
        'RastContext',
        'rasterize_vertex_attr',
        'texture',
        'warp_image_by_depth',
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


if __name__ == '__main__':
    from .quadmesh import *
    from .transforms import *
    from .mesh import *
    from .utils import *
    from .rasterization import *