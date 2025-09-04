# Auto-generated interface file
from typing import List, Tuple, Dict, Union, Optional, Any, overload, Literal, Callable
import numpy as numpy_
import torch as torch_
import nvdiffrast.torch
import numbers
from . import numpy, torch
import utils3d.numpy, utils3d.torch

__all__ = ["sliding_window_1d", 
"sliding_window_nd", 
"sliding_window_2d", 
"max_pool_1d", 
"max_pool_2d", 
"max_pool_nd", 
"lookup", 
"perspective_from_fov", 
"perspective_from_window", 
"intrinsics_from_focal_center", 
"intrinsics_from_fov", 
"fov_to_focal", 
"focal_to_fov", 
"intrinsics_to_fov", 
"view_look_at", 
"extrinsics_look_at", 
"perspective_to_intrinsics", 
"perspective_to_near_far", 
"intrinsics_to_perspective", 
"extrinsics_to_view", 
"view_to_extrinsics", 
"normalize_intrinsics", 
"crop_intrinsics", 
"pixel_to_uv", 
"pixel_to_ndc", 
"uv_to_pixel", 
"depth_linear_to_buffer", 
"depth_buffer_to_linear", 
"unproject_cv", 
"unproject_gl", 
"project_cv", 
"project_gl", 
"project", 
"unproject", 
"screen_coord_to_view_coord", 
"quaternion_to_matrix", 
"axis_angle_to_matrix", 
"matrix_to_quaternion", 
"extrinsics_to_essential", 
"euler_axis_angle_rotation", 
"euler_angles_to_matrix", 
"skew_symmetric", 
"rotation_matrix_from_vectors", 
"ray_intersection", 
"make_se3_matrix", 
"slerp_quaternion", 
"slerp_vector", 
"lerp", 
"lerp_se3_matrix", 
"piecewise_lerp", 
"piecewise_lerp_se3_matrix", 
"transform", 
"angle_between", 
"triangulate_mesh", 
"compute_face_normals", 
"compute_face_corner_angles", 
"compute_face_corner_normals", 
"compute_vertex_normals", 
"remove_corrupted_faces", 
"merge_duplicate_vertices", 
"remove_unused_vertices", 
"subdivide_mesh", 
"mesh_relations", 
"flatten_mesh_indices", 
"cube", 
"icosahedron", 
"square", 
"camera_frustum", 
"merge_meshes", 
"calc_quad_candidates", 
"calc_quad_distortion", 
"calc_quad_direction", 
"calc_quad_smoothness", 
"solve_quad", 
"solve_quad_qp", 
"tri_to_quad", 
"depth_map_edge", 
"normal_map_edge", 
"depth_map_aliasing", 
"screen_coord_map", 
"uv_map", 
"pixel_coord_map", 
"build_mesh_from_map", 
"build_mesh_from_depth_map", 
"point_map_to_normal_map", 
"depth_map_to_point_map", 
"depth_map_to_normal_map", 
"chessboard", 
"RastContext", 
"rasterize_triangles", 
"rasterize_triangles_peeling", 
"rasterize_lines", 
"rasterize_point_cloud", 
"sample_texture", 
"test_rasterization", 
"masked_min", 
"masked_max", 
"matrix_to_euler_angles", 
"matrix_to_axis_angle", 
"axis_angle_to_quaternion", 
"quaternion_to_axis_angle", 
"slerp", 
"interpolate_extrinsics", 
"interpolate_view", 
"rotation_matrix_2d", 
"rotate_2d", 
"translate_2d", 
"scale_2d", 
"compute_edges", 
"compute_connected_components", 
"compute_edge_connected_components", 
"compute_boundaries", 
"compute_dual_graph", 
"remove_isolated_pieces", 
"compute_face_tbn", 
"compute_vertex_tbn", 
"laplacian", 
"laplacian_smooth_mesh", 
"taubin_smooth_mesh", 
"laplacian_hc_smooth_mesh", 
"bounding_rect_from_mask", 
"texture_composite", 
"warp_image_by_depth", 
"warp_image_by_forward_flow"]

@overload
def sliding_window_1d(x: numpy_.ndarray, window_size: int, stride: int, axis: int = -1):
    """Return x view of the input array with x sliding window of the given kernel size and stride.
The sliding window is performed over the given axis, and the window dimension is append to the end of the output array's shape.

## Parameters
    x (np.ndarray): input array with shape (..., axis_size, ...)
    kernel_size (int): size of the sliding window
    stride (int): stride of the sliding window
    axis (int): axis to perform sliding window over

## Returns
    a_sliding (np.ndarray): view of the input array with shape (..., n_windows, ..., kernel_size), where n_windows = (axis_size - kernel_size + 1) // stride"""
    utils3d.numpy.utils.sliding_window_1d

@overload
def sliding_window_nd(x: numpy_.ndarray, window_size: Tuple[int, ...], stride: Tuple[int, ...], axis: Tuple[int, ...]) -> numpy_.ndarray:
    utils3d.numpy.utils.sliding_window_nd

@overload
def sliding_window_2d(x: numpy_.ndarray, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)) -> numpy_.ndarray:
    utils3d.numpy.utils.sliding_window_2d

@overload
def max_pool_1d(x: numpy_.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    utils3d.numpy.utils.max_pool_1d

@overload
def max_pool_2d(x: numpy_.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    utils3d.numpy.utils.max_pool_2d

@overload
def max_pool_nd(x: numpy_.ndarray, kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...], axis: Tuple[int, ...]) -> numpy_.ndarray:
    utils3d.numpy.utils.max_pool_nd

@overload
def lookup(key: numpy_.ndarray, query: numpy_.ndarray, value: Optional[numpy_.ndarray] = None, default_value: Optional[numpy_.ndarray] = None) -> numpy_.ndarray:
    """Look up `query` in `key` like a dictionary.

### Parameters
    `key` (np.ndarray): shape (num_keys, *query_key_shape), the array to search in
    `query` (np.ndarray): shape (num_queries, *query_key_shape), the array to search for
    `value` (Optional[np.ndarray]): shape (K, *value_shape), the array to get values from
    `default_value` (Optional[np.ndarray]): shape (*value_shape), default values to return if query is not found

### Returns
    If `value` is None, return the indices (num_queries,) of `query` in `key`, or -1. If a query is not found in key, the corresponding index will be -1.
    If `value` is provided, return the corresponding values (num_queries, *value_shape), or default_value if not found."""
    utils3d.numpy.utils.lookup

@overload
def perspective_from_fov(*, fov_x: Union[float, numpy_.ndarray, NoneType] = None, fov_y: Union[float, numpy_.ndarray, NoneType] = None, fov_min: Union[float, numpy_.ndarray, NoneType] = None, fov_max: Union[float, numpy_.ndarray, NoneType] = None, aspect_ratio: Union[float, numpy_.ndarray, NoneType] = None, near: Union[float, numpy_.ndarray, NoneType], far: Union[float, numpy_.ndarray, NoneType]) -> numpy_.ndarray:
    """Get OpenGL perspective matrix from field of view 

## Returns
    (ndarray): [..., 4, 4] perspective matrix"""
    utils3d.numpy.transforms.perspective_from_fov

@overload
def perspective_from_window(left: Union[float, numpy_.ndarray], right: Union[float, numpy_.ndarray], bottom: Union[float, numpy_.ndarray], top: Union[float, numpy_.ndarray], near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Get OpenGL perspective matrix from the window of z=-1 projection plane

## Returns
    (ndarray): [..., 4, 4] perspective matrix"""
    utils3d.numpy.transforms.perspective_from_window

@overload
def intrinsics_from_focal_center(fx: Union[float, numpy_.ndarray], fy: Union[float, numpy_.ndarray], cx: Union[float, numpy_.ndarray], cy: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Get OpenCV intrinsics matrix

## Returns
    (ndarray): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.numpy.transforms.intrinsics_from_focal_center

@overload
def intrinsics_from_fov(fov_x: Union[float, numpy_.ndarray, NoneType] = None, fov_y: Union[float, numpy_.ndarray, NoneType] = None, fov_max: Union[float, numpy_.ndarray, NoneType] = None, fov_min: Union[float, numpy_.ndarray, NoneType] = None, aspect_ratio: Union[float, numpy_.ndarray, NoneType] = None) -> numpy_.ndarray:
    """Get normalized OpenCV intrinsics matrix from given field of view.
You can provide either fov_x, fov_y, fov_max or fov_min and aspect_ratio

## Parameters
    fov_x (float | ndarray): field of view in x axis
    fov_y (float | ndarray): field of view in y axis
    fov_max (float | ndarray): field of view in largest dimension
    fov_min (float | ndarray): field of view in smallest dimension
    aspect_ratio (float | ndarray): aspect ratio of the image

## Returns
    (ndarray): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.numpy.transforms.intrinsics_from_fov

@overload
def fov_to_focal(fov: numpy_.ndarray):
    utils3d.numpy.transforms.fov_to_focal

@overload
def focal_to_fov(focal: numpy_.ndarray):
    utils3d.numpy.transforms.focal_to_fov

@overload
def intrinsics_to_fov(intrinsics: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    utils3d.numpy.transforms.intrinsics_to_fov

@overload
def view_look_at(eye: numpy_.ndarray, look_at: numpy_.ndarray, up: numpy_.ndarray) -> numpy_.ndarray:
    """Get OpenGL view matrix looking at something

## Parameters
    eye (ndarray): [..., 3] the eye position
    look_at (ndarray): [..., 3] the position to look at
    up (ndarray): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

## Returns
    (ndarray): [..., 4, 4], view matrix"""
    utils3d.numpy.transforms.view_look_at

@overload
def extrinsics_look_at(eye: numpy_.ndarray, look_at: numpy_.ndarray, up: numpy_.ndarray) -> numpy_.ndarray:
    """Get OpenCV extrinsics matrix looking at something

## Parameters
    eye (ndarray): [..., 3] the eye position
    look_at (ndarray): [..., 3] the position to look at
    up (ndarray): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

## Returns
    (ndarray): [..., 4, 4], extrinsics matrix"""
    utils3d.numpy.transforms.extrinsics_look_at

@overload
def perspective_to_intrinsics(perspective: numpy_.ndarray) -> numpy_.ndarray:
    """OpenGL perspective matrix to OpenCV intrinsics

## Parameters
    perspective (ndarray): [..., 4, 4] OpenGL perspective matrix

## Returns
    (ndarray): shape [..., 3, 3] OpenCV intrinsics"""
    utils3d.numpy.transforms.perspective_to_intrinsics

@overload
def perspective_to_near_far(perspective: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Get near and far planes from OpenGL perspective matrix

## Parameters"""
    utils3d.numpy.transforms.perspective_to_near_far

@overload
def intrinsics_to_perspective(intrinsics: numpy_.ndarray, near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """OpenCV intrinsics to OpenGL perspective matrix
NOTE: not work for tile-shifting intrinsics currently

## Parameters
    intrinsics (ndarray): [..., 3, 3] OpenCV intrinsics matrix
    near (float | ndarray): [...] near plane to clip
    far (float | ndarray): [...] far plane to clip
## Returns
    (ndarray): [..., 4, 4] OpenGL perspective matrix"""
    utils3d.numpy.transforms.intrinsics_to_perspective

@overload
def extrinsics_to_view(extrinsics: numpy_.ndarray) -> numpy_.ndarray:
    """OpenCV camera extrinsics to OpenGL view matrix

## Parameters
    extrinsics (ndarray): [..., 4, 4] OpenCV camera extrinsics matrix

## Returns
    (ndarray): [..., 4, 4] OpenGL view matrix"""
    utils3d.numpy.transforms.extrinsics_to_view

@overload
def view_to_extrinsics(view: numpy_.ndarray) -> numpy_.ndarray:
    """OpenGL view matrix to OpenCV camera extrinsics

## Parameters
    view (ndarray): [..., 4, 4] OpenGL view matrix

## Returns
    (ndarray): [..., 4, 4] OpenCV camera extrinsics matrix"""
    utils3d.numpy.transforms.view_to_extrinsics

@overload
def normalize_intrinsics(intrinsics: numpy_.ndarray, width: Union[numbers.Number, numpy_.ndarray], height: Union[numbers.Number, numpy_.ndarray], integer_pixel_centers: bool = True) -> numpy_.ndarray:
    """Normalize intrinsics from pixel cooridnates to uv coordinates

## Parameters
    intrinsics (ndarray): [..., 3, 3] camera intrinsics(s) to normalize
    width (int | ndarray): [...] image width(s)
    height (int | ndarray): [...] image height(s)
    integer_pixel_centers (bool): whether the integer pixel coordinates are at the center of the pixel. If False, the integer coordinates are at the left-top corner of the pixel.

## Returns
    (ndarray): [..., 3, 3] normalized camera intrinsics(s)"""
    utils3d.numpy.transforms.normalize_intrinsics

@overload
def crop_intrinsics(intrinsics: numpy_.ndarray, width: Union[numbers.Number, numpy_.ndarray], height: Union[numbers.Number, numpy_.ndarray], left: Union[numbers.Number, numpy_.ndarray], top: Union[numbers.Number, numpy_.ndarray], crop_width: Union[numbers.Number, numpy_.ndarray], crop_height: Union[numbers.Number, numpy_.ndarray]) -> numpy_.ndarray:
    """Evaluate the new intrinsics(s) after crop the image: cropped_img = img[top:top+crop_height, left:left+crop_width]

## Parameters
    intrinsics (ndarray): [..., 3, 3] camera intrinsics(s) to crop
    width (int | ndarray): [...] image width(s)
    height (int | ndarray): [...] image height(s)
    left (int | ndarray): [...] left crop boundary
    top (int | ndarray): [...] top crop boundary
    crop_width (int | ndarray): [...] crop width
    crop_height (int | ndarray): [...] crop height

## Returns
    (ndarray): [..., 3, 3] cropped camera intrinsics(s)"""
    utils3d.numpy.transforms.crop_intrinsics

@overload
def pixel_to_uv(pixel: numpy_.ndarray, width: Union[numbers.Number, numpy_.ndarray], height: Union[numbers.Number, numpy_.ndarray], pixel_definition: str = 'corner') -> numpy_.ndarray:
    """Convert pixel coordiantes to UV coordinates.

## Parameters
    pixel (ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
    width (Number | ndarray): [...] image width(s)
    height (Number | ndarray): [...] image height(s)

## Returns
    (ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)"""
    utils3d.numpy.transforms.pixel_to_uv

@overload
def pixel_to_ndc(pixel: numpy_.ndarray, width: Union[int, numpy_.ndarray], height: Union[int, numpy_.ndarray], pixel_definition: str = 'corner') -> numpy_.ndarray:
    """Convert pixel coordinates to NDC (Normalized Device Coordinates).

## Parameters
    pixel (ndarray): [..., 2] pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
    width (int | ndarray): [...] image width(s)
    height (int | ndarray): [...] image height(s)

## Returns
    (ndarray): [..., 2] pixel coordinrates defined in ndc space, the range is (-1, 1)"""
    utils3d.numpy.transforms.pixel_to_ndc

@overload
def uv_to_pixel(uv: numpy_.ndarray, width: Union[int, numpy_.ndarray], height: Union[int, numpy_.ndarray], pixel_definition: str = 'corner') -> numpy_.ndarray:
    """Convert UV coordinates to pixel coordinates.

## Parameters
    pixel (ndarray): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
    width (int | ndarray): [...] image width(s)
    height (int | ndarray): [...] image height(s)

## Returns
    (ndarray): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)"""
    utils3d.numpy.transforms.uv_to_pixel

@overload
def depth_linear_to_buffer(depth: numpy_.ndarray, near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """Project linear depth to depth value in screen space

## Parameters
    depth (ndarray): [...] depth value
    near (float | ndarray): [...] near plane to clip
    far (float | ndarray): [...] far plane to clip

## Returns
    (ndarray): [..., 1] depth value in screen space, value ranging in [0, 1]"""
    utils3d.numpy.transforms.depth_linear_to_buffer

@overload
def depth_buffer_to_linear(depth_buffer: numpy_.ndarray, near: Union[float, numpy_.ndarray], far: Union[float, numpy_.ndarray]) -> numpy_.ndarray:
    """OpenGL depth buffer to linear depth

## Parameters
    depth_buffer (ndarray): [...] depth value
    near (float | ndarray): [...] near plane to clip
    far (float | ndarray): [...] far plane to clip

## Returns
    (ndarray): [..., 1] linear depth"""
    utils3d.numpy.transforms.depth_buffer_to_linear

@overload
def unproject_cv(uv: numpy_.ndarray, depth: numpy_.ndarray, intrinsics: numpy_.ndarray, extrinsics: numpy_.ndarray = None) -> numpy_.ndarray:
    """Unproject uv coordinates to 3D view space following the OpenCV convention

## Parameters
    uv (ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    depth (ndarray): [..., N] depth value
    extrinsics (ndarray): [..., 4, 4] extrinsics matrix
    intrinsics (ndarray): [..., 3, 3] intrinsics matrix

## Returns
    points (ndarray): [..., N, 3] 3d points"""
    utils3d.numpy.transforms.unproject_cv

@overload
def unproject_gl(uv: numpy_.ndarray, depth: numpy_.ndarray, projection: numpy_.ndarray, view: Optional[numpy_.ndarray] = None) -> numpy_.ndarray:
    """Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices)

## Parameters
    uv (ndarray): (..., N, 2) screen space XY coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & bottom
    depth (ndarray): (..., N) linear depth values
    projection (ndarray): (..., 4, 4) projection  matrix
    view (ndarray): (..., 4, 4) view matrix
    
## Returns
    points (ndarray): (..., N, 3) 3d points"""
    utils3d.numpy.transforms.unproject_gl

@overload
def project_cv(points: numpy_.ndarray, intrinsics: numpy_.ndarray, extrinsics: Optional[numpy_.ndarray] = None) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Project 3D points to 2D following the OpenCV convention

## Parameters
    points (ndarray): [..., N, 3]
    extrinsics (ndarray): [..., 4, 4] extrinsics matrix
    intrinsics (ndarray): [..., 3, 3] intrinsics matrix

## Returns
    uv_coord (ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    linear_depth (ndarray): [..., N] linear depth"""
    utils3d.numpy.transforms.project_cv

@overload
def project_gl(points: numpy_.ndarray, projection: numpy_.ndarray, view: numpy_.ndarray = None) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrices)

## Parameters
    points (ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last 
        dimension is 4, the points are assumed to be in homogeneous coordinates
    view (ndarray): [..., 4, 4] view matrix
    projection (ndarray): [..., 4, 4] projection matrix

## Returns
    scr_coord (ndarray): [..., N, 2] OpenGL screen space XY coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & bottom
    linear_depth (ndarray): [..., N] linear depth"""
    utils3d.numpy.transforms.project_gl

@overload
def project(points: numpy_.ndarray, *, intrinsics: Optional[numpy_.ndarray] = None, extrinsics: Optional[numpy_.ndarray] = None, view: Optional[numpy_.ndarray] = None, projection: Optional[numpy_.ndarray] = None) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Calculate projection. 
- For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
- For OpenGL convention, use `view` and `projection` matrices.

## Parameters

- `points`: (..., N, 3) 3D world-space points
- `intrinsics`: (..., 3, 3) intrinsics matrix
- `extrinsics`: (..., 4, 4) extrinsics matrix
- `view`: (..., 4, 4) view matrix
- `projection`: (..., 4, 4) projection matrix

## Returns

- `uv`: (..., N, 2) 2D coordinates. 
    - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
    - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
- `depth`: (..., N) linear depth values, where `depth > 0` is visible.
    - For OpenCV convention, it is the Z coordinate in camera space.
    - For OpenGL convention, it is the -Z coordinate in camera space."""
    utils3d.numpy.transforms.project

@overload
def unproject(uv: numpy_.ndarray, depth: Optional[numpy_.ndarray], *, intrinsics: Optional[numpy_.ndarray] = None, extrinsics: Optional[numpy_.ndarray] = None, projection: Optional[numpy_.ndarray] = None, view: Optional[numpy_.ndarray] = None) -> numpy_.ndarray:
    """Calculate inverse projection. 
- For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
- For OpenGL convention, use `view` and `projection` matrices.

## Parameters

- `uv`: (..., N, 2) 2D coordinates. 
    - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
    - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
- `depth`: (..., N) linear depth values, where `depth > 0` is visible.
    - For OpenCV convention, it is the Z coordinate in camera space.
    - For OpenGL convention, it is the -Z coordinate in camera space.
- `intrinsics`: (..., 3, 3) intrinsics matrix
- `extrinsics`: (..., 4, 4) extrinsics matrix
- `view`: (..., 4, 4) view matrix
- `projection`: (..., 4, 4) projection matrix

## Returns

- `points`: (..., N, 3) 3D world-space points"""
    utils3d.numpy.transforms.unproject

@overload
def screen_coord_to_view_coord(screen_coord: numpy_.ndarray, projection: numpy_.ndarray) -> numpy_.ndarray:
    """Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices)

## Parameters
    screen_coord (ndarray): (..., N, 3) screen space XYZ coordinates, value ranging in [0, 1]
        The origin (0., 0.) is corresponding to the left & bottom
    projection (ndarray): (..., 4, 4) projection matrix

## Returns
    points (ndarray): [..., N, 3] 3d points in view space"""
    utils3d.numpy.transforms.screen_coord_to_view_coord

@overload
def quaternion_to_matrix(quaternion: numpy_.ndarray, eps: float = 1e-12) -> numpy_.ndarray:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices

## Parameters
    quaternion (ndarray): shape (..., 4), the quaternions to convert

## Returns
    ndarray: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions"""
    utils3d.numpy.transforms.quaternion_to_matrix

@overload
def axis_angle_to_matrix(axis_angle: numpy_.ndarray, eps: float = 1e-12) -> numpy_.ndarray:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

## Parameters
    axis_angle (ndarray): shape (..., 3), axis-angle vcetors

## Returns
    ndarray: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters"""
    utils3d.numpy.transforms.axis_angle_to_matrix

@overload
def matrix_to_quaternion(rot_mat: numpy_.ndarray, eps: float = 1e-12) -> numpy_.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

## Parameters
    rot_mat (ndarray): shape (..., 3, 3), the rotation matrices to convert

## Returns
    ndarray: shape (..., 4), the quaternions corresponding to the given rotation matrices"""
    utils3d.numpy.transforms.matrix_to_quaternion

@overload
def extrinsics_to_essential(extrinsics: numpy_.ndarray):
    """extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

## Parameters
    extrinsics (np.ndaray): [..., 4, 4] extrinsics matrix

## Returns
    (np.ndaray): [..., 3, 3] essential matrix"""
    utils3d.numpy.transforms.extrinsics_to_essential

@overload
def euler_axis_angle_rotation(axis: str, angle: numpy_.ndarray) -> numpy_.ndarray:
    """Return the rotation matrices for one of the rotations about an axis
of which Euler angles describe, for each value of the angle given.

## Parameters
    axis: Axis label "X" or "Y or "Z".
    angle: any shape tensor of Euler angles in radians

## Returns
    Rotation matrices as tensor of shape (..., 3, 3)."""
    utils3d.numpy.transforms.euler_axis_angle_rotation

@overload
def euler_angles_to_matrix(euler_angles: numpy_.ndarray, convention: str = 'XYZ') -> numpy_.ndarray:
    """Convert rotations given as Euler angles in radians to rotation matrices.

## Parameters
    euler_angles: Euler angles in radians as ndarray of shape (..., 3), XYZ
    convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

## Returns
    Rotation matrices as ndarray of shape (..., 3, 3)."""
    utils3d.numpy.transforms.euler_angles_to_matrix

@overload
def skew_symmetric(v: numpy_.ndarray):
    """Skew symmetric matrix from a 3D vector"""
    utils3d.numpy.transforms.skew_symmetric

@overload
def rotation_matrix_from_vectors(v1: numpy_.ndarray, v2: numpy_.ndarray):
    """Rotation matrix that rotates v1 to v2"""
    utils3d.numpy.transforms.rotation_matrix_from_vectors

@overload
def ray_intersection(p1: numpy_.ndarray, d1: numpy_.ndarray, p2: numpy_.ndarray, d2: numpy_.ndarray):
    """Compute the intersection/closest point of two D-dimensional rays
If the rays are intersecting, the closest point is the intersection point.

## Parameters
    p1 (ndarray): (..., D) origin of ray 1
    d1 (ndarray): (..., D) direction of ray 1
    p2 (ndarray): (..., D) origin of ray 2
    d2 (ndarray): (..., D) direction of ray 2

## Returns
    (ndarray): (..., N) intersection point"""
    utils3d.numpy.transforms.ray_intersection

@overload
def make_se3_matrix(R: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Convert rotation matrix and translation vector to 4x4 transformation matrix.

## Parameters
    R (ndarray): [..., 3, 3] rotation matrix
    t (ndarray): [..., 3] translation vector

## Returns
    ndarray: [..., 4, 4] transformation matrix"""
    utils3d.numpy.transforms.make_se3_matrix

@overload
def slerp_quaternion(q1: numpy_.ndarray, q2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Spherical linear interpolation between two unit quaternions.

## Parameters
    q1 (ndarray): [..., d] unit vector 1
    q2 (ndarray): [..., d] unit vector 2
    t (ndarray): [...] interpolation parameter in [0, 1]

## Returns
    ndarray: [..., 3] interpolated unit vector"""
    utils3d.numpy.transforms.slerp_quaternion

@overload
def slerp_vector(v1: numpy_.ndarray, v2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Spherical linear interpolation between two unit vectors. The vectors are assumed to be normalized.

## Parameters
    v1 (ndarray): [..., d] unit vector 1
    v2 (ndarray): [..., d] unit vector 2
    t (ndarray): [...] interpolation parameter in [0, 1]

## Returns
    ndarray: [..., d] interpolated unit vector"""
    utils3d.numpy.transforms.slerp_vector

@overload
def lerp(x1: numpy_.ndarray, x2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Linear interpolation between two vectors.

## Parameters
    x1 (ndarray): [..., d] vector 1
    x2 (ndarray): [..., d] vector 2
    t (ndarray): [...] interpolation parameter. [0, 1] for interpolation between x1 and x2, otherwise for extrapolation.

## Returns
    ndarray: [..., d] interpolated vector"""
    utils3d.numpy.transforms.lerp

@overload
def lerp_se3_matrix(T1: numpy_.ndarray, T2: numpy_.ndarray, t: numpy_.ndarray) -> numpy_.ndarray:
    """Linear interpolation between two SE(3) matrices.

## Parameters
    T1 (ndarray): [..., 4, 4] SE(3) matrix 1
    T2 (ndarray): [..., 4, 4] SE(3) matrix 2
    t (ndarray): [...] interpolation parameter in [0, 1]

## Returns
    ndarray: [..., 4, 4] interpolated SE(3) matrix"""
    utils3d.numpy.transforms.lerp_se3_matrix

@overload
def piecewise_lerp(x: numpy_.ndarray, t: numpy_.ndarray, s: numpy_.ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> numpy_.ndarray:
    """Linear spline interpolation.

## Parameters
- `x`: ndarray, shape (n, d): the values of data points.
- `t`: ndarray, shape (n,): the times of the data points.
- `s`: ndarray, shape (m,): the times to be interpolated.
- `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

## Returns
- `y`: ndarray, shape (..., m, d): the interpolated values."""
    utils3d.numpy.transforms.piecewise_lerp

@overload
def piecewise_lerp_se3_matrix(T: numpy_.ndarray, t: numpy_.ndarray, s: numpy_.ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> numpy_.ndarray:
    """Linear spline interpolation for SE(3) matrices.

## Parameters
- `T`: ndarray, shape (n, 4, 4): the SE(3) matrices.
- `t`: ndarray, shape (n,): the times of the data points.
- `s`: ndarray, shape (m,): the times to be interpolated.
- `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

## Returns
- `T_interp`: ndarray, shape (..., m, 4, 4): the interpolated SE(3) matrices."""
    utils3d.numpy.transforms.piecewise_lerp_se3_matrix

@overload
def transform(x: numpy_.ndarray, *Ts: numpy_.ndarray) -> numpy_.ndarray:
    """Apply affine transformation(s) to a point or a set of points.

## Parameters
- `x`: ndarray, shape (..., D): the point or a set of points to be transformed.
- `Ts`: ndarray, shape (..., D + 1, D + 1): the affine transformation matrix (matrices)
    If more than one transformation is given, they will be applied in corresponding order.
## Returns
- `y`: ndarray, shape (..., D): the transformed point or a set of points.

## Example Usage
```
y = transform(x, T1, T2, T3)
```"""
    utils3d.numpy.transforms.transform

@overload
def angle_between(v1: numpy_.ndarray, v2: numpy_.ndarray):
    """Calculate the angle between two vectors."""
    utils3d.numpy.transforms.angle_between

@overload
def triangulate_mesh(faces: numpy_.ndarray, vertices: numpy_.ndarray = None, method: Literal['fan', 'strip', 'diagonal'] = 'fan') -> numpy_.ndarray:
    """Triangulate a polygonal mesh.

## Parameters
    faces (np.ndarray): [L, P] polygonal faces
    vertices (np.ndarray, optional): [N, 3] 3-dimensional vertices.
        If given, the triangulation is performed according to the distance
        between vertices. Defaults to None.
    backslash (np.ndarray, optional): [L] boolean array indicating
        how to triangulate the quad faces. Defaults to None.

## Returns
    (np.ndarray): [L * (P - 2), 3] triangular faces"""
    utils3d.numpy.mesh.triangulate_mesh

@overload
def compute_face_normals(vertices: numpy_.ndarray, faces: numpy_.ndarray) -> numpy_.ndarray:
    """Compute face normals of a mesh

## Parameters
    vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, P] face indices

## Returns
    normals (np.ndarray): [..., T, 3] face normals"""
    utils3d.numpy.mesh.compute_face_normals

@overload
def compute_face_corner_angles(vertices: numpy_.ndarray, faces: numpy_.ndarray) -> numpy_.ndarray:
    """Compute face corner angles of a mesh

## Parameters
    vertices (np.ndarray): [..., N, 3] vertices
    faces (np.ndarray): [T, P] face vertex indices, where P is the number of vertices per face

## Returns
    angles (np.ndarray): [..., T, P] face corner angles"""
    utils3d.numpy.mesh.compute_face_corner_angles

@overload
def compute_face_corner_normals(vertices: numpy_.ndarray, faces: numpy_.ndarray, normalized: bool = True) -> numpy_.ndarray:
    """Compute the face corner normals of a mesh

## Parameters
    vertices (np.ndarray): [..., N, 3] vertices
    faces (np.ndarray): [T, P] face vertex indices, where P is the number of vertices per face

## Returns
    angles (np.ndarray): [..., T, P, 3] face corner normals"""
    utils3d.numpy.mesh.compute_face_corner_normals

@overload
def compute_vertex_normals(vertices: numpy_.ndarray, faces: numpy_.ndarray, weighted: Literal['uniform', 'area', 'angle'] = 'uniform') -> numpy_.ndarray:
    """Compute vertex normals of a triangular mesh by averaging neighboring face normals

## Parameters
    vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, P] face vertex indices, where P is the number of vertices per face

## Returns
    normals (np.ndarray): [..., N, 3] vertex normals (already normalized to unit vectors)"""
    utils3d.numpy.mesh.compute_vertex_normals

@overload
def remove_corrupted_faces(faces: numpy_.ndarray) -> numpy_.ndarray:
    """Remove corrupted faces (faces with duplicated vertices)

## Parameters
    faces (np.ndarray): [T, 3] triangular face indices

## Returns
    np.ndarray: [T_, 3] triangular face indices"""
    utils3d.numpy.mesh.remove_corrupted_faces

@overload
def merge_duplicate_vertices(vertices: numpy_.ndarray, faces: numpy_.ndarray, tol: float = 1e-06) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Merge duplicate vertices of a triangular mesh. 
Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

## Parameters
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices
    tol (float, optional): tolerance for merging. Defaults to 1e-6.

## Returns
    vertices (np.ndarray): [N_, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices"""
    utils3d.numpy.mesh.merge_duplicate_vertices

@overload
def remove_unused_vertices(faces: numpy_.ndarray, *vertice_attrs, return_indices: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Remove unreferenced vertices of a mesh. 
Unreferenced vertices are removed, and the face indices are updated accordingly.

## Parameters
    faces (np.ndarray): [T, P] face indices
    *vertice_attrs: vertex attributes

## Returns
    faces (np.ndarray): [T, P] face indices
    *vertice_attrs: vertex attributes
    indices (np.ndarray, optional): [N] indices of vertices that are kept. Defaults to None."""
    utils3d.numpy.mesh.remove_unused_vertices

@overload
def subdivide_mesh(vertices: numpy_.ndarray, faces: numpy_.ndarray, n: int = 1) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.

## Parameters
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices
    n (int, optional): number of subdivisions. Defaults to 1.

## Returns
    vertices (np.ndarray): [N_, 3] subdivided 3-dimensional vertices
    faces (np.ndarray): [4 * T, 3] subdivided triangular face indices"""
    utils3d.numpy.mesh.subdivide_mesh

@overload
def mesh_relations(faces: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Calculate the relation between vertices and faces.
NOTE: The input mesh must be a manifold triangle mesh.

## Parameters
    faces (np.ndarray): [T, 3] triangular face indices

## Returns
    edges (np.ndarray): [E, 2] edge indices
    edge2face (np.ndarray): [E, 2] edge to face relation. The second column is -1 if the edge is boundary.
    face2edge (np.ndarray): [T, 3] face to edge relation
    face2face (np.ndarray): [T, 3] face to face relation"""
    utils3d.numpy.mesh.mesh_relations

@overload
def flatten_mesh_indices(*args: numpy_.ndarray) -> Tuple[numpy_.ndarray, ...]:
    utils3d.numpy.mesh.flatten_mesh_indices

@overload
def cube(tri: bool = False) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Get x cube mesh of size 1 centered at origin.

### Parameters
    tri (bool, optional): return triangulated mesh. Defaults to False, which returns quad mesh.

### Returns
    vertices (np.ndarray): shape (8, 3) 
    faces (np.ndarray): shape (12, 3)"""
    utils3d.numpy.mesh.cube

@overload
def icosahedron():
    utils3d.numpy.mesh.icosahedron

@overload
def square(tri: bool = False) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Get a square mesh of area 1 centered at origin in the xy-plane.

## Returns
    vertices (np.ndarray): shape (4, 3)
    faces (np.ndarray): shape (1, 4)"""
    utils3d.numpy.mesh.square

@overload
def camera_frustum(extrinsics: numpy_.ndarray, intrinsics: numpy_.ndarray, depth: float = 1.0) -> Tuple[numpy_.ndarray, numpy_.ndarray, numpy_.ndarray]:
    """Get x triangle mesh of camera frustum."""
    utils3d.numpy.mesh.camera_frustum

@overload
def merge_meshes(meshes: List[Tuple[numpy_.ndarray, ...]]) -> Tuple[numpy_.ndarray, ...]:
    """Merge multiple meshes into one mesh. Vertices will be no longer shared.

## Parameters
    - `meshes`: a list of tuple (faces, vertices_attr1, vertices_attr2, ....)

## Returns
    - `faces`: [sum(T_i), P] merged face indices, contigous from 0 to sum(T_i) * P - 1
    - `*vertice_attrs`: [sum(T_i) * P, ...] merged vertex attributes, where every P values correspond to a face"""
    utils3d.numpy.mesh.merge_meshes

@overload
def calc_quad_candidates(edges: numpy_.ndarray, face2edge: numpy_.ndarray, edge2face: numpy_.ndarray):
    """Calculate the candidate quad faces.

## Parameters
    edges (np.ndarray): [E, 2] edge indices
    face2edge (np.ndarray): [T, 3] face to edge relation
    edge2face (np.ndarray): [E, 2] edge to face relation

## Returns
    quads (np.ndarray): [Q, 4] quad candidate indices
    quad2edge (np.ndarray): [Q, 4] edge to quad candidate relation
    quad2adj (np.ndarray): [Q, 8] adjacent quad candidates of each quad candidate
    quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid"""
    utils3d.numpy.mesh.calc_quad_candidates

@overload
def calc_quad_distortion(vertices: numpy_.ndarray, quads: numpy_.ndarray):
    """Calculate the distortion of each candidate quad face.

## Parameters
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    quads (np.ndarray): [Q, 4] quad face indices

## Returns
    distortion (np.ndarray): [Q] distortion of each quad face"""
    utils3d.numpy.mesh.calc_quad_distortion

@overload
def calc_quad_direction(vertices: numpy_.ndarray, quads: numpy_.ndarray):
    """Calculate the direction of each candidate quad face.

## Parameters
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    quads (np.ndarray): [Q, 4] quad face indices

## Returns
    direction (np.ndarray): [Q, 4] direction of each quad face.
        Represented by the angle between the crossing and each edge."""
    utils3d.numpy.mesh.calc_quad_direction

@overload
def calc_quad_smoothness(quad2edge: numpy_.ndarray, quad2adj: numpy_.ndarray, quads_direction: numpy_.ndarray):
    """Calculate the smoothness of each candidate quad face connection.

## Parameters
    quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
    quads_direction (np.ndarray): [Q, 4] direction of each quad face

## Returns
    smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection"""
    utils3d.numpy.mesh.calc_quad_smoothness

@overload
def solve_quad(face2edge: numpy_.ndarray, edge2face: numpy_.ndarray, quad2adj: numpy_.ndarray, quads_distortion: numpy_.ndarray, quads_smoothness: numpy_.ndarray, quads_valid: numpy_.ndarray):
    """Solve the quad mesh from the candidate quad faces.

## Parameters
    face2edge (np.ndarray): [T, 3] face to edge relation
    edge2face (np.ndarray): [E, 2] edge to face relation
    quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
    quads_distortion (np.ndarray): [Q] distortion of each quad face
    quads_smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
    quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid

## Returns
    weights (np.ndarray): [Q] weight of each valid quad face"""
    utils3d.numpy.mesh.solve_quad

@overload
def solve_quad_qp(face2edge: numpy_.ndarray, edge2face: numpy_.ndarray, quad2adj: numpy_.ndarray, quads_distortion: numpy_.ndarray, quads_smoothness: numpy_.ndarray, quads_valid: numpy_.ndarray):
    """Solve the quad mesh from the candidate quad faces.

## Parameters
    face2edge (np.ndarray): [T, 3] face to edge relation
    edge2face (np.ndarray): [E, 2] edge to face relation
    quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
    quads_distortion (np.ndarray): [Q] distortion of each quad face
    quads_smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
    quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid

## Returns
    weights (np.ndarray): [Q] weight of each valid quad face"""
    utils3d.numpy.mesh.solve_quad_qp

@overload
def tri_to_quad(vertices: numpy_.ndarray, faces: numpy_.ndarray) -> Tuple[numpy_.ndarray, numpy_.ndarray]:
    """Convert a triangle mesh to a quad mesh.
NOTE: The input mesh must be a manifold mesh.

## Parameters
    vertices (np.ndarray): [N, 3] 3-dimensional vertices
    faces (np.ndarray): [T, 3] triangular face indices

## Returns
    vertices (np.ndarray): [N_, 3] 3-dimensional vertices
    faces (np.ndarray): [Q, 4] quad face indices"""
    utils3d.numpy.mesh.tri_to_quad

@overload
def depth_map_edge(depth: numpy_.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.

## Parameters
    depth (np.ndarray): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

## Returns
    edge (np.ndarray): shape (..., height, width) of dtype torch.bool"""
    utils3d.numpy.maps.depth_map_edge

@overload
def normal_map_edge(normals: numpy_.ndarray, tol: float, kernel_size: int = 3, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute the edge mask from normal map.

## Parameters
    normal (np.ndarray): shape (..., height, width, 3), normal map
    tol (float): tolerance in degrees

## Returns
    edge (np.ndarray): shape (..., height, width) of dtype torch.bool"""
    utils3d.numpy.maps.normal_map_edge

@overload
def depth_map_aliasing(depth: numpy_.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: numpy_.ndarray = None) -> numpy_.ndarray:
    """Compute the map that indicates the aliasing of x depth map, identifying pixels which neither close to the maximum nor the minimum of its neighbors.
## Parameters
    depth (np.ndarray): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

## Returns
    edge (np.ndarray): shape (..., height, width) of dtype torch.bool"""
    utils3d.numpy.maps.depth_map_aliasing

@overload
def screen_coord_map(height: int, width: int, left: float = 0.0, top: float = 1.0, right: float = 1.0, bottom: float = 0.0, dtype: numpy_.dtype = numpy_.float32) -> numpy_.ndarray:
    """Get screen space coordinate map, where (0., 0.) is the bottom-left corner of the image, and (1., 1.) is the top-right corner of the image.
This is commonly used in graphics APIs like OpenGL.

## Parameters
    - `height`: `int` map height
    - `width`: `int` map width
    - `left`: `float`, optional left boundary in the screen space. Defaults to 0.
    - `top`: `float`, optional top boundary in the screen space. Defaults to 1.
    - `right`: `float`, optional right boundary in the screen space. Defaults to 1.
    - `bottom`: `float`, optional bottom boundary in the screen space. Defaults to 0.
    - `dtype`: `np.dtype`, optional data type of the output map. Defaults to np.float32.

## Returns
    (np.ndarray): shape (height, width, 2)"""
    utils3d.numpy.maps.screen_coord_map

@overload
def uv_map(height: int, width: int, left: float = 0.0, top: float = 0.0, right: float = 1.0, bottom: float = 1.0, dtype: numpy_.dtype = numpy_.float32) -> numpy_.ndarray:
    """Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.
This is commonly used as normalized image coordinates in texture mapping (when image is not flipped vertically).

## Parameters
    * `height`: `int` image height
    * `width`: `int` image width
    * `left`: `float`, optional left boundary in uv space. Defaults to 0.
    * `top`: `float`, optional top boundary in uv space. Defaults to 0.
    * `right`: `float`, optional right boundary in uv space. Defaults to 1.
    * `bottom`: `float`, optional bottom boundary in uv space. Defaults to 1.
    * `dtype`: `np.dtype`, optional data type of the output uv map. Defaults to np.float32.

## Returns
    - `uv (np.ndarray)`: shape `(height, width, 2)`

## Example Usage

>>> uv_map(10, 10):
[[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
 [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
  ...             ...                  ...
 [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]"""
    utils3d.numpy.maps.uv_map

@overload
def pixel_coord_map(height: int, width: int, left: int = 0, top: int = 0, definition: Literal['corner', 'center'] = 'corner', dtype: numpy_.dtype = numpy_.float32) -> numpy_.ndarray:
    """Get image pixel coordinates map, where (0, 0) is the top-left corner of the top-left pixel, and (width, height) is the bottom-right corner of the bottom-right pixel.

## Parameters
    - `height`: `int` image height
    - `width`: `int` image width
    - `left`: `int`, optional left boundary of the pixel coord map. Defaults to 0.
    - `top`: `int`, optional top boundary of the pixel coord map. Defaults to 0.
    - `definition`: `str`, optional 'corner' or 'center', whether the coordinates represent the corner or the center of the pixel. Defaults to 'corner'.
        - 'corner': coordinates range in [0, width - 1], [0, height - 1]
        - 'center': coordinates range in [0.5, width - 0.5], [0.5, height - 0.5]
    - `dtype`: `np.dtype`, optional data type of the output pixel coord map. Defaults to np.float32.

## Returns
    np.ndarray: shape (height, width, 2)

>>> pixel_coord_map(10, 10, definition='center', dtype=np.float32):
[[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
 [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
  ...             ...                  ...
[[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]

>>> pixel_coord_map(10, 10, definition='corner', dtype=np.int32):
[[[0, 0], [1, 0], ..., [9, 0]],
 [[0, 1], [1, 1], ..., [9, 1]],
    ...      ...         ...
 [[0, 9], [1, 9], ..., [9, 9]]]"""
    utils3d.numpy.maps.pixel_coord_map

@overload
def build_mesh_from_map(*maps: numpy_.ndarray, mask: Optional[numpy_.ndarray] = None, tri: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces.

## Parameters
    *maps (np.ndarray): attribute maps in shape (height, width, [channels])
    mask (np.ndarray, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

## Returns
    faces (np.ndarray): faces connecting neighboring pixels. shape (T, 4) if tri is False, else (T, 3)
    *attributes (np.ndarray): vertex attributes in corresponding order with input maps
    indices (np.ndarray, optional): indices of vertices in the original mesh"""
    utils3d.numpy.maps.build_mesh_from_map

@overload
def build_mesh_from_depth_map(depth: numpy_.ndarray, *other_maps: numpy_.ndarray, intrinsics: numpy_.ndarray, extrinsics: Optional[numpy_.ndarray] = None, atol: Optional[float] = None, rtol: Optional[float] = 0.05, tri: bool = False) -> Tuple[numpy_.ndarray, ...]:
    """Get a mesh by lifting depth map to 3D, while removing depths of large depth difference.

## Parameters
    depth (np.ndarray): [H, W] depth map
    extrinsics (np.ndarray, optional): [4, 4] extrinsics matrix. Defaults to None.
    intrinsics (np.ndarray, optional): [3, 3] intrinsics matrix. Defaults to None.
    *other_maps (np.ndarray): [H, W, C] vertex attributes. Defaults to None.
    atol (float, optional): absolute tolerance. Defaults to None.
    rtol (float, optional): relative tolerance. Defaults to None.
        triangles with vertices having depth difference larger than atol + rtol * depth will be marked.
    remove_by_depth (bool, optional): whether to remove triangles with large depth difference. Defaults to True.
    return_uv (bool, optional): whether to return uv coordinates. Defaults to False.
    return_indices (bool, optional): whether to return indices of vertices in the original mesh. Defaults to False.

## Returns
    faces (np.ndarray): [T, 3] faces
    vertices (np.ndarray): [N, 3] vertices
    *other_attrs (np.ndarray): [N, C] vertex attributes"""
    utils3d.numpy.maps.build_mesh_from_depth_map

@overload
def point_map_to_normal_map(point: numpy_.ndarray, mask: numpy_.ndarray = None, edge_threshold: float = None) -> numpy_.ndarray:
    """Calculate normal map from point map. Value range is [-1, 1]. 

## Parameters
    point (np.ndarray): shape (height, width, 3), point map
    mask (optional, np.ndarray): shape (height, width), dtype=bool. Mask of valid depth pixels. Defaults to None.
    edge_threshold (optional, float): threshold for the angle (in degrees) between the normal and the view direction. Defaults to None.

## Returns
    normal (np.ndarray): shape (height, width, 3), normal map. """
    utils3d.numpy.maps.point_map_to_normal_map

@overload
def depth_map_to_point_map(depth: numpy_.ndarray, intrinsics: numpy_.ndarray, extrinsics: numpy_.ndarray = None) -> numpy_.ndarray:
    """Unproject depth map to 3D points.

## Parameters
    depth (np.ndarray): [..., H, W] depth value
    intrinsics ( np.ndarray): [..., 3, 3] intrinsics matrix
    extrinsics (optional, np.ndarray): [..., 4, 4] extrinsics matrix

## Returns
    points (np.ndarray): [..., N, 3] 3d points"""
    utils3d.numpy.maps.depth_map_to_point_map

@overload
def depth_map_to_normal_map(depth: numpy_.ndarray, intrinsics: numpy_.ndarray, mask: numpy_.ndarray = None, edge_threshold: float = None) -> numpy_.ndarray:
    """Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenCV identity camera's coordinate system.

## Parameters
    depth (np.ndarray): shape (height, width), linear depth map
    intrinsics (np.ndarray): shape (3, 3), intrinsics matrix
    mask (optional, np.ndarray): shape (height, width), dtype=bool. Mask of valid depth pixels. Defaults to None.
    edge_threshold (optional, float): threshold for the angle (in degrees) between the normal and the view direction. Defaults to None.

## Returns
    normal (np.ndarray): shape (height, width, 3), normal map. """
    utils3d.numpy.maps.depth_map_to_normal_map

@overload
def chessboard(height: int, width: int, grid_size: int, color_a: numpy_.ndarray, color_b: numpy_.ndarray) -> numpy_.ndarray:
    """get x chessboard image

## Parameters
    height (int): image height
    width (int): image width
    grid_size (int): size of chessboard grid
    color_a (np.ndarray): color of the grid at the top-left corner
    color_b (np.ndarray): color in complementary grid cells

## Returns
    image (np.ndarray): shape (height, width, channels), chessboard image"""
    utils3d.numpy.maps.chessboard

@overload
def RastContext(*args, **kwargs):
    utils3d.numpy.rasterization.RastContext

@overload
def rasterize_triangles(ctx: utils3d.numpy.rasterization.RastContext, width: int, height: int, *, vertices: numpy_.ndarray, attributes: Optional[numpy_.ndarray] = None, attributes_domain: Optional[Literal['vertex', 'face']] = 'vertex', faces: Optional[numpy_.ndarray] = None, view: numpy_.ndarray = None, projection: numpy_.ndarray = None, cull_backface: bool = False, return_depth: bool = False, return_interpolation: bool = False, background_image: Optional[numpy_.ndarray] = None, background_depth: Optional[numpy_.ndarray] = None, background_interpolation_id: Optional[numpy_.ndarray] = None, background_interpolation_uv: Optional[numpy_.ndarray] = None) -> Dict[str, numpy_.ndarray]:
    """Rasterize triangles.

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
    - `interpolation_uv` (np.ndarray): (H, W, 2) float32 triangle UV (first two channels of barycentric coordinates)"""
    utils3d.numpy.rasterization.rasterize_triangles

@overload
def rasterize_triangles_peeling(ctx: utils3d.numpy.rasterization.RastContext, width: int, height: int, *, vertices: numpy_.ndarray, attributes: numpy_.ndarray, attributes_domain: Literal['vertex', 'face'] = 'vertex', faces: Optional[numpy_.ndarray] = None, view: numpy_.ndarray = None, projection: numpy_.ndarray = None, cull_backface: bool = False, return_depth: bool = False, return_interpolation: bool = False) -> Iterator[Iterator[Dict[str, numpy_.ndarray]]]:
    """Rasterize triangles with depth peeling.

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
```"""
    utils3d.numpy.rasterization.rasterize_triangles_peeling

@overload
def rasterize_lines(ctx: utils3d.numpy.rasterization.RastContext, width: int, height: int, *, vertices: numpy_.ndarray, lines: numpy_.ndarray, attributes: Optional[numpy_.ndarray], attributes_domain: Literal['vertex', 'line'] = 'vertex', view: Optional[numpy_.ndarray] = None, projection: Optional[numpy_.ndarray] = None, line_width: float = 1.0, return_depth: bool = False, return_interpolation: bool = False, background_image: Optional[numpy_.ndarray] = None, background_depth: Optional[numpy_.ndarray] = None, background_interpolation_id: Optional[numpy_.ndarray] = None, background_interpolation_uv: Optional[numpy_.ndarray] = None) -> Tuple[numpy_.ndarray, ...]:
    """Rasterize lines.

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
    - `interpolation_uv` (np.ndarray): (H, W, 2) float32 triangle UV (first two channels of barycentric coordinates)"""
    utils3d.numpy.rasterization.rasterize_lines

@overload
def rasterize_point_cloud(ctx: utils3d.numpy.rasterization.RastContext, width: int, height: int, *, points: numpy_.ndarray, point_sizes: Union[float, numpy_.ndarray] = 10, point_size_in: Literal['2d', '3d'] = '2d', point_shape: Literal['triangle', 'square', 'pentagon', 'hexagon', 'circle'] = 'square', attributes: Optional[numpy_.ndarray] = None, view: numpy_.ndarray = None, projection: numpy_.ndarray = None, return_depth: bool = False, return_point_id: bool = False, background_image: Optional[numpy_.ndarray] = None, background_depth: Optional[numpy_.ndarray] = None, background_point_id: Optional[numpy_.ndarray] = None) -> Dict[str, numpy_.ndarray]:
    """Rasterize point cloud.

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
    - `point_id` (np.ndarray): (H, W) int32 point ID map"""
    utils3d.numpy.rasterization.rasterize_point_cloud

@overload
def sample_texture(ctx: utils3d.numpy.rasterization.RastContext, uv_map: numpy_.ndarray, texture_map: numpy_.ndarray, interpolation: Literal['linear', 'nearest'] = 'linear', mipmap_level: Union[int, Tuple[int, int]] = 0, repeat: Union[bool, Tuple[bool, bool]] = False, anisotropic: float = 1.0) -> numpy_.ndarray:
    """Sample from a texture map with a UV map."""
    utils3d.numpy.rasterization.sample_texture

@overload
def test_rasterization(ctx: utils3d.numpy.rasterization.RastContext):
    """Test if rasterization works. It will render a cube with random colors and save it as a CHECKME.png file."""
    utils3d.numpy.rasterization.test_rasterization

@overload
def sliding_window_1d(x: torch_.Tensor, window_size: int, stride: int = 1, dim: int = -1) -> torch_.Tensor:
    """Sliding window view of the input tensor. The dimension of the sliding window is appended to the end of the input tensor's shape.
NOTE: Since Pytorch has `unfold` function, 1D sliding window view is just a wrapper of it."""
    utils3d.torch.utils.sliding_window_1d

@overload
def sliding_window_2d(x: torch_.Tensor, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], dim: Union[int, Tuple[int, int]] = (-2, -1)) -> torch_.Tensor:
    utils3d.torch.utils.sliding_window_2d

@overload
def sliding_window_nd(x: torch_.Tensor, window_size: Tuple[int, ...], stride: Tuple[int, ...], dim: Tuple[int, ...]) -> torch_.Tensor:
    utils3d.torch.utils.sliding_window_nd

@overload
def masked_min(input: torch_.Tensor, mask: torch_.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch_.Tensor, Tuple[torch_.Tensor, torch_.Tensor]]:
    """Similar to torch.min, but with mask
    """
    utils3d.torch.utils.masked_min

@overload
def masked_max(input: torch_.Tensor, mask: torch_.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch_.Tensor, Tuple[torch_.Tensor, torch_.Tensor]]:
    """Similar to torch.max, but with mask
    """
    utils3d.torch.utils.masked_max

@overload
def lookup(key: torch_.Tensor, query: torch_.Tensor) -> torch_.LongTensor:
    """Find the indices of `query` in `key`.

### Parameters
    key (torch.Tensor): shape (K, ...), the array to search in
    query (torch.Tensor): shape (Q, ...), the array to search for

### Returns
    torch.Tensor: shape (Q,), indices of `query` in `key`, or -1. If a query is not found in key, the corresponding index will be -1."""
    utils3d.torch.utils.lookup

@overload
def perspective_from_fov(*, fov_x: Union[float, torch_.Tensor, NoneType] = None, fov_y: Union[float, torch_.Tensor, NoneType] = None, fov_min: Union[float, torch_.Tensor, NoneType] = None, fov_max: Union[float, torch_.Tensor, NoneType] = None, aspect_ratio: Union[float, torch_.Tensor, NoneType] = None, near: Union[float, torch_.Tensor, NoneType], far: Union[float, torch_.Tensor, NoneType]) -> torch_.Tensor:
    """Get OpenGL perspective matrix from field of view 

## Returns
    (Tensor): [..., 4, 4] perspective matrix"""
    utils3d.torch.transforms.perspective_from_fov

@overload
def perspective_from_window(left: Union[float, torch_.Tensor], right: Union[float, torch_.Tensor], bottom: Union[float, torch_.Tensor], top: Union[float, torch_.Tensor], near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Get OpenGL perspective matrix from the window of z=-1 projection plane

## Returns
    (Tensor): [..., 4, 4] perspective matrix"""
    utils3d.torch.transforms.perspective_from_window

@overload
def intrinsics_from_fov(fov_x: Union[float, torch_.Tensor, NoneType] = None, fov_y: Union[float, torch_.Tensor, NoneType] = None, fov_max: Union[float, torch_.Tensor, NoneType] = None, fov_min: Union[float, torch_.Tensor, NoneType] = None, aspect_ratio: Union[float, torch_.Tensor, NoneType] = None) -> torch_.Tensor:
    """Get normalized OpenCV intrinsics matrix from given field of view.
You can provide either fov_x, fov_y, fov_max or fov_min and aspect_ratio

## Parameters
    fov_x (float | Tensor): field of view in x axis
    fov_y (float | Tensor): field of view in y axis
    fov_max (float | Tensor): field of view in largest dimension
    fov_min (float | Tensor): field of view in smallest dimension
    aspect_ratio (float | Tensor): aspect ratio of the image

## Returns
    (Tensor): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.torch.transforms.intrinsics_from_fov

@overload
def intrinsics_from_focal_center(fx: Union[float, torch_.Tensor], fy: Union[float, torch_.Tensor], cx: Union[float, torch_.Tensor], cy: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Get OpenCV intrinsics matrix

## Parameters
    focal_x (float | Tensor): focal length in x axis
    focal_y (float | Tensor): focal length in y axis
    cx (float | Tensor): principal point in x axis
    cy (float | Tensor): principal point in y axis

## Returns
    (Tensor): [..., 3, 3] OpenCV intrinsics matrix"""
    utils3d.torch.transforms.intrinsics_from_focal_center

@overload
def focal_to_fov(focal: torch_.Tensor):
    utils3d.torch.transforms.focal_to_fov

@overload
def fov_to_focal(fov: torch_.Tensor):
    utils3d.torch.transforms.fov_to_focal

@overload
def intrinsics_to_fov(intrinsics: torch_.Tensor) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """NOTE: approximate FOV by assuming centered principal point"""
    utils3d.torch.transforms.intrinsics_to_fov

@overload
def view_look_at(eye: torch_.Tensor, look_at: torch_.Tensor, up: torch_.Tensor) -> torch_.Tensor:
    """Get OpenGL view matrix looking at something

## Parameters
    eye (Tensor): [..., 3] the eye position
    look_at (Tensor): [..., 3] the position to look at
    up (Tensor): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

## Returns
    (Tensor): [..., 4, 4], view matrix"""
    utils3d.torch.transforms.view_look_at

@overload
def extrinsics_look_at(eye: torch_.Tensor, look_at: torch_.Tensor, up: torch_.Tensor) -> torch_.Tensor:
    """Get OpenCV extrinsics matrix looking at something

## Parameters
    eye (Tensor): [..., 3] the eye position
    look_at (Tensor): [..., 3] the position to look at
    up (Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

## Returns
    (Tensor): [..., 4, 4], extrinsics matrix"""
    utils3d.torch.transforms.extrinsics_look_at

@overload
def perspective_to_intrinsics(perspective: torch_.Tensor) -> torch_.Tensor:
    """OpenGL perspective matrix to OpenCV intrinsics

## Parameters
    perspective (Tensor): [..., 4, 4] OpenGL perspective matrix

## Returns
    (Tensor): shape [..., 3, 3] OpenCV intrinsics"""
    utils3d.torch.transforms.perspective_to_intrinsics

@overload
def intrinsics_to_perspective(intrinsics: torch_.Tensor, near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """OpenCV intrinsics to OpenGL perspective matrix
NOTE: not work for tile-shifting intrinsics currently

## Parameters
    intrinsics (Tensor): [..., 3, 3] OpenCV intrinsics matrix
    near (float | Tensor): [...] near plane to clip
    far (float | Tensor): [...] far plane to clip
## Returns
    (Tensor): [..., 4, 4] OpenGL perspective matrix"""
    utils3d.torch.transforms.intrinsics_to_perspective

@overload
def extrinsics_to_view(extrinsics: torch_.Tensor) -> torch_.Tensor:
    """OpenCV camera extrinsics to OpenGL view matrix

## Parameters
    extrinsics (Tensor): [..., 4, 4] OpenCV camera extrinsics matrix

## Returns
    (Tensor): [..., 4, 4] OpenGL view matrix"""
    utils3d.torch.transforms.extrinsics_to_view

@overload
def view_to_extrinsics(view: torch_.Tensor) -> torch_.Tensor:
    """OpenGL view matrix to OpenCV camera extrinsics

## Parameters
    view (Tensor): [..., 4, 4] OpenGL view matrix

## Returns
    (Tensor): [..., 4, 4] OpenCV camera extrinsics matrix"""
    utils3d.torch.transforms.view_to_extrinsics

@overload
def normalize_intrinsics(intrinsics: torch_.Tensor, width: Union[numbers.Number, torch_.Tensor], height: Union[numbers.Number, torch_.Tensor]) -> torch_.Tensor:
    """Normalize camera intrinsics(s) to uv space

## Parameters
    intrinsics (Tensor): [..., 3, 3] camera intrinsics(s) to normalize
    width (int | Tensor): [...] image width(s)
    height (int | Tensor): [...] image height(s)

## Returns
    (Tensor): [..., 3, 3] normalized camera intrinsics(s)"""
    utils3d.torch.transforms.normalize_intrinsics

@overload
def crop_intrinsics(intrinsics: torch_.Tensor, original_height: Union[numbers.Number, torch_.Tensor], original_width: Union[numbers.Number, torch_.Tensor], cropped_left: Union[numbers.Number, torch_.Tensor], cropped_top: Union[numbers.Number, torch_.Tensor], cropped_height: Union[numbers.Number, torch_.Tensor], cropped_width: Union[numbers.Number, torch_.Tensor]) -> torch_.Tensor:
    """Evaluate the new intrinsics(s) after crop the image: cropped_img = img[top:top+crop_height, left:left+crop_width]

## Parameters
    intrinsics (Tensor): [..., 3, 3] camera intrinsics(s) to crop
    original_height (int | Tensor): [...] original image height(s)
    original_width (int | Tensor): [...] original image width(s)
    cropped_left (int | Tensor): [...] left pixel index of the cropped image(s)
    cropped_top (int | Tensor): [...] top pixel index of the cropped image(s)
    cropped_height (int | Tensor): [...] cropped image height(s)
    cropped_width (int | Tensor): [...] cropped image width(s)

## Returns
    (Tensor): [..., 3, 3] cropped camera intrinsics(s)"""
    utils3d.torch.transforms.crop_intrinsics

@overload
def pixel_to_uv(pixel: torch_.Tensor, width: Union[numbers.Number, torch_.Tensor], height: Union[numbers.Number, torch_.Tensor], pixel_definition: Literal['corner', 'center'] = 'corner') -> torch_.Tensor:
    """## Parameters
    pixel (Tensor): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
    width (int | Tensor): [...] image width(s)
    height (int | Tensor): [...] image height(s)

## Returns
    (Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)"""
    utils3d.torch.transforms.pixel_to_uv

@overload
def pixel_to_ndc(pixel: torch_.Tensor, width: Union[int, torch_.Tensor], height: Union[int, torch_.Tensor], pixel_definition: Literal['corner', 'center'] = 'corner') -> torch_.Tensor:
    """## Parameters
    pixel (Tensor): [..., 2] pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
    width (int | Tensor): [...] image width(s)
    height (int | Tensor): [...] image height(s)

## Returns
    (Tensor): [..., 2] pixel coordinrates defined in ndc space, the range is (-1, 1)"""
    utils3d.torch.transforms.pixel_to_ndc

@overload
def uv_to_pixel(uv: torch_.Tensor, width: Union[int, torch_.Tensor], height: Union[int, torch_.Tensor], pixel_definition: Literal['corner', 'center'] = 'corner') -> torch_.Tensor:
    """## Parameters
    uv (Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    width (int | Tensor): [...] image width(s)
    height (int | Tensor): [...] image height(s)

## Returns
    (Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)"""
    utils3d.torch.transforms.uv_to_pixel

@overload
def depth_linear_to_buffer(depth: torch_.Tensor, near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Project linear depth to depth value in screen space

## Parameters
    depth (Tensor): [...] depth value
    near (float | Tensor): [...] near plane to clip
    far (float | Tensor): [...] far plane to clip

## Returns
    (Tensor): [..., 1] depth value in screen space, value ranging in [0, 1]"""
    utils3d.torch.transforms.depth_linear_to_buffer

@overload
def depth_buffer_to_linear(depth: torch_.Tensor, near: Union[float, torch_.Tensor], far: Union[float, torch_.Tensor]) -> torch_.Tensor:
    """Linearize depth value to linear depth

## Parameters
    depth (Tensor): [...] screen depth value, ranging in [0, 1]
    near (float | Tensor): [...] near plane to clip
    far (float | Tensor): [...] far plane to clip

## Returns
    (Tensor): [...] linear depth"""
    utils3d.torch.transforms.depth_buffer_to_linear

@overload
def project_gl(points: torch_.Tensor, projection: torch_.Tensor, view: torch_.Tensor = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrices)

## Parameters
    points (Tensor): [..., N, 3] or [..., N, 4] 3D points to project, if the last 
        dimension is 4, the points are assumed to be in homogeneous coordinates
    view (Tensor): [..., 4, 4] view matrix
    projection (Tensor): [..., 4, 4] projection matrix

## Returns
    scr_coord (Tensor): [..., N, 3] screen space coordinates, value ranging in [0, 1].
        The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
    linear_depth (Tensor): [..., N] linear depth"""
    utils3d.torch.transforms.project_gl

@overload
def project_cv(points: torch_.Tensor, intrinsics: torch_.Tensor, extrinsics: Optional[torch_.Tensor] = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Project 3D points to 2D following the OpenCV convention

## Parameters
    points (Tensor): [..., N, 3] 3D points
    intrinsics (Tensor): [..., 3, 3] intrinsics matrix
    extrinsics (Tensor): [..., 4, 4] extrinsics matrix

## Returns
    uv_coord (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    linear_depth (Tensor): [..., N] linear depth"""
    utils3d.torch.transforms.project_cv

@overload
def unproject_gl(uv: torch_.Tensor, depth: torch_.Tensor, projection: torch_.Tensor, view: Optional[torch_.Tensor] = None) -> torch_.Tensor:
    """Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices)

## Parameters
    uv (Tensor): (..., N, 2) screen space XY coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & bottom
    depth (Tensor): (..., N) linear depth values
    projection (Tensor): (..., 4, 4) projection  matrix
    view (Tensor): (..., 4, 4) view matrix
    
## Returns
    points (Tensor): (..., N, 3) 3d points"""
    utils3d.torch.transforms.unproject_gl

@overload
def unproject_cv(uv: torch_.Tensor, depth: torch_.Tensor, intrinsics: torch_.Tensor, extrinsics: torch_.Tensor = None) -> torch_.Tensor:
    """Unproject uv coordinates to 3D view space following the OpenCV convention

## Parameters
    uv (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
        The origin (0., 0.) is corresponding to the left & top
    depth (Tensor): [..., N] depth value
    extrinsics (Tensor): [..., 4, 4] extrinsics matrix
    intrinsics (Tensor): [..., 3, 3] intrinsics matrix

## Returns
    points (Tensor): [..., N, 3] 3d points"""
    utils3d.torch.transforms.unproject_cv

@overload
def project(points: torch_.Tensor, *, intrinsics: Optional[torch_.Tensor] = None, extrinsics: Optional[torch_.Tensor] = None, view: Optional[torch_.Tensor] = None, projection: Optional[torch_.Tensor] = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Calculate projection. 
- For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
- For OpenGL convention, use `view` and `projection` matrices.

## Parameters

- `points`: (..., N, 3) 3D world-space points
- `intrinsics`: (..., 3, 3) intrinsics matrix
- `extrinsics`: (..., 4, 4) extrinsics matrix
- `view`: (..., 4, 4) view matrix
- `projection`: (..., 4, 4) projection matrix

## Returns

- `uv`: (..., N, 2) 2D coordinates. 
    - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
    - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
- `depth`: (..., N) linear depth values, where `depth > 0` is visible.
    - For OpenCV convention, it is the Z coordinate in camera space.
    - For OpenGL convention, it is the -Z coordinate in camera space."""
    utils3d.torch.transforms.project

@overload
def unproject(uv: torch_.Tensor, depth: Optional[torch_.Tensor], *, intrinsics: Optional[torch_.Tensor] = None, extrinsics: Optional[torch_.Tensor] = None, projection: Optional[torch_.Tensor] = None, view: Optional[torch_.Tensor] = None) -> torch_.Tensor:
    """Calculate inverse projection. 
- For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
- For OpenGL convention, use `view` and `projection` matrices.

## Parameters

- `uv`: (..., N, 2) 2D coordinates. 
    - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
    - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
- `depth`: (..., N) linear depth values, where `depth > 0` is visible.
    - For OpenCV convention, it is the Z coordinate in camera space.
    - For OpenGL convention, it is the -Z coordinate in camera space.
- `intrinsics`: (..., 3, 3) intrinsics matrix
- `extrinsics`: (..., 4, 4) extrinsics matrix
- `view`: (..., 4, 4) view matrix
- `projection`: (..., 4, 4) projection matrix

## Returns

- `points`: (..., N, 3) 3D world-space points"""
    utils3d.torch.transforms.unproject

@overload
def skew_symmetric(v: torch_.Tensor):
    """Skew symmetric matrix from a 3D vector"""
    utils3d.torch.transforms.skew_symmetric

@overload
def rotation_matrix_from_vectors(v1: torch_.Tensor, v2: torch_.Tensor):
    """Rotation matrix that rotates v1 to v2"""
    utils3d.torch.transforms.rotation_matrix_from_vectors

@overload
def euler_axis_angle_rotation(axis: str, angle: torch_.Tensor) -> torch_.Tensor:
    """Return the rotation matrices for one of the rotations about an axis
of which Euler angles describe, for each value of the angle given.

## Parameters
    axis: Axis label "X" or "Y or "Z".
    angle: any shape tensor of Euler angles in radians

## Returns
    Rotation matrices as tensor of shape (..., 3, 3)."""
    utils3d.torch.transforms.euler_axis_angle_rotation

@overload
def euler_angles_to_matrix(euler_angles: torch_.Tensor, convention: str = 'XYZ') -> torch_.Tensor:
    """Convert rotations given as Euler angles in radians to rotation matrices.

## Parameters
    euler_angles: Euler angles in radians as tensor of shape (..., 3), XYZ
    convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

## Returns
    Rotation matrices as tensor of shape (..., 3, 3)."""
    utils3d.torch.transforms.euler_angles_to_matrix

@overload
def matrix_to_euler_angles(matrix: torch_.Tensor, convention: str) -> torch_.Tensor:
    """Convert rotations given as rotation matrices to Euler angles in radians.
NOTE: The composition order eg. `XYZ` means `Rz * Ry * Rx` (like blender), instead of `Rx * Ry * Rz` (like pytorch3d)

## Parameters
    matrix: Rotation matrices as tensor of shape (..., 3, 3).
    convention: Convention string of three uppercase letters.

## Returns
    Euler angles in radians as tensor of shape (..., 3), in the order of XYZ (like blender), instead of convention (like pytorch3d)"""
    utils3d.torch.transforms.matrix_to_euler_angles

@overload
def matrix_to_quaternion(rot_mat: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

## Parameters
    rot_mat (Tensor): shape (..., 3, 3), the rotation matrices to convert

## Returns
    Tensor: shape (..., 4), the quaternions corresponding to the given rotation matrices"""
    utils3d.torch.transforms.matrix_to_quaternion

@overload
def quaternion_to_matrix(quaternion: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices

## Parameters
    quaternion (Tensor): shape (..., 4), the quaternions to convert

## Returns
    Tensor: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions"""
    utils3d.torch.transforms.quaternion_to_matrix

@overload
def matrix_to_axis_angle(rot_mat: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert a batch of 3x3 rotation matrices to axis-angle representation (rotation vector)

## Parameters
    rot_mat (Tensor): shape (..., 3, 3), the rotation matrices to convert

## Returns
    Tensor: shape (..., 3), the axis-angle vectors corresponding to the given rotation matrices"""
    utils3d.torch.transforms.matrix_to_axis_angle

@overload
def axis_angle_to_matrix(axis_angle: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

## Parameters
    axis_angle (Tensor): shape (..., 3), axis-angle vcetors

## Returns
    Tensor: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters"""
    utils3d.torch.transforms.axis_angle_to_matrix

@overload
def axis_angle_to_quaternion(axis_angle: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert axis-angle representation (rotation vector) to quaternion (w, x, y, z)

## Parameters
    axis_angle (Tensor): shape (..., 3), axis-angle vcetors

## Returns
    Tensor: shape (..., 4) The quaternions for the given axis-angle parameters"""
    utils3d.torch.transforms.axis_angle_to_quaternion

@overload
def quaternion_to_axis_angle(quaternion: torch_.Tensor, eps: float = 1e-12) -> torch_.Tensor:
    """Convert a batch of quaternions (w, x, y, z) to axis-angle representation (rotation vector)

## Parameters
    quaternion (Tensor): shape (..., 4), the quaternions to convert

## Returns
    Tensor: shape (..., 3), the axis-angle vectors corresponding to the given quaternions"""
    utils3d.torch.transforms.quaternion_to_axis_angle

@overload
def slerp(rot_mat_1: torch_.Tensor, rot_mat_2: torch_.Tensor, t: Union[numbers.Number, torch_.Tensor]) -> torch_.Tensor:
    """Spherical linear interpolation between two rotation matrices

## Parameters
    rot_mat_1 (Tensor): shape (..., 3, 3), the first rotation matrix
    rot_mat_2 (Tensor): shape (..., 3, 3), the second rotation matrix
    t (Tensor): scalar or shape (...,), the interpolation factor

## Returns
    Tensor: shape (..., 3, 3), the interpolated rotation matrix"""
    utils3d.torch.transforms.slerp

@overload
def interpolate_extrinsics(ext1: torch_.Tensor, ext2: torch_.Tensor, t: Union[numbers.Number, torch_.Tensor]) -> torch_.Tensor:
    """Interpolate extrinsics between two camera poses. Linear interpolation for translation, spherical linear interpolation for rotation.

## Parameters
    ext1 (Tensor): shape (..., 4, 4), the first camera pose
    ext2 (Tensor): shape (..., 4, 4), the second camera pose
    t (Tensor): scalar or shape (...,), the interpolation factor

## Returns
    Tensor: shape (..., 4, 4), the interpolated camera pose"""
    utils3d.torch.transforms.interpolate_extrinsics

@overload
def interpolate_view(view1: torch_.Tensor, view2: torch_.Tensor, t: Union[numbers.Number, torch_.Tensor]):
    """Interpolate view matrices between two camera poses. Linear interpolation for translation, spherical linear interpolation for rotation.

## Parameters
    ext1 (Tensor): shape (..., 4, 4), the first camera pose
    ext2 (Tensor): shape (..., 4, 4), the second camera pose
    t (Tensor): scalar or shape (...,), the interpolation factor

## Returns
    Tensor: shape (..., 4, 4), the interpolated camera pose"""
    utils3d.torch.transforms.interpolate_view

@overload
def extrinsics_to_essential(extrinsics: torch_.Tensor):
    """extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

## Parameters
    extrinsics (Tensor): [..., 4, 4] extrinsics matrix

## Returns
    (Tensor): [..., 3, 3] essential matrix"""
    utils3d.torch.transforms.extrinsics_to_essential

@overload
def make_se3_matrix(R: torch_.Tensor, t: torch_.Tensor):
    """Compose rotation matrix and translation vector to 4x4 transformation matrix

## Parameters
    R (Tensor): [..., 3, 3] rotation matrix
    t (Tensor): [..., 3] translation vector

## Returns
    (Tensor): [..., 4, 4] transformation matrix"""
    utils3d.torch.transforms.make_se3_matrix

@overload
def rotation_matrix_2d(theta: Union[float, torch_.Tensor]):
    """2x2 matrix for 2D rotation

## Parameters
    theta (float | Tensor): rotation angle in radians, arbitrary shape (...,)

## Returns
    (Tensor): (..., 2, 2) rotation matrix"""
    utils3d.torch.transforms.rotation_matrix_2d

@overload
def rotate_2d(theta: Union[float, torch_.Tensor], center: torch_.Tensor = None):
    """3x3 matrix for 2D rotation around a center
```
   [[Rxx, Rxy, tx],
    [Ryx, Ryy, ty],
    [0,     0,  1]]
```
## Parameters
    theta (float | Tensor): rotation angle in radians, arbitrary shape (...,)
    center (Tensor): rotation center, arbitrary shape (..., 2). Default to (0, 0)
    
## Returns
    (Tensor): (..., 3, 3) transformation matrix"""
    utils3d.torch.transforms.rotate_2d

@overload
def translate_2d(translation: torch_.Tensor):
    """Translation matrix for 2D translation
```
   [[1, 0, tx],
    [0, 1, ty],
    [0, 0,  1]]
```
## Parameters
    translation (Tensor): translation vector, arbitrary shape (..., 2)

## Returns
    (Tensor): (..., 3, 3) transformation matrix"""
    utils3d.torch.transforms.translate_2d

@overload
def scale_2d(scale: Union[float, torch_.Tensor], center: torch_.Tensor = None):
    """Scale matrix for 2D scaling
```
   [[s, 0, tx],
    [0, s, ty],
    [0, 0,  1]]
```
## Parameters
    scale (float | Tensor): scale factor, arbitrary shape (...,)
    center (Tensor): scale center, arbitrary shape (..., 2). Default to (0, 0)

## Returns
    (Tensor): (..., 3, 3) transformation matrix"""
    utils3d.torch.transforms.scale_2d

@overload
def transform(x: torch_.Tensor, *Ts: torch_.Tensor) -> torch_.Tensor:
    """Apply affine transformation(s) to a point or a set of points.

## Parameters
- `x`: Tensor, shape (..., D): the point or a set of points to be transformed.
- `Ts`: Tensor, shape (..., D + 1, D + 1): the affine transformation matrix (matrices)
    If more than one transformation is given, they will be applied in corresponding order.
## Returns
- `y`: Tensor, shape (..., D): the transformed point or a set of points.

## Example Usage
```
y = transform(x, T1, T2, T3)
```"""
    utils3d.torch.transforms.transform

@overload
def angle_between(v1: torch_.Tensor, v2: torch_.Tensor, eps: float = 1e-08) -> torch_.Tensor:
    """Calculate the angle between two vectors.

NOTE: `eps` prevents zero angle difference which is indifferentiable."""
    utils3d.torch.transforms.angle_between

@overload
def triangulate_mesh(faces: torch_.Tensor, vertices: torch_.Tensor = None, method: Literal['fan', 'strip', 'diagonal'] = 'fan') -> torch_.Tensor:
    """Triangulate a polygonal mesh.

## Parameters
    faces (Tensor): [L, P] polygonal faces
    vertices (Tensor, optional): [N, 3] 3-dimensional vertices.
        If given, the triangulation is performed according to the distance
        between vertices. Defaults to None.
    backslash (Tensor, optional): [L] boolean array indicating
        how to triangulate the quad faces. Defaults to None.

## Returns
    (Tensor): [L * (P - 2), 3] triangular faces"""
    utils3d.torch.mesh.triangulate_mesh

@overload
def compute_face_normals(vertices: torch_.Tensor, faces: torch_.Tensor) -> torch_.Tensor:
    """Compute face normals of a polygon mesh

## Parameters
    vertices (Tensor): [..., N, 3] 3-dimensional vertices
    faces (Tensor): [T, P] face indices

## Returns
    normals (Tensor): [..., T, 3] face normals"""
    utils3d.torch.mesh.compute_face_normals

@overload
def compute_face_corner_normals(vertices: torch_.Tensor, faces: torch_.Tensor, normalized: bool = True) -> torch_.Tensor:
    """Compute the face corner normals of a polygon mesh

## Parameters
    vertices (Tensor): [..., N, 3] vertices
    faces (Tensor): [T, P] face vertex indices, where P is the number of vertices per face

## Returns
    angles (Tensor): [..., T, P, 3] face corner normals"""
    utils3d.torch.mesh.compute_face_corner_normals

@overload
def compute_face_corner_angles(vertices: torch_.Tensor, faces: torch_.Tensor) -> torch_.Tensor:
    """Compute face corner angles of a polygon mesh

## Parameters
    vertices (Tensor): [..., N, 3] vertices
    faces (Tensor): [T, P] face vertex indices, where P is the number of vertices per face

## Returns
    angles (Tensor): [..., T, P] face corner angles"""
    utils3d.torch.mesh.compute_face_corner_angles

@overload
def compute_vertex_normals(vertices: torch_.Tensor, faces: torch_.Tensor, weighted: Literal['uniform', 'area', 'angle'] = 'uniform') -> torch_.Tensor:
    """Compute vertex normals of a polygon mesh by averaging neighboring face normals

## Parameters
    vertices (Tensor): [..., N, 3] 3-dimensional vertices
    faces (Tensor): [T, P] face vertex indices, where P is the number of vertices per face

## Returns
    normals (Tensor): [..., N, 3] vertex normals (already normalized to unit vectors)"""
    utils3d.torch.mesh.compute_vertex_normals

@overload
def compute_edges(faces: torch_.Tensor) -> Tuple[torch_.Tensor, torch_.Tensor, torch_.Tensor]:
    """Compute edges of a mesh.

## Parameters
    faces (Tensor): [T, 3] triangular face indices
    
## Returns
    edges (Tensor): [E, 2] edge indices
    face2edge (Tensor): [T, 3] mapping from face to edge
    counts (Tensor): [E] degree of each edge"""
    utils3d.torch.mesh.compute_edges

@overload
def compute_connected_components(faces: torch_.Tensor, edges: torch_.Tensor = None, face2edge: torch_.Tensor = None) -> List[torch_.Tensor]:
    """Compute connected faces of a mesh.

## Parameters
    faces (Tensor): [T, 3] triangular face indices
    edges (Tensor, optional): [E, 2] edge indices. Defaults to None.
    face2edge (Tensor, optional): [T, 3] mapping from face to edge. Defaults to None.
        NOTE: If edges and face2edge are not provided, they will be computed.

## Returns
    components (List[Tensor]): list of connected faces"""
    utils3d.torch.mesh.compute_connected_components

@overload
def compute_edge_connected_components(edges: torch_.Tensor) -> List[torch_.Tensor]:
    """Compute connected edges of a mesh.

## Parameters
    edges (Tensor): [E, 2] edge indices

## Returns
    components (List[Tensor]): list of connected edges"""
    utils3d.torch.mesh.compute_edge_connected_components

@overload
def compute_boundaries(faces: torch_.Tensor, edges: torch_.Tensor = None, face2edge: torch_.Tensor = None, edge_degrees: torch_.Tensor = None) -> Tuple[List[torch_.Tensor], List[torch_.Tensor]]:
    """Compute boundary edges of a mesh.

## Parameters
    faces (Tensor): [T, 3] triangular face indices
    edges (Tensor): [E, 2] edge indices.
    face2edge (Tensor): [T, 3] mapping from face to edge.
    edge_degrees (Tensor): [E] degree of each edge.

## Returns
    boundary_edge_indices (List[Tensor]): list of boundary edge indices
    boundary_face_indices (List[Tensor]): list of boundary face indices"""
    utils3d.torch.mesh.compute_boundaries

@overload
def compute_dual_graph(face2edge: torch_.Tensor) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Compute dual graph of a mesh.

## Parameters
    face2edge (Tensor): [T, 3] mapping from face to edge.
        
## Returns
    dual_edges (Tensor): [DE, 2] face indices of dual edges
    dual_edge2edge (Tensor): [DE] mapping from dual edge to edge"""
    utils3d.torch.mesh.compute_dual_graph

@overload
def remove_unused_vertices(faces: torch_.Tensor, *vertice_attrs, return_indices: bool = False) -> Tuple[torch_.Tensor, ...]:
    """Remove unreferenced vertices of a mesh. 
Unreferenced vertices are removed, and the face indices are updated accordingly.

## Parameters
    faces (Tensor): [T, P] face indices
    *vertice_attrs: vertex attributes

## Returns
    faces (Tensor): [T, P] face indices
    *vertice_attrs: vertex attributes
    indices (Tensor, optional): [N] indices of vertices that are kept. Defaults to None."""
    utils3d.torch.mesh.remove_unused_vertices

@overload
def remove_corrupted_faces(faces: torch_.Tensor) -> torch_.Tensor:
    """Remove corrupted faces (faces with duplicated vertices)

## Parameters
    faces (Tensor): [T, 3] triangular face indices

## Returns
    Tensor: [T_, 3] triangular face indices"""
    utils3d.torch.mesh.remove_corrupted_faces

@overload
def remove_isolated_pieces(vertices: torch_.Tensor, faces: torch_.Tensor, connected_components: List[torch_.Tensor] = None, thresh_num_faces: int = None, thresh_radius: float = None, thresh_boundary_ratio: float = None, remove_unreferenced: bool = True) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Remove isolated pieces of a mesh. 
Isolated pieces are removed, and the face indices are updated accordingly.
If no face is left, will return the largest connected component.

## Parameters
    vertices (Tensor): [N, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices
    connected_components (List[Tensor], optional): connected components of the mesh. If None, it will be computed. Defaults to None.
    thresh_num_faces (int, optional): threshold of number of faces for isolated pieces. Defaults to None.
    thresh_radius (float, optional): threshold of radius for isolated pieces. Defaults to None.
    remove_unreferenced (bool, optional): remove unreferenced vertices after removing isolated pieces. Defaults to True.

## Returns
    vertices (Tensor): [N_, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices"""
    utils3d.torch.mesh.remove_isolated_pieces

@overload
def merge_duplicate_vertices(vertices: torch_.Tensor, faces: torch_.Tensor, tol: float = 1e-06) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Merge duplicate vertices of a triangular mesh. 
Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

## Parameters
    vertices (Tensor): [N, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices
    tol (float, optional): tolerance for merging. Defaults to 1e-6.

## Returns
    vertices (Tensor): [N_, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices"""
    utils3d.torch.mesh.merge_duplicate_vertices

@overload
def subdivide_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, n: int = 1) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.

## Parameters
    vertices (Tensor): [N, 3] 3-dimensional vertices
    faces (Tensor): [T, 3] triangular face indices
    n (int, optional): number of subdivisions. Defaults to 1.

## Returns
    vertices (Tensor): [N_, 3] subdivided 3-dimensional vertices
    faces (Tensor): [4 * T, 3] subdivided triangular face indices"""
    utils3d.torch.mesh.subdivide_mesh

@overload
def compute_face_tbn(pos: torch_.Tensor, faces_pos: torch_.Tensor, uv: torch_.Tensor, faces_uv: torch_.Tensor, eps: float = 1e-07) -> torch_.Tensor:
    """compute TBN matrix for each face

## Parameters
    pos (Tensor): shape (..., N_pos, 3), positions
    faces_pos (Tensor): shape(T, 3) 
    uv (Tensor): shape (..., N_uv, 3) uv coordinates, 
    faces_uv (Tensor): shape(T, 3) 
    
## Returns
    Tensor: (..., T, 3, 3) TBN matrix for each face. Note TBN vectors are normalized but not necessarily orthognal"""
    utils3d.torch.mesh.compute_face_tbn

@overload
def compute_vertex_tbn(faces_topo: torch_.Tensor, pos: torch_.Tensor, faces_pos: torch_.Tensor, uv: torch_.Tensor, faces_uv: torch_.Tensor) -> torch_.Tensor:
    """compute TBN matrix for each face

## Parameters
    faces_topo (Tensor): (T, 3), face indice of topology
    pos (Tensor): shape (..., N_pos, 3), positions
    faces_pos (Tensor): shape(T, 3) 
    uv (Tensor): shape (..., N_uv, 3) uv coordinates, 
    faces_uv (Tensor): shape(T, 3) 
    
## Returns
    Tensor: (..., V, 3, 3) TBN matrix for each face. Note TBN vectors are normalized but not necessarily orthognal"""
    utils3d.torch.mesh.compute_vertex_tbn

@overload
def laplacian(vertices: torch_.Tensor, faces: torch_.Tensor, weight: str = 'uniform') -> torch_.Tensor:
    """Laplacian smooth with cotangent weights

## Parameters
    vertices (Tensor): shape (..., N, 3)
    faces (Tensor): shape (T, 3)
    weight (str): 'uniform' or 'cotangent'"""
    utils3d.torch.mesh.laplacian

@overload
def laplacian_smooth_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, weight: str = 'uniform', times: int = 5) -> torch_.Tensor:
    """Laplacian smooth with cotangent weights

## Parameters
    vertices (Tensor): shape (..., N, 3)
    faces (Tensor): shape (T, 3)
    weight (str): 'uniform' or 'cotangent'"""
    utils3d.torch.mesh.laplacian_smooth_mesh

@overload
def taubin_smooth_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, lambda_: float = 0.5, mu_: float = -0.51) -> torch_.Tensor:
    """Taubin smooth mesh

## Parameters
    vertices (Tensor): _description_
    faces (Tensor): _description_
    lambda_ (float, optional): _description_. Defaults to 0.5.
    mu_ (float, optional): _description_. Defaults to -0.51.

## Returns
    Tensor: _description_"""
    utils3d.torch.mesh.taubin_smooth_mesh

@overload
def laplacian_hc_smooth_mesh(vertices: torch_.Tensor, faces: torch_.Tensor, times: int = 5, alpha: float = 0.5, beta: float = 0.5, weight: str = 'uniform'):
    """HC algorithm from Improved Laplacian Smoothing of Noisy Surface Meshes by J.Vollmer et al.
    """
    utils3d.torch.mesh.laplacian_hc_smooth_mesh

@overload
def uv_map(height: int, width: int, left: float = 0.0, top: float = 0.0, right: float = 1.0, bottom: float = 1.0, dtype: torch_.dtype = torch_.float32, device: torch_.device = None) -> torch_.Tensor:
    """Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.
This is commonly used as normalized image coordinates in texture mapping (when image is not flipped vertically).

## Parameters
    * `height`: `int` image height
    * `width`: `int` image width
    * `left`: `float`, optional left boundary in uv space. Defaults to 0.
    * `top`: `float`, optional top boundary in uv space. Defaults to 0.
    * `right`: `float`, optional right boundary in uv space. Defaults to 1.
    * `bottom`: `float`, optional bottom boundary in uv space. Defaults to 1.
    * `dtype`: `np.dtype`, optional data type of the output uv map. Defaults to np.float32.

## Returns
    - `uv (Tensor)`: shape `(height, width, 2)`

## Example Usage

>>> uv_map(10, 10):
[[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
 [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
  ...             ...                  ...
 [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]"""
    utils3d.torch.maps.uv_map

@overload
def pixel_coord_map(height: int, width: int, left: int = 0, top: int = 0, definition: Literal['corner', 'center'] = 'corner', dtype: torch_.dtype = torch_.float32, device: torch_.device = None) -> torch_.Tensor:
    """Get image pixel coordinates map, where (0, 0) is the top-left corner of the top-left pixel, and (width, height) is the bottom-right corner of the bottom-right pixel.

## Parameters
    - `height`: `int` image height
    - `width`: `int` image width
    - `left`: `int`, optional left boundary of the pixel coord map. Defaults to 0.
    - `top`: `int`, optional top boundary of the pixel coord map. Defaults to 0.
    - `definition`: `str`, optional 'corner' or 'center', whether the coordinates represent the corner or the center of the pixel. Defaults to 'corner'.
        - 'corner': coordinates range in [0, width - 1], [0, height - 1]
        - 'center': coordinates range in [0.5, width - 0.5], [0.5, height - 0.5]
    - `dtype`: `np.dtype`, optional data type of the output pixel coord map. Defaults to np.float32.

## Returns
    Tensor: shape (height, width, 2)

>>> pixel_coord_map(10, 10, definition='center', dtype=np.float32):
[[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
 [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
  ...             ...                  ...
[[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]

>>> pixel_coord_map(10, 10, definition='corner', dtype=np.int32):
[[[0, 0], [1, 0], ..., [9, 0]],
 [[0, 1], [1, 1], ..., [9, 1]],
    ...      ...         ...
 [[0, 9], [1, 9], ..., [9, 9]]]"""
    utils3d.torch.maps.pixel_coord_map

@overload
def build_mesh_from_map(*maps: torch_.Tensor, mask: Optional[torch_.Tensor] = None, tri: bool = False) -> Tuple[torch_.Tensor, ...]:
    """Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces.

## Parameters
    *maps (Tensor): attribute maps in shape (height, width, [channels])
    mask (Tensor, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

## Returns
    faces (Tensor): faces connecting neighboring pixels. shape (T, 4) if tri is False, else (T, 3)
    *attributes (Tensor): vertex attributes in corresponding order with input maps
    indices (Tensor, optional): indices of vertices in the original mesh"""
    utils3d.torch.maps.build_mesh_from_map

@overload
def build_mesh_from_depth_map(depth: torch_.Tensor, *other_maps: torch_.Tensor, intrinsics: torch_.Tensor, extrinsics: Optional[torch_.Tensor] = None, atol: Optional[float] = None, rtol: Optional[float] = 0.05, tri: bool = False) -> Tuple[torch_.Tensor, ...]:
    """Get a mesh by lifting depth map to 3D, while removing depths of large depth difference.

## Parameters
    depth (Tensor): [H, W] depth map
    extrinsics (Tensor, optional): [4, 4] extrinsics matrix. Defaults to None.
    intrinsics (Tensor, optional): [3, 3] intrinsics matrix. Defaults to None.
    *other_maps (Tensor): [H, W, C] vertex attributes. Defaults to None.
    atol (float, optional): absolute tolerance. Defaults to None.
    rtol (float, optional): relative tolerance. Defaults to None.
        triangles with vertices having depth difference larger than atol + rtol * depth will be marked.
    remove_by_depth (bool, optional): whether to remove triangles with large depth difference. Defaults to True.
    return_uv (bool, optional): whether to return uv coordinates. Defaults to False.
    return_indices (bool, optional): whether to return indices of vertices in the original mesh. Defaults to False.

## Returns
    faces (Tensor): [T, 3] faces
    vertices (Tensor): [N, 3] vertices
    *other_attrs (Tensor): [N, C] vertex attributes"""
    utils3d.torch.maps.build_mesh_from_depth_map

@overload
def depth_map_edge(depth: torch_.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch_.Tensor = None) -> torch_.BoolTensor:
    """Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.

## Parameters
    depth (Tensor): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

## Returns
    edge (Tensor): shape (..., height, width) of dtype torch.bool"""
    utils3d.torch.maps.depth_map_edge

@overload
def depth_map_aliasing(depth: torch_.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch_.Tensor = None) -> torch_.BoolTensor:
    """Compute the map that indicates the aliasing of a depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
## Parameters
    depth (Tensor): shape (..., height, width), linear depth map
    atol (float): absolute tolerance
    rtol (float): relative tolerance

## Returns
    edge (Tensor): shape (..., height, width) of dtype torch.bool"""
    utils3d.torch.maps.depth_map_aliasing

@overload
def point_map_to_normal_map(point: torch_.Tensor, mask: torch_.Tensor = None) -> torch_.Tensor:
    """Calculate normal map from point map. Value range is [-1, 1].

## Parameters
    point (Tensor): shape (..., height, width, 3), point map
## Returns
    normal (Tensor): shape (..., height, width, 3), normal map. """
    utils3d.torch.maps.point_map_to_normal_map

@overload
def depth_map_to_point_map(depth: torch_.Tensor, intrinsics: torch_.Tensor, extrinsics: torch_.Tensor = None):
    utils3d.torch.maps.depth_map_to_point_map

@overload
def depth_map_to_normal_map(depth: torch_.Tensor, intrinsics: torch_.Tensor, mask: torch_.Tensor = None) -> torch_.Tensor:
    """Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenCV identity camera's coordinate system.

## Parameters
    depth (Tensor): shape (..., height, width), linear depth map
    intrinsics (Tensor): shape (..., 3, 3), intrinsics matrix
## Returns
    normal (Tensor): shape (..., 3, height, width), normal map. """
    utils3d.torch.maps.depth_map_to_normal_map

@overload
def chessboard(width: int, height: int, grid_size: int, color_a: torch_.Tensor, color_b: torch_.Tensor) -> torch_.Tensor:
    """get a chessboard image

## Parameters
    width (int): image width
    height (int): image height
    grid_size (int): size of chessboard grid
    color_a (Tensor): shape (chanenls,), color of the grid at the top-left corner
    color_b (Tensor): shape (chanenls,), color in complementary grids

## Returns
    image (Tensor): shape (height, width, channels), chessboard image"""
    utils3d.torch.maps.chessboard

@overload
def bounding_rect_from_mask(mask: torch_.BoolTensor):
    """get bounding rectangle of a mask

## Parameters
    mask (Tensor): shape (..., height, width), mask

## Returns
    rect (Tensor): shape (..., 4), bounding rectangle (left, top, right, bottom)"""
    utils3d.torch.maps.bounding_rect_from_mask

@overload
def RastContext(nvd_ctx: Union[nvdiffrast.torch.ops.RasterizeCudaContext, nvdiffrast.torch.ops.RasterizeGLContext] = None, *, backend: Literal['cuda', 'gl'] = 'gl', device: Union[str, torch_.device] = None):
    """Create a rasterization context. Nothing but a wrapper of nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext."""
    utils3d.torch.rasterization.RastContext

@overload
def rasterize_triangles(ctx: utils3d.torch.rasterization.RastContext, width: int, height: int, *, vertices: torch_.Tensor, faces: torch_.Tensor, attr: torch_.Tensor = None, uv: torch_.Tensor = None, texture: torch_.Tensor = None, model: torch_.Tensor = None, view: torch_.Tensor = None, projection: torch_.Tensor = None, antialiasing: Union[bool, List[int]] = True, diff_attrs: Optional[List[int]] = None) -> Tuple[torch_.Tensor, torch_.Tensor, Optional[torch_.Tensor]]:
    """Rasterize a mesh with vertex attributes.

## Parameters
    ctx (GLContext): rasterizer context
    vertices (np.ndarray): (B, N, 2 or 3 or 4)
    faces (torch.Tensor): (T, 3)
    width (int): width of the output image
    height (int): height of the output image
    attr (torch.Tensor, optional): (B, N, C) vertex attributes. Defaults to None.
    uv (torch.Tensor, optional): (B, N, 2) uv coordinates. Defaults to None.
    texture (torch.Tensor, optional): (B, C, H, W) texture. Defaults to None.
    model (torch.Tensor, optional): ([B,] 4, 4) model matrix. Defaults to None (identity).
    view (torch.Tensor, optional): ([B,] 4, 4) view matrix. Defaults to None (identity).
    projection (torch.Tensor, optional): ([B,] 4, 4) projection matrix. Defaults to None (identity).
    antialiasing (Union[bool, List[int]], optional): whether to perform antialiasing. Defaults to True. If a list of indices is provided, only those channels will be antialiased.
    diff_attrs (Union[None, List[int]], optional): indices of attributes to compute screen-space derivatives. Defaults to None.

## Returns
    Dictionary containing:
      - image: (torch.Tensor): (B, C, H, W)
      - depth: (torch.Tensor): (B, H, W) screen space depth, ranging from 0 (near) to 1. (far)
               NOTE: Empty pixels will have depth 1., i.e. far plane.
      - mask: (torch.BoolTensor): (B, H, W) mask of valid pixels
      - image_dr: (torch.Tensor): (B, *, H, W) screen space derivatives of the attributes
      - face_id: (torch.Tensor): (B, H, W) face ids
      - uv: (torch.Tensor): (B, H, W, 2) uv coordinates (if uv is not None)
      - uv_dr: (torch.Tensor): (B, H, W, 4) uv derivatives (if uv is not None)
      - texture: (torch.Tensor): (B, C, H, W) texture (if uv and texture are not None)"""
    utils3d.torch.rasterization.rasterize_triangles

@overload
def rasterize_triangles_peeling(ctx: utils3d.torch.rasterization.RastContext, vertices: torch_.Tensor, faces: torch_.Tensor, width: int, height: int, max_layers: int, attr: torch_.Tensor = None, uv: torch_.Tensor = None, texture: torch_.Tensor = None, model: torch_.Tensor = None, view: torch_.Tensor = None, projection: torch_.Tensor = None, antialiasing: Union[bool, List[int]] = True, diff_attrs: Optional[List[int]] = None) -> Tuple[torch_.Tensor, torch_.Tensor, Optional[torch_.Tensor]]:
    """Rasterize a mesh with vertex attributes using depth peeling.

## Parameters
    ctx (GLContext): rasterizer context
    vertices (np.ndarray): (B, N, 2 or 3 or 4)
    faces (torch.Tensor): (T, 3)
    width (int): width of the output image
    height (int): height of the output image
    max_layers (int): maximum number of layers
        NOTE: if the number of layers is less than max_layers, the output will contain less than max_layers images.
    attr (torch.Tensor, optional): (B, N, C) vertex attributes. Defaults to None.
    uv (torch.Tensor, optional): (B, N, 2) uv coordinates. Defaults to None.
    texture (torch.Tensor, optional): (B, C, H, W) texture. Defaults to None.
    model (torch.Tensor, optional): ([B,] 4, 4) model matrix. Defaults to None (identity).
    view (torch.Tensor, optional): ([B,] 4, 4) view matrix. Defaults to None (identity).
    projection (torch.Tensor, optional): ([B,] 4, 4) projection matrix. Defaults to None (identity).
    antialiasing (Union[bool, List[int]], optional): whether to perform antialiasing. Defaults to True. If a list of indices is provided, only those channels will be antialiased.
    diff_attrs (Union[None, List[int]], optional): indices of attributes to compute screen-space derivatives. Defaults to None.

## Returns
    Dictionary containing:
      - image: (List[torch.Tensor]): list of (B, C, H, W) rendered images
      - depth: (List[torch.Tensor]): list of (B, H, W) screen space depth, ranging from 0 (near) to 1. (far)
                 NOTE: Empty pixels will have depth 1., i.e. far plane.
      - mask: (List[torch.BoolTensor]): list of (B, H, W) mask of valid pixels
      - image_dr: (List[torch.Tensor]): list of (B, *, H, W) screen space derivatives of the attributes
      - face_id: (List[torch.Tensor]): list of (B, H, W) face ids
      - uv: (List[torch.Tensor]): list of (B, H, W, 2) uv coordinates (if uv is not None)
      - uv_dr: (List[torch.Tensor]): list of (B, H, W, 4) uv derivatives (if uv is not None)
      - texture: (List[torch.Tensor]): list of (B, C, H, W) texture (if uv and texture are not None)"""
    utils3d.torch.rasterization.rasterize_triangles_peeling

@overload
def sample_texture(texture: torch_.Tensor, uv: torch_.Tensor, uv_da: torch_.Tensor) -> torch_.Tensor:
    """Interpolate texture using uv coordinates.

## Parameters
    texture (torch.Tensor): (B, C, H, W) texture
    uv (torch.Tensor): (B, H, W, 2) uv coordinates
    uv_da (torch.Tensor): (B, H, W, 4) uv derivatives
    
## Returns
    torch.Tensor: (B, C, H, W) interpolated texture"""
    utils3d.torch.rasterization.sample_texture

@overload
def texture_composite(texture: torch_.Tensor, uv: List[torch_.Tensor], uv_da: List[torch_.Tensor], background: torch_.Tensor = None) -> Tuple[torch_.Tensor, torch_.Tensor]:
    """Composite textures with depth peeling output.

## Parameters
    texture (torch.Tensor): (B, C+1, H, W) texture
        NOTE: the last channel is alpha channel
    uv (List[torch.Tensor]): list of (B, H, W, 2) uv coordinates
    uv_da (List[torch.Tensor]): list of (B, H, W, 4) uv derivatives
    background (Optional[torch.Tensor], optional): (B, C, H, W) background image. Defaults to None (black).
    
## Returns
    image: (torch.Tensor): (B, C, H, W) rendered image
    alpha: (torch.Tensor): (B, H, W) alpha channel"""
    utils3d.torch.rasterization.texture_composite

@overload
def warp_image_by_depth(ctx: utils3d.torch.rasterization.RastContext, depth: torch_.FloatTensor, image: torch_.FloatTensor = None, mask: torch_.BoolTensor = None, width: int = None, height: int = None, *, extrinsics_src: torch_.FloatTensor = None, extrinsics_tgt: torch_.FloatTensor = None, intrinsics_src: torch_.FloatTensor = None, intrinsics_tgt: torch_.FloatTensor = None, near: float = 0.1, far: float = 100.0, antialiasing: bool = True, backslash: bool = False, padding: int = 0, return_uv: bool = False, return_dr: bool = False) -> Tuple[torch_.FloatTensor, torch_.FloatTensor, torch_.BoolTensor, Optional[torch_.FloatTensor], Optional[torch_.FloatTensor]]:
    """Warp image by depth. 
NOTE: if batch size is 1, image mesh will be triangulated aware of the depth, yielding less distorted results.
Otherwise, image mesh will be triangulated simply for batch rendering.

## Parameters
    ctx (Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]): rasterization context
    depth (torch.Tensor): (B, H, W) linear depth
    image (torch.Tensor): (B, C, H, W). None to use image space uv. Defaults to None.
    width (int, optional): width of the output image. None to use the same as depth. Defaults to None.
    height (int, optional): height of the output image. Defaults the same as depth..
    extrinsics_src (torch.Tensor, optional): (B, 4, 4) extrinsics matrix for source. None to use identity. Defaults to None.
    extrinsics_tgt (torch.Tensor, optional): (B, 4, 4) extrinsics matrix for target. None to use identity. Defaults to None.
    intrinsics_src (torch.Tensor, optional): (B, 3, 3) intrinsics matrix for source. None to use the same as target. Defaults to None.
    intrinsics_tgt (torch.Tensor, optional): (B, 3, 3) intrinsics matrix for target. None to use the same as source. Defaults to None.
    near (float, optional): near plane. Defaults to 0.1. 
    far (float, optional): far plane. Defaults to 100.0.
    antialiasing (bool, optional): whether to perform antialiasing. Defaults to True.
    backslash (bool, optional): whether to use backslash triangulation. Defaults to False.
    padding (int, optional): padding of the image. Defaults to 0.
    return_uv (bool, optional): whether to return the uv. Defaults to False.
    return_dr (bool, optional): whether to return the image-space derivatives of uv. Defaults to False.

## Returns
    image: (torch.FloatTensor): (B, C, H, W) rendered image
    depth: (torch.FloatTensor): (B, H, W) linear depth, ranging from 0 to inf
    mask: (torch.BoolTensor): (B, H, W) mask of valid pixels
    uv: (torch.FloatTensor): (B, 2, H, W) image-space uv
    dr: (torch.FloatTensor): (B, 4, H, W) image-space derivatives of uv"""
    utils3d.torch.rasterization.warp_image_by_depth

@overload
def warp_image_by_forward_flow(ctx: utils3d.torch.rasterization.RastContext, image: torch_.FloatTensor, flow: torch_.FloatTensor, depth: torch_.FloatTensor = None, *, antialiasing: bool = True, backslash: bool = False) -> Tuple[torch_.FloatTensor, torch_.BoolTensor]:
    """Warp image by forward flow.
NOTE: if batch size is 1, image mesh will be triangulated aware of the depth, yielding less distorted results.
Otherwise, image mesh will be triangulated simply for batch rendering.

## Parameters
    ctx (Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]): rasterization context
    image (torch.Tensor): (B, C, H, W) image
    flow (torch.Tensor): (B, 2, H, W) forward flow
    depth (torch.Tensor, optional): (B, H, W) linear depth. If None, will use the same for all pixels. Defaults to None.
    antialiasing (bool, optional): whether to perform antialiasing. Defaults to True.
    backslash (bool, optional): whether to use backslash triangulation. Defaults to False.

## Returns
    image: (torch.FloatTensor): (B, C, H, W) rendered image
    mask: (torch.BoolTensor): (B, H, W) mask of valid pixels"""
    utils3d.torch.rasterization.warp_image_by_forward_flow

