# utils3d

A collectioin of useful Python utility functions for 3D computer vision and graphics researchers.

> ***NOTE: This repo is frequently updated and never ensures backward compatibility.***
> - If you are using `utils3d` as a dependency, please use a commit ID or fork this repo.
> - If you find some functions helpful here, consider copying and pasting the functions to your own code.

## Install

Install by git

```bash
pip install git+https://github.com/EasternJournalist/utils3d.git
```

or clone the repo and install with `-e` option for convenient updating and modifying.

```bash
git clone https://github.com/EasternJournalist/utils3d.git
pip install -e ./utils3d
```

## Documentation

### Transformations

| Function | Numpy | Pytorch| Description |
|----------|-------| -----  | ------------|
| `angle_between` | [`angle_between(v1, v2)`](utils3d/numpy/transforms.py#L1190) | [`angle_between(v1, v2, eps)`](utils3d/torch/transforms.py#L1236) | Computes the angle between two vectors. |
| `axis_angle_to_matrix` | [`axis_angle_to_matrix(axis_angle, eps)`](utils3d/numpy/transforms.py#L932) | [`axis_angle_to_matrix(axis_angle, eps)`](utils3d/torch/transforms.py#L870) | Converts an axis-angle representation to a rotation matrix. |
| `axis_angle_to_quaternion` | - | [`axis_angle_to_quaternion(axis_angle, eps)`](utils3d/torch/transforms.py#L927) | Converts an axis-angle representation to a quaternion. |
| `crop_intrinsics` | [`crop_intrinsics(intrinsics, width, height, left, top, crop_width, crop_height)`](utils3d/numpy/transforms.py#L391) | [`crop_intrinsics(intrinsics, width, height, left, top, crop_width, crop_height)`](utils3d/torch/transforms.py#L375) | Adjusts camera intrinsics for a cropped image. |
| `depth_buffer_to_linear` | [`depth_buffer_to_linear(depth_buffer, near, far)`](utils3d/numpy/transforms.py#L510) | [`depth_buffer_to_linear(depth, near, far)`](utils3d/torch/transforms.py#L495) | Converts depth buffer values to linear depth. |
| `depth_linear_to_buffer` | [`depth_linear_to_buffer(depth, near, far)`](utils3d/numpy/transforms.py#L490) | [`depth_linear_to_buffer(depth, near, far)`](utils3d/torch/transforms.py#L476) | Converts linear depth values to depth buffer values. |
| `euler_angles_to_matrix` | [`euler_angles_to_matrix(euler_angles, convention)`](utils3d/numpy/transforms.py#L881) | [`euler_angles_to_matrix(euler_angles, convention)`](utils3d/torch/transforms.py#L747) | Converts Euler angles to a rotation matrix. |
| `euler_axis_angle_rotation` | [`euler_axis_angle_rotation(axis, angle)`](utils3d/numpy/transforms.py#L851) | [`euler_axis_angle_rotation(axis, angle)`](utils3d/torch/transforms.py#L717) | Computes a rotation matrix from an axis and angle. |
| `extrinsics_look_at` | [`extrinsics_look_at(eye, look_at, up)`](utils3d/numpy/transforms.py#L239) | [`extrinsics_look_at(eye, look_at, up)`](utils3d/torch/transforms.py#L251) | Generates extrinsics for a camera looking at a target. |
| `extrinsics_to_essential` | [`extrinsics_to_essential(extrinsics)`](utils3d/numpy/transforms.py#L829) | [`extrinsics_to_essential(extrinsics)`](utils3d/torch/transforms.py#L1072) | Converts extrinsics to an essential matrix. |
| `extrinsics_to_view` | [`extrinsics_to_view(extrinsics)`](utils3d/numpy/transforms.py#L328) | [`extrinsics_to_view(extrinsics)`](utils3d/torch/transforms.py#L319) | Converts extrinsics to a view matrix. |
| `focal_to_fov` | [`focal_to_fov(focal)`](utils3d/numpy/transforms.py#L198) | [`focal_to_fov(focal)`](utils3d/torch/transforms.py#L209) | Converts focal length to field of view. |
| `fov_to_focal` | [`fov_to_focal(fov)`](utils3d/numpy/transforms.py#L202) | [`fov_to_focal(fov)`](utils3d/torch/transforms.py#L213) | Converts field of view to focal length. |
| `interpolate_extrinsics` | - | [`interpolate_extrinsics(ext1, ext2, t)`](utils3d/torch/transforms.py#L1034) | Interpolates between two extrinsics. |
| `interpolate_view` | - | [`interpolate_view(view1, view2, t)`](utils3d/torch/transforms.py#L1048) | Interpolates between two view matrices. |
| `intrinsics_from_focal_center` | [`intrinsics_from_focal_center(fx, fy, cx, cy)`](utils3d/numpy/transforms.py#L128) | [`intrinsics_from_focal_center(fx, fy, cx, cy)`](utils3d/torch/transforms.py#L135) | Generates camera intrinsics from focal length and principal point. |
| `intrinsics_from_fov` | [`intrinsics_from_fov(fov_x, fov_y, fov_max, fov_min, aspect_ratio)`](utils3d/numpy/transforms.py#L154) | [`intrinsics_from_fov(fov_x, fov_y, fov_max, fov_min, aspect_ratio)`](utils3d/torch/transforms.py#L165) | Generates camera intrinsics from field of view. |
| `intrinsics_to_fov` | [`intrinsics_to_fov(intrinsics)`](utils3d/numpy/transforms.py#L206) | [`intrinsics_to_fov(intrinsics)`](utils3d/torch/transforms.py#L217) | Converts camera intrinsics to field of view. |
| `intrinsics_to_perspective` | [`intrinsics_to_perspective(intrinsics, near, far)`](utils3d/numpy/transforms.py#L300) | [`intrinsics_to_perspective(intrinsics, near, far)`](utils3d/torch/transforms.py#L290) | Converts camera intrinsics to a perspective matrix. |
| `lerp` | [`lerp(x1, x2, t)`](utils3d/numpy/transforms.py#L1074) | - | Performs linear interpolation between two values. |
| `lerp_se3_matrix` | [`lerp_se3_matrix(T1, T2, t)`](utils3d/numpy/transforms.py#L1089) | - | Performs linear interpolation between two SE(3) matrices. |
| `make_se3_matrix` | [`make_se3_matrix(R, t)`](utils3d/numpy/transforms.py#L987) | [`make_se3_matrix(R, t)`](utils3d/torch/transforms.py#L1094) | Constructs an SE(3) matrix from rotation and translation. |
| `matrix_to_axis_angle` | - | [`matrix_to_axis_angle(rot_mat, eps)`](utils3d/torch/transforms.py#L897) | Converts a rotation matrix to axis-angle representation. |
| `matrix_to_euler_angles` | - | [`matrix_to_euler_angles(matrix, convention)`](utils3d/torch/transforms.py#L832) | Converts a rotation matrix to Euler angles. |
| `matrix_to_quaternion` | [`matrix_to_quaternion(rot_mat, eps)`](utils3d/numpy/transforms.py#L783) | [`matrix_to_quaternion(rot_mat, eps)`](utils3d/torch/transforms.py#L942) | Converts a rotation matrix to a quaternion. |
| `normalize_intrinsics` | [`normalize_intrinsics(intrinsics, width, height, integer_pixel_centers)`](utils3d/numpy/transforms.py#L354) | [`normalize_intrinsics(intrinsics, width, height)`](utils3d/torch/transforms.py#L347) | Normalizes camera intrinsics for image dimensions. |
| `perspective_from_fov` | [`perspective_from_fov(fov_x, fov_y, fov_min, fov_max, aspect_ratio, near, far)`](utils3d/numpy/transforms.py#L59) | [`perspective_from_fov(fov_x, fov_y, fov_min, fov_max, aspect_ratio, near, far)`](utils3d/torch/transforms.py#L64) | Generates a perspective matrix from field of view. |
| `perspective_from_window` | [`perspective_from_window(left, right, bottom, top, near, far)`](utils3d/numpy/transforms.py#L101) | [`perspective_from_window(left, right, bottom, top, near, far)`](utils3d/torch/transforms.py#L108) | Generates a perspective matrix from window dimensions. |
| `perspective_to_intrinsics` | [`perspective_to_intrinsics(perspective)`](utils3d/numpy/transforms.py#L272) | [`perspective_to_intrinsics(perspective)`](utils3d/torch/transforms.py#L272) | Converts a perspective matrix to camera intrinsics. |
| `perspective_to_near_far` | [`perspective_to_near_far(perspective)`](utils3d/numpy/transforms.py#L289) | - | Extracts near and far plane distances from a perspective matrix. |
| `piecewise_lerp` | [`piecewise_lerp(x, t, s, extrapolation_mode)`](utils3d/numpy/transforms.py#L1110) | - | Performs piecewise linear interpolation. |
| `piecewise_lerp_se3_matrix` | [`piecewise_lerp_se3_matrix(T, t, s, extrapolation_mode)`](utils3d/numpy/transforms.py#L1139) | - | Performs piecewise linear interpolation for SE(3) matrices. |
| `pixel_to_ndc` | [`pixel_to_ndc(pixel, width, height)`](utils3d/numpy/transforms.py#L467) | [`pixel_to_ndc(pixel, width, height)`](utils3d/torch/transforms.py#L453) | Converts pixel coordinates to normalized device coordinates. |
| `pixel_to_uv` | [`pixel_to_uv(pixel, width, height)`](utils3d/numpy/transforms.py#L427) | [`pixel_to_uv(pixel, width, height)`](utils3d/torch/transforms.py#L411) | Converts pixel coordinates to UV coordinates. |
| `project` | [`project(points, intrinsics, extrinsics, view, projection)`](utils3d/numpy/transforms.py#L675) | [`project(points, intrinsics, extrinsics, view, projection)`](utils3d/torch/transforms.py#L634) | Projects 3D points to 2D image coordinates. |
| `project_cv` | [`project_cv(points, intrinsics, extrinsics)`](utils3d/numpy/transforms.py#L559) | [`project_cv(points, intrinsics, extrinsics)`](utils3d/torch/transforms.py#L543) | Projects 3D points using OpenCV conventions. |
| `project_gl` | [`project_gl(points, projection, view)`](utils3d/numpy/transforms.py#L530) | [`project_gl(points, projection, view)`](utils3d/torch/transforms.py#L514) | Projects 3D points using OpenGL conventions. |
| `quaternion_to_axis_angle` | - | [`quaternion_to_axis_angle(quaternion, eps)`](utils3d/torch/transforms.py#L911) | Converts a quaternion to axis-angle representation. |
| `quaternion_to_matrix` | [`quaternion_to_matrix(quaternion, eps)`](utils3d/numpy/transforms.py#L758) | [`quaternion_to_matrix(quaternion, eps)`](utils3d/torch/transforms.py#L988) | Converts a quaternion to a rotation matrix. |
| `ray_intersection` | [`ray_intersection(p1, d1, p2, d2)`](utils3d/numpy/transforms.py#L959) | - | Computes the intersection of two rays. |
| `rotate_2d` | - | [`rotate_2d(theta, center)`](utils3d/torch/transforms.py#L1130) | Rotates a 2D point around a center. |
| `rotation_matrix_2d` | - | [`rotation_matrix_2d(theta)`](utils3d/torch/transforms.py#L1112) | Generates a 2D rotation matrix. |
| `rotation_matrix_from_vectors` | [`rotation_matrix_from_vectors(v1, v2)`](utils3d/numpy/transforms.py#L920) | [`rotation_matrix_from_vectors(v1, v2)`](utils3d/torch/transforms.py#L787) | Computes a rotation matrix aligning two vectors. |
| `scale_2d` | - | [`scale_2d(scale, center)`](utils3d/torch/transforms.py#L1184) | Scales a 2D point around a center. |
| `screen_coord_to_view_coord` | [`screen_coord_to_view_coord(screen_coord, projection)`](utils3d/numpy/transforms.py#L621) | - | Converts screen coordinates to view coordinates. |
| `skew_symmetric` | [`skew_symmetric(v)`](utils3d/numpy/transforms.py#L908) | [`skew_symmetric(v)`](utils3d/torch/transforms.py#L775) | Generates a skew-symmetric matrix from a vector. |
| `slerp` | - | [`slerp(rot_mat_1, rot_mat_2, t)`](utils3d/torch/transforms.py#L1013) | Performs spherical linear interpolation between two rotation matrices. |
| `slerp_quaternion` | [`slerp_quaternion(q1, q2, t)`](utils3d/numpy/transforms.py#L1006) | - | Performs spherical linear interpolation between two quaternions. |
| `slerp_vector` | [`slerp_vector(v1, v2, t)`](utils3d/numpy/transforms.py#L1051) | - | Performs spherical linear interpolation between two vectors. |
| `transform` | [`transform(x, Ts)`](utils3d/numpy/transforms.py#L1168) | [`transform(x, Ts)`](utils3d/torch/transforms.py#L1214) | Applies a transformation matrix to a set of points. |
| `translate_2d` | - | [`translate_2d(translation)`](utils3d/torch/transforms.py#L1161) | Translates a 2D point. |
| `unproject` | [`unproject(uv, depth, intrinsics, extrinsics, projection, view)`](utils3d/numpy/transforms.py#L716) | [`unproject(uv, depth, intrinsics, extrinsics, projection, view)`](utils3d/torch/transforms.py#L675) | Unprojects 2D points to 3D space. |
| `unproject_cv` | [`unproject_cv(uv, depth, intrinsics, extrinsics)`](utils3d/numpy/transforms.py#L645) | [`unproject_cv(uv, depth, intrinsics, extrinsics)`](utils3d/torch/transforms.py#L603) | Unprojects 2D points using OpenCV conventions. |
| `unproject_gl` | [`unproject_gl(uv, depth, projection, view)`](utils3d/numpy/transforms.py#L591) | [`unproject_gl(uv, depth, projection, view)`](utils3d/torch/transforms.py#L573) | Unprojects 2D points using OpenGL conventions. |
| `uv_to_pixel` | [`uv_to_pixel(uv, width, height)`](utils3d/numpy/transforms.py#L448) | [`uv_to_pixel(uv, width, height)`](utils3d/torch/transforms.py#L433) | Converts UV coordinates to pixel coordinates. |
| `view_look_at` | [`view_look_at(eye, look_at, up)`](utils3d/numpy/transforms.py#L212) | [`view_look_at(eye, look_at, up)`](utils3d/torch/transforms.py#L224) | Generates a view matrix for a camera looking at a target. |
| `view_to_extrinsics` | [`view_to_extrinsics(view)`](utils3d/numpy/transforms.py#L341) | [`view_to_extrinsics(view)`](utils3d/torch/transforms.py#L334) | Converts a view matrix to extrinsics. |

### Mesh

| Function | Numpy | Pytorch| Description |
|----------|-------|--------|-------------|
| `calc_quad_candidates` | [`calc_quad_candidates(edges, face2edge, edge2face)`](utils3d/numpy/mesh.py#L490) | - | Identifies candidate quads from edges. |
| `calc_quad_direction` | [`calc_quad_direction(vertices, quads)`](utils3d/numpy/mesh.py#L602) | - | Computes the direction of quads. |
| `calc_quad_distortion` | [`calc_quad_distortion(vertices, quads)`](utils3d/numpy/mesh.py#L551) | - | Measures distortion in quads. |
| `calc_quad_smoothness` | [`calc_quad_smoothness(quad2edge, quad2adj, quads_direction)`](utils3d/numpy/mesh.py#L643) | - | Evaluates smoothness of quads. |
| `camera_frustum` | [`camera_frustum(extrinsics, intrinsics, depth)`](utils3d/numpy/mesh.py#L423) | - | Computes the camera frustum. |
| `compute_boundarys` | - | [`compute_boundarys(faces, edges, face2edge, edge_degrees)`](utils3d/torch/mesh.py#L331) | Computes mesh boundaries. |
| `compute_connected_components` | - | [`compute_connected_components(faces, edges, face2edge)`](utils3d/torch/mesh.py#L250) | Finds connected components in a mesh. |
| `compute_dual_graph` | - | [`compute_dual_graph(face2edge)`](utils3d/torch/mesh.py#L387) | Computes the dual graph of a mesh. |
| `compute_edge_connected_components` | - | [`compute_edge_connected_components(edges)`](utils3d/torch/mesh.py#L292) | Finds edge-connected components. |
| `compute_edges` | - | [`compute_edges(faces)`](utils3d/torch/mesh.py#L228) | Computes edges from faces. |
| `compute_face_angle` | [`compute_face_angle(vertices, faces, eps)`](utils3d/numpy/mesh.py#L107) | - | Computes angles of mesh faces. |
| `compute_face_angles` | - | [`compute_face_angles(vertices, faces)`](utils3d/torch/mesh.py#L145) | Computes angles of mesh faces. |
| `compute_face_normal` | [`compute_face_normal(vertices, faces)`](utils3d/numpy/mesh.py#L82) | [`compute_face_normal(vertices, faces)`](utils3d/torch/mesh.py#L120) | Computes normals of mesh faces. |
| `compute_face_tbn` | - | [`compute_face_tbn(pos, faces_pos, uv, faces_uv, eps)`](utils3d/torch/mesh.py#L573) | Computes tangent, bitangent, and normal for faces. |
| `compute_vertex_normal` | [`compute_vertex_normal(vertices, faces, face_normal)`](utils3d/numpy/mesh.py#L135) | [`compute_vertex_normal(vertices, faces, face_normal)`](utils3d/torch/mesh.py#L170) | Computes vertex normals by averaging face normals. |
| `compute_vertex_normal_weighted` | [`compute_vertex_normal_weighted(vertices, faces, face_normal)`](utils3d/numpy/mesh.py#L168) | [`compute_vertex_normal_weighted(vertices, faces, face_normal)`](utils3d/torch/mesh.py#L199) | Computes vertex normals weighted by face angles. |
| `compute_vertex_tbn` | - | [`compute_vertex_tbn(faces_topo, pos, faces_pos, uv, faces_uv)`](utils3d/torch/mesh.py#L596) | Computes tangent, bitangent, and normal for vertices. |
| `cube` | [`cube(tri)`](utils3d/numpy/mesh.py#L392) | - | Generates a cube mesh. |
| `flatten_mesh_indices` | [`flatten_mesh_indices(args)`](utils3d/numpy/mesh.py#L361) | - | Flattens mesh indices. |
| `icosahedron` | [`icosahedron()`](utils3d/numpy/mesh.py#L449) | - | Generates an icosahedron mesh. |
| `laplacian` | - | [`laplacian(vertices, faces, weight)`](utils3d/torch/mesh.py#L619) | Computes the Laplacian of a mesh. |
| `laplacian_hc_smooth_mesh` | - | [`laplacian_hc_smooth_mesh(vertices, faces, times, alpha, beta, weight)`](utils3d/torch/mesh.py#L678) | Applies Laplacian HC smoothing to a mesh. |
| `laplacian_smooth_mesh` | - | [`laplacian_smooth_mesh(vertices, faces, weight, times)`](utils3d/torch/mesh.py#L648) | Applies Laplacian smoothing to a mesh. |
| `merge_duplicate_vertices` | [`merge_duplicate_vertices(vertices, faces, tol)`](utils3d/numpy/mesh.py#L216) | [`merge_duplicate_vertices(vertices, faces, tol)`](utils3d/torch/mesh.py#L451) | Merges duplicate vertices in a mesh. |
| `merge_meshes` | [`merge_meshes(meshes)`](utils3d/numpy/mesh.py#L465) | - | Merges multiple meshes into one. |
| `mesh_relations` | [`mesh_relations(faces)`](utils3d/numpy/mesh.py#L306) | - | Computes relationships between mesh elements. |
| `remove_corrupted_faces` | [`remove_corrupted_faces(faces)`](utils3d/numpy/mesh.py#L202) | [`remove_corrupted_faces(faces)`](utils3d/torch/mesh.py#L435) | Removes corrupted faces from a mesh. |
| `remove_isolated_pieces` | - | [`remove_isolated_pieces(vertices, faces, connected_components, thresh_num_faces, thresh_radius, thresh_boundary_ratio, remove_unreferenced)`](utils3d/torch/mesh.py#L476) | Removes isolated pieces from a mesh. |
| `remove_unused_vertices` | [`remove_unused_vertices(faces, vertice_attrs, return_indices)`](utils3d/numpy/mesh.py#L241) | [`remove_unused_vertices(faces, vertice_attrs, return_indices)`](utils3d/torch/mesh.py#L406) | Removes unused vertices from a mesh. |
| `solve_quad` | [`solve_quad(face2edge, edge2face, quad2adj, quads_distortion, quads_smoothness, quads_valid)`](utils3d/numpy/mesh.py#L672) | - | Solves for optimal quads in a mesh. |
| `solve_quad_qp` | [`solve_quad_qp(face2edge, edge2face, quad2adj, quads_distortion, quads_smoothness, quads_valid)`](utils3d/numpy/mesh.py#L785) | - | Solves for optimal quads using quadratic programming. |
| `square` | [`square(tri)`](utils3d/numpy/mesh.py#L374) | - | Generates a square mesh. |
| `subdivide_mesh` | [`subdivide_mesh(vertices, faces, n)`](utils3d/numpy/mesh.py#L270) | [`subdivide_mesh(vertices, faces, n)`](utils3d/torch/mesh.py#L542) | Subdivides a mesh into smaller faces. |
| `taubin_smooth_mesh` | - | [`taubin_smooth_mesh(vertices, faces, lambda_, mu_)`](utils3d/torch/mesh.py#L661) | Applies Taubin smoothing to a mesh. |
| `tri_to_quad` | [`tri_to_quad(vertices, faces)`](utils3d/numpy/mesh.py#L871) | - | Converts triangular faces to quads. |
| `triangulate_mesh` | [`triangulate_mesh(faces, vertices, method)`](utils3d/numpy/mesh.py#L34) | [`triangulate_mesh(faces, vertices, method)`](utils3d/torch/mesh.py#L71) | Converts polygonal faces to triangles. |

### Maps


| Function | Numpy | Pytorch| Description |
|----------|-------|--------|-------------|
| `bounding_rect_from_mask` | - | [`bounding_rect_from_mask(mask)`](utils3d/torch/maps.py#L278) | Computes the bounding rectangle of a mask. |
| `build_mesh_from_depth_map` | [`build_mesh_from_depth_map(depth, extrinsics, intrinsics, vertice_attrs, atol, rtol, remove_by_depth, return_uv, return_indices)`](utils3d/numpy/maps.py#L413) | [`build_mesh_from_depth_map(depth, extrinsics, intrinsics, tri)`](utils3d/torch/maps.py#L198) | Constructs a mesh from a depth map. |
| `build_mesh_from_map` | [`build_mesh_from_map(maps, mask, tri, return_indices)`](utils3d/numpy/maps.py#L370) | [`build_mesh_from_map(maps, mask, device, dtype, return_indices)`](utils3d/torch/maps.py#L153) | Constructs a mesh from a map. |
| `chessboard` | [`chessboard(height, width, grid_size, color_a, color_b)`](utils3d/numpy/maps.py#L476) | [`chessboard(width, height, grid_size, color_a, color_b)`](utils3d/torch/maps.py#L295) | Generates a chessboard pattern. |
| `depth_map_aliasing` | [`depth_map_aliasing(depth, atol, rtol, kernel_size, mask)`](utils3d/numpy/maps.py#L55) | [`depth_map_aliasing(depth, atol, rtol, kernel_size, mask)`](utils3d/torch/maps.py#L121) | Identifies aliasing in a depth map. |
| `depth_map_edge` | [`depth_map_edge(depth, atol, rtol, kernel_size, mask)`](utils3d/numpy/maps.py#L28) | [`depth_map_edge(depth, atol, rtol, kernel_size, mask)`](utils3d/torch/maps.py#L90) | Detects edges in a depth map. |
| `depth_map_to_normal_map` | [`depth_map_to_normal_map(depth, intrinsics, mask, edge_threshold)`](utils3d/numpy/maps.py#L179) | [`depth_map_to_normal_map(depth, intrinsics, mask)`](utils3d/torch/maps.py#L257) | Converts a depth map to a normal map. |
| `depth_map_to_point_map` | [`depth_map_to_point_map(depth, intrinsics, extrinsics)`](utils3d/numpy/maps.py#L200) | [`depth_map_to_point_map(depth, intrinsics, extrinsics)`](utils3d/torch/maps.py#L271) | Converts a depth map to a point map. |
| `normal_map_edge` | [`normal_map_edge(normals, tol, kernel_size, mask)`](utils3d/numpy/maps.py#L83) | - | Detects edges in a normal map. |
| `pixel_center_coord_map` | [`pixel_center_coord_map(height, width, left, top, right, bottom, dtype)`](utils3d/numpy/maps.py#L299) | [`pixel_center_coord_map(height, width, left, top, right, bottom, dtype, device)`](utils3d/torch/maps.py#L53) | Generates a map of pixel center coordinates. |
| `pixel_coord_map` | [`pixel_coord_map(height, width, left, top, right, bottom, dtype)`](utils3d/numpy/maps.py#L335) | - | Generates a map of pixel coordinates. |
| `point_map_to_normal_map` | [`point_map_to_normal_map(point, mask, edge_threshold)`](utils3d/numpy/maps.py#L121) | [`point_map_to_normal_map(point, mask)`](utils3d/torch/maps.py#L214) | Converts a point map to a normal map. |
| `screen_coord_map` | [`screen_coord_map(height, width)`](utils3d/numpy/maps.py#L227) | - | Generates a screen coordinate map. |
| `uv_map` | [`uv_map(height, width, mask, left, top, right, bottom, dtype)`](utils3d/numpy/maps.py#L250) | [`uv_map(height, width, left, top, right, bottom, device, dtype)`](utils3d/torch/maps.py#L25) | Generates a UV coordinate map. |

### Rasterization

| Function | Numpy | Pytorch| Description |
|----------|-------| -----  | ------------|
| `RastContext` | [`RastContext(args, kwargs)`](utils3d/numpy/rasterization.py#L53) | [`RastContext(nvd_ctx, backend, device)`](utils3d/torch/rasterization.py#L21) | Rasterization context for managing GPU resources and state. |
| `rasterize_lines` | [`rasterize_lines(ctx, width, height, vertices, lines, attributes, attributes_domain, view, projection, line_width, return_depth, return_interpolation, background_image, background_depth, background_interpolation_id, background_interpolation_uv)`](utils3d/numpy/rasterization.py#L897) | - | Rasterizes lines with attributes. | 
| `rasterize_point_cloud` | [`rasterize_point_cloud(ctx, width, height, points, point_sizes, point_size_in, point_shape, attributes, view, projection, return_depth, return_point_id, background_image, background_depth, background_point_id)`](utils3d/numpy/rasterization.py#L1115) | - | Rasterizes point clouds with attributes. |
| `rasterize_triangles` | [`rasterize_triangles(ctx, width, height, vertices, attributes, attributes_domain, faces, view, projection, cull_backface, return_depth, return_interpolation, background_image, background_depth, background_interpolation_id, background_interpolation_uv)`](utils3d/numpy/rasterization.py#L445) | [`rasterize_triangles(ctx, width, height, vertices, faces, attr, uv, texture, model, view, projection, antialiasing, diff_attrs)`](utils3d/torch/rasterization.py#L38) | Rasterizes triangle meshes with attributes. |
| `rasterize_triangles_peeling` | [`rasterize_triangles_peeling(ctx, width, height, vertices, attributes, attributes_domain, faces, view, projection, cull_backface, return_depth, return_interpolation)`](utils3d/numpy/rasterization.py#L666) | [`rasterize_triangles_peeling(ctx, vertices, faces, width, height, max_layers, attr, uv, texture, model, view, projection, antialiasing, diff_attrs)`](utils3d/torch/rasterization.py#L145) | Rasterizes triangle meshes with depth peeling to obtain multiple layers of surfaces. |
| `sample_texture` | [`sample_texture(ctx, uv_map, texture_map, interpolation, mipmap_level, repeat, anisotropic)`](utils3d/numpy/rasterization.py#L1304) | [`sample_texture(texture, uv, uv_da)`](utils3d/torch/rasterization.py#L271) | Samples a texture map using a UV map, supporting interpolation and wrapping. |
| `test_rasterization` | [`test_rasterization(ctx)`](utils3d/numpy/rasterization.py#L1455) | - | Runs a test rendering of a cube and saves the result as an image. |
| `texture_composite` | - | [`texture_composite(texture, uv, uv_da, background)`](utils3d/torch/rasterization.py#L291) | |
| `warp_image_by_depth` | - | [`warp_image_by_depth(ctx, depth, image, mask, width, height, extrinsics_src, extrinsics_tgt, intrinsics_src, intrinsics_tgt, near, far, antialiasing, backslash, padding, return_uv, return_dr)`](utils3d/torch/rasterization.py#L332) | Warps an image from one camera view to another using a depth map. |
| `warp_image_by_forward_flow` | - | [`warp_image_by_forward_flow(ctx, image, flow, depth, antialiasing, backslash)`](utils3d/torch/rasterization.py#L502) | Warps an image using a forward optical flow field and optional depth. |

### Utils

| Function | Numpy | Pytorch| Description |
|----------|-------| -----  | ------------|
| `lookup` | [`lookup(key, query, value, default_value)`](utils3d/numpy/utils.py#L84) | [`lookup(key, query)`](utils3d/torch/utils.py#L63) | Looks up query in key |
| `masked_max` | - | [`masked_max(input, mask, dim, keepdim)`](utils3d/torch/utils.py#L54) | Computes the maximum of input along a dimension, considering only elements where mask is True. |
| `masked_min` | - | [`masked_min(input, mask, dim, keepdim)`](utils3d/torch/utils.py#L45) | Computes the minimum of input along a dimension, considering only elements where mask is True. |
| `max_pool_1d` | [`max_pool_1d(x, kernel_size, stride, padding, axis)`](utils3d/numpy/utils.py#L56) | - | 1D max pooling over a specified axis with given kernel size, stride, and optional padding. |
| `max_pool_2d` | [`max_pool_2d(x, kernel_size, stride, padding, axis)`](utils3d/numpy/utils.py#L73) | - | 2D max pooling over specified axes with given kernel size, stride, and optional padding.|
| `max_pool_nd` | [`max_pool_nd(x, kernel_size, stride, padding, axis)`](utils3d/numpy/utils.py#L67) | - | N-dimensional max pooling by applying 1D max pooling sequentially over multiple axes. |
| `sliding_window_1d` | [`sliding_window_1d(x, window_size, stride, axis)`](utils3d/numpy/utils.py#L19) | [`sliding_window_1d(x, window_size, stride, dim)`](utils3d/torch/utils.py#L20) | Returns a sliding window view of the input array/tensor along one axis or dimension. |
| `sliding_window_2d` | [`sliding_window_2d(x, window_size, stride, axis)`](utils3d/numpy/utils.py#L48) | [`sliding_window_2d(x, window_size, stride, dim)`](utils3d/torch/utils.py#L36) | Returns a sliding window view of the input array/tensor along two axes or dimensions. |
| `sliding_window_nd` | [`sliding_window_nd(x, window_size, stride, axis)`](utils3d/numpy/utils.py#L41) | [`sliding_window_nd(x, window_size, stride, dim)`](utils3d/torch/utils.py#L28) | Returns a sliding window view of the input array/tensor along multiple axes or dimensions. |


