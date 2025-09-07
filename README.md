# utils3d

A collection of useful Python utility functions for 3D computer vision and graphics researchers.

> ***NOTE: This repo is frequently updated and never guarantees backward compatibility.***
> - If using `utils3d` as a dependency, please use a commit ID or fork this repo.
> - If you find some functions helpful here, consider copying and pasting the functions to your own code.

## Install

Install via pip + git

```bash
pip install git+https://github.com/EasternJournalist/utils3d.git
```

or clone the repo and install 

```bash
git clone https://github.com/EasternJournalist/utils3d.git
pip install ./utils3d
```

## Table of Contents

### Transforms

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.angle_between`<br>Calculate the angle between two vectors. | [`utils3d.np.angle_between(v1, v2)`](utils3d/numpy/transforms.py#L1209) | [`utils3d.th.angle_between(v1, v2, eps)`](utils3d/torch/transforms.py#L1257) | 
| `utils3d.axis_angle_to_matrix`<br>Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation | [`utils3d.np.axis_angle_to_matrix(axis_angle, eps)`](utils3d/numpy/transforms.py#L951) | [`utils3d.th.axis_angle_to_matrix(axis_angle, eps)`](utils3d/torch/transforms.py#L891) | 
| `utils3d.axis_angle_to_quaternion`<br> | - | [`utils3d.th.axis_angle_to_quaternion(axis_angle, eps)`](utils3d/torch/transforms.py#L948) | 
| `utils3d.crop_intrinsics`<br>Evaluate the new intrinsics after cropping the image | [`utils3d.np.crop_intrinsics(intrinsics, size, cropped_top, cropped_left, cropped_height, cropped_width)`](utils3d/numpy/transforms.py#L395) | [`utils3d.th.crop_intrinsics(intrinsics, size, cropped_top, cropped_left, cropped_height, cropped_width)`](utils3d/torch/transforms.py#L386) | 
| `utils3d.depth_buffer_to_linear`<br>OpenGL depth buffer to linear depth | [`utils3d.np.depth_buffer_to_linear(depth_buffer, near, far)`](utils3d/numpy/transforms.py#L529) | [`utils3d.th.depth_buffer_to_linear(depth, near, far)`](utils3d/torch/transforms.py#L516) | 
| `utils3d.depth_linear_to_buffer`<br>Project linear depth to depth value in screen space | [`utils3d.np.depth_linear_to_buffer(depth, near, far)`](utils3d/numpy/transforms.py#L509) | [`utils3d.th.depth_linear_to_buffer(depth, near, far)`](utils3d/torch/transforms.py#L497) | 
| `utils3d.euler_angles_to_matrix`<br>Convert rotations given as Euler angles in radians to rotation matrices. | [`utils3d.np.euler_angles_to_matrix(euler_angles, convention)`](utils3d/numpy/transforms.py#L900) | [`utils3d.th.euler_angles_to_matrix(euler_angles, convention)`](utils3d/torch/transforms.py#L768) | 
| `utils3d.euler_axis_angle_rotation`<br>Return the rotation matrices for one of the rotations about an axis | [`utils3d.np.euler_axis_angle_rotation(axis, angle)`](utils3d/numpy/transforms.py#L870) | [`utils3d.th.euler_axis_angle_rotation(axis, angle)`](utils3d/torch/transforms.py#L738) | 
| `utils3d.extrinsics_look_at`<br>Get OpenCV extrinsics matrix looking at something | [`utils3d.np.extrinsics_look_at(eye, look_at, up)`](utils3d/numpy/transforms.py#L239) | [`utils3d.th.extrinsics_look_at(eye, look_at, up)`](utils3d/torch/transforms.py#L251) | 
| `utils3d.extrinsics_to_essential`<br>extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0` | [`utils3d.np.extrinsics_to_essential(extrinsics)`](utils3d/numpy/transforms.py#L848) | [`utils3d.th.extrinsics_to_essential(extrinsics)`](utils3d/torch/transforms.py#L1093) | 
| `utils3d.extrinsics_to_view`<br>OpenCV camera extrinsics to OpenGL view matrix | [`utils3d.np.extrinsics_to_view(extrinsics)`](utils3d/numpy/transforms.py#L328) | [`utils3d.th.extrinsics_to_view(extrinsics)`](utils3d/torch/transforms.py#L319) | 
| `utils3d.focal_to_fov`<br> | [`utils3d.np.focal_to_fov(focal)`](utils3d/numpy/transforms.py#L198) | [`utils3d.th.focal_to_fov(focal)`](utils3d/torch/transforms.py#L209) | 
| `utils3d.fov_to_focal`<br> | [`utils3d.np.fov_to_focal(fov)`](utils3d/numpy/transforms.py#L202) | [`utils3d.th.fov_to_focal(fov)`](utils3d/torch/transforms.py#L213) | 
| `utils3d.interpolate_extrinsics`<br> | - | [`utils3d.th.interpolate_extrinsics(ext1, ext2, t)`](utils3d/torch/transforms.py#L1055) | 
| `utils3d.interpolate_view`<br> | - | [`utils3d.th.interpolate_view(view1, view2, t)`](utils3d/torch/transforms.py#L1069) | 
| `utils3d.intrinsics_from_focal_center`<br>Get OpenCV intrinsics matrix | [`utils3d.np.intrinsics_from_focal_center(fx, fy, cx, cy)`](utils3d/numpy/transforms.py#L128) | [`utils3d.th.intrinsics_from_focal_center(fx, fy, cx, cy)`](utils3d/torch/transforms.py#L135) | 
| `utils3d.intrinsics_from_fov`<br>Get normalized OpenCV intrinsics matrix from given field of view. | [`utils3d.np.intrinsics_from_fov(fov_x, fov_y, fov_max, fov_min, aspect_ratio)`](utils3d/numpy/transforms.py#L154) | [`utils3d.th.intrinsics_from_fov(fov_x, fov_y, fov_max, fov_min, aspect_ratio)`](utils3d/torch/transforms.py#L165) | 
| `utils3d.intrinsics_to_fov`<br> | [`utils3d.np.intrinsics_to_fov(intrinsics)`](utils3d/numpy/transforms.py#L206) | [`utils3d.th.intrinsics_to_fov(intrinsics)`](utils3d/torch/transforms.py#L217) | 
| `utils3d.intrinsics_to_perspective`<br>OpenCV intrinsics to OpenGL perspective matrix | [`utils3d.np.intrinsics_to_perspective(intrinsics, near, far)`](utils3d/numpy/transforms.py#L300) | [`utils3d.th.intrinsics_to_perspective(intrinsics, near, far)`](utils3d/torch/transforms.py#L290) | 
| `utils3d.lerp`<br>Linear interpolation between two vectors. | [`utils3d.np.lerp(x1, x2, t)`](utils3d/numpy/transforms.py#L1093) | - | 
| `utils3d.lerp_se3_matrix`<br>Linear interpolation between two SE(3) matrices. | [`utils3d.np.lerp_se3_matrix(T1, T2, t)`](utils3d/numpy/transforms.py#L1108) | - | 
| `utils3d.make_affine_matrix`<br>Make an affine transformation matrix from a linear matrix and a translation vector. | [`utils3d.np.make_affine_matrix(M, t)`](utils3d/numpy/transforms.py#L1006) | [`utils3d.th.make_affine_matrix(M, t)`](utils3d/torch/transforms.py#L1115) | 
| `utils3d.matrix_to_axis_angle`<br> | - | [`utils3d.th.matrix_to_axis_angle(rot_mat, eps)`](utils3d/torch/transforms.py#L918) | 
| `utils3d.matrix_to_euler_angles`<br> | - | [`utils3d.th.matrix_to_euler_angles(matrix, convention)`](utils3d/torch/transforms.py#L853) | 
| `utils3d.matrix_to_quaternion`<br>Convert 3x3 rotation matrix to quaternion (w, x, y, z) | [`utils3d.np.matrix_to_quaternion(rot_mat, eps)`](utils3d/numpy/transforms.py#L802) | [`utils3d.th.matrix_to_quaternion(rot_mat, eps)`](utils3d/torch/transforms.py#L963) | 
| `utils3d.normalize_intrinsics`<br>Normalize intrinsics from pixel cooridnates to uv coordinates | [`utils3d.np.normalize_intrinsics(intrinsics, size, pixel_definition)`](utils3d/numpy/transforms.py#L354) | [`utils3d.th.normalize_intrinsics(intrinsics, size, pixel_definition)`](utils3d/torch/transforms.py#L345) | 
| `utils3d.perspective_from_fov`<br>Get OpenGL perspective matrix from field of view  | [`utils3d.np.perspective_from_fov(fov_x, fov_y, fov_min, fov_max, aspect_ratio, near, far)`](utils3d/numpy/transforms.py#L59) | [`utils3d.th.perspective_from_fov(fov_x, fov_y, fov_min, fov_max, aspect_ratio, near, far)`](utils3d/torch/transforms.py#L64) | 
| `utils3d.perspective_from_window`<br>Get OpenGL perspective matrix from the window of z=-1 projection plane | [`utils3d.np.perspective_from_window(left, right, bottom, top, near, far)`](utils3d/numpy/transforms.py#L101) | [`utils3d.th.perspective_from_window(left, right, bottom, top, near, far)`](utils3d/torch/transforms.py#L108) | 
| `utils3d.perspective_to_intrinsics`<br>OpenGL perspective matrix to OpenCV intrinsics | [`utils3d.np.perspective_to_intrinsics(perspective)`](utils3d/numpy/transforms.py#L272) | [`utils3d.th.perspective_to_intrinsics(perspective)`](utils3d/torch/transforms.py#L272) | 
| `utils3d.perspective_to_near_far`<br>Get near and far planes from OpenGL perspective matrix | [`utils3d.np.perspective_to_near_far(perspective)`](utils3d/numpy/transforms.py#L289) | - | 
| `utils3d.piecewise_lerp`<br>Linear spline interpolation. | [`utils3d.np.piecewise_lerp(x, t, s, extrapolation_mode)`](utils3d/numpy/transforms.py#L1129) | - | 
| `utils3d.piecewise_lerp_se3_matrix`<br>Linear spline interpolation for SE(3) matrices. | [`utils3d.np.piecewise_lerp_se3_matrix(T, t, s, extrapolation_mode)`](utils3d/numpy/transforms.py#L1158) | - | 
| `utils3d.pixel_to_ndc`<br>Convert pixel coordinates to NDC (Normalized Device Coordinates). | [`utils3d.np.pixel_to_ndc(pixel, size, pixel_definition)`](utils3d/numpy/transforms.py#L481) | [`utils3d.th.pixel_to_ndc(pixel, size, pixel_definition)`](utils3d/torch/transforms.py#L470) | 
| `utils3d.pixel_to_uv`<br>Convert pixel space coordiantes to UV space coordinates. | [`utils3d.np.pixel_to_uv(pixel, size, pixel_definition)`](utils3d/numpy/transforms.py#L431) | [`utils3d.th.pixel_to_uv(pixel, size, pixel_definition)`](utils3d/torch/transforms.py#L422) | 
| `utils3d.project`<br>Calculate projection.  | [`utils3d.np.project(points, intrinsics, extrinsics, view, projection)`](utils3d/numpy/transforms.py#L694) | [`utils3d.th.project(points, intrinsics, extrinsics, view, projection)`](utils3d/torch/transforms.py#L655) | 
| `utils3d.project_cv`<br>Project 3D points to 2D following the OpenCV convention | [`utils3d.np.project_cv(points, intrinsics, extrinsics)`](utils3d/numpy/transforms.py#L578) | [`utils3d.th.project_cv(points, intrinsics, extrinsics)`](utils3d/torch/transforms.py#L564) | 
| `utils3d.project_gl`<br>Project 3D points to 2D following the OpenGL convention (except for row major matrices) | [`utils3d.np.project_gl(points, projection, view)`](utils3d/numpy/transforms.py#L549) | [`utils3d.th.project_gl(points, projection, view)`](utils3d/torch/transforms.py#L535) | 
| `utils3d.quaternion_to_axis_angle`<br> | - | [`utils3d.th.quaternion_to_axis_angle(quaternion, eps)`](utils3d/torch/transforms.py#L932) | 
| `utils3d.quaternion_to_matrix`<br>Converts a batch of quaternions (w, x, y, z) to rotation matrices | [`utils3d.np.quaternion_to_matrix(quaternion, eps)`](utils3d/numpy/transforms.py#L777) | [`utils3d.th.quaternion_to_matrix(quaternion, eps)`](utils3d/torch/transforms.py#L1009) | 
| `utils3d.ray_intersection`<br>Compute the intersection/closest point of two D-dimensional rays | [`utils3d.np.ray_intersection(p1, d1, p2, d2)`](utils3d/numpy/transforms.py#L978) | - | 
| `utils3d.rotate_2d`<br> | - | [`utils3d.th.rotate_2d(theta, center)`](utils3d/torch/transforms.py#L1151) | 
| `utils3d.rotation_matrix_2d`<br> | - | [`utils3d.th.rotation_matrix_2d(theta)`](utils3d/torch/transforms.py#L1133) | 
| `utils3d.rotation_matrix_from_vectors`<br>Rotation matrix that rotates v1 to v2 | [`utils3d.np.rotation_matrix_from_vectors(v1, v2)`](utils3d/numpy/transforms.py#L939) | [`utils3d.th.rotation_matrix_from_vectors(v1, v2)`](utils3d/torch/transforms.py#L808) | 
| `utils3d.scale_2d`<br> | - | [`utils3d.th.scale_2d(scale, center)`](utils3d/torch/transforms.py#L1205) | 
| `utils3d.screen_coord_to_view_coord`<br>Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices) | [`utils3d.np.screen_coord_to_view_coord(screen_coord, projection)`](utils3d/numpy/transforms.py#L640) | - | 
| `utils3d.skew_symmetric`<br>Skew symmetric matrix from a 3D vector | [`utils3d.np.skew_symmetric(v)`](utils3d/numpy/transforms.py#L927) | [`utils3d.th.skew_symmetric(v)`](utils3d/torch/transforms.py#L796) | 
| `utils3d.slerp`<br> | - | [`utils3d.th.slerp(rot_mat_1, rot_mat_2, t)`](utils3d/torch/transforms.py#L1034) | 
| `utils3d.slerp_quaternion`<br>Spherical linear interpolation between two unit quaternions. | [`utils3d.np.slerp_quaternion(q1, q2, t)`](utils3d/numpy/transforms.py#L1025) | - | 
| `utils3d.slerp_vector`<br>Spherical linear interpolation between two unit vectors. The vectors are assumed to be normalized. | [`utils3d.np.slerp_vector(v1, v2, t)`](utils3d/numpy/transforms.py#L1070) | - | 
| `utils3d.transform`<br>Apply affine transformation(s) to a point or a set of points. | [`utils3d.np.transform(x, Ts)`](utils3d/numpy/transforms.py#L1187) | [`utils3d.th.transform(x, Ts)`](utils3d/torch/transforms.py#L1235) | 
| `utils3d.translate_2d`<br> | - | [`utils3d.th.translate_2d(translation)`](utils3d/torch/transforms.py#L1182) | 
| `utils3d.unproject`<br>Calculate inverse projection.  | [`utils3d.np.unproject(uv, depth, intrinsics, extrinsics, projection, view)`](utils3d/numpy/transforms.py#L735) | [`utils3d.th.unproject(uv, depth, intrinsics, extrinsics, projection, view)`](utils3d/torch/transforms.py#L696) | 
| `utils3d.unproject_cv`<br>Unproject uv coordinates to 3D view space following the OpenCV convention | [`utils3d.np.unproject_cv(uv, depth, intrinsics, extrinsics)`](utils3d/numpy/transforms.py#L661) | [`utils3d.th.unproject_cv(uv, depth, intrinsics, extrinsics)`](utils3d/torch/transforms.py#L624) | 
| `utils3d.unproject_gl`<br>Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices) | [`utils3d.np.unproject_gl(uv, depth, projection, view)`](utils3d/numpy/transforms.py#L610) | [`utils3d.th.unproject_gl(uv, depth, projection, view)`](utils3d/torch/transforms.py#L594) | 
| `utils3d.uv_to_pixel`<br>Convert UV space coordinates to pixel space coordinates. | [`utils3d.np.uv_to_pixel(uv, size, pixel_definition)`](utils3d/numpy/transforms.py#L457) | [`utils3d.th.uv_to_pixel(uv, size, pixel_definition)`](utils3d/torch/transforms.py#L446) | 
| `utils3d.view_look_at`<br>Get OpenGL view matrix looking at something | [`utils3d.np.view_look_at(eye, look_at, up)`](utils3d/numpy/transforms.py#L212) | [`utils3d.th.view_look_at(eye, look_at, up)`](utils3d/torch/transforms.py#L224) | 
| `utils3d.view_to_extrinsics`<br>OpenGL view matrix to OpenCV camera extrinsics | [`utils3d.np.view_to_extrinsics(view)`](utils3d/numpy/transforms.py#L341) | [`utils3d.th.view_to_extrinsics(view)`](utils3d/torch/transforms.py#L332) | 


### Maps

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.bounding_rect_from_mask`<br> | - | [`utils3d.th.bounding_rect_from_mask(mask)`](utils3d/torch/maps.py#L366) | 
| `utils3d.build_mesh_from_depth_map`<br>Get a mesh by lifting depth map to 3D, while removing depths of large depth difference. | [`utils3d.np.build_mesh_from_depth_map(depth, other_maps, intrinsics, extrinsics, atol, rtol, tri)`](utils3d/numpy/maps.py#L185) | [`utils3d.th.build_mesh_from_depth_map(depth, other_maps, intrinsics, extrinsics, atol, rtol, tri)`](utils3d/torch/maps.py#L197) | 
| `utils3d.build_mesh_from_map`<br>Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces. | [`utils3d.np.build_mesh_from_map(maps, mask, tri)`](utils3d/numpy/maps.py#L152) | [`utils3d.th.build_mesh_from_map(maps, mask, tri)`](utils3d/torch/maps.py#L158) | 
| `utils3d.chessboard`<br>Get a chessboard image | [`utils3d.np.chessboard(size, grid_size, color_a, color_b)`](utils3d/numpy/maps.py#L423) | [`utils3d.th.chessboard(size, grid_size, color_a, color_b)`](utils3d/torch/maps.py#L383) | 
| `utils3d.depth_map_aliasing`<br>Compute the map that indicates the aliasing of x depth map, identifying pixels which neither close to the maximum nor the minimum of its neighbors. | [`utils3d.np.depth_map_aliasing(depth, atol, rtol, kernel_size, mask)`](utils3d/numpy/maps.py#L254) | [`utils3d.th.depth_map_aliasing(depth, atol, rtol, kernel_size, mask)`](utils3d/torch/maps.py#L265) | 
| `utils3d.depth_map_edge`<br>Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth. | [`utils3d.np.depth_map_edge(depth, atol, rtol, kernel_size, mask)`](utils3d/numpy/maps.py#L227) | [`utils3d.th.depth_map_edge(depth, atol, rtol, kernel_size, mask)`](utils3d/torch/maps.py#L239) | 
| `utils3d.depth_map_to_normal_map`<br>Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenCV identity camera's coordinate system. | [`utils3d.np.depth_map_to_normal_map(depth, intrinsics, mask, edge_threshold)`](utils3d/numpy/maps.py#L377) | [`utils3d.th.depth_map_to_normal_map(depth, intrinsics, mask)`](utils3d/torch/maps.py#L344) | 
| `utils3d.depth_map_to_point_map`<br>Unproject depth map to 3D points. | [`utils3d.np.depth_map_to_point_map(depth, intrinsics, extrinsics)`](utils3d/numpy/maps.py#L397) | [`utils3d.th.depth_map_to_point_map(depth, intrinsics, extrinsics)`](utils3d/torch/maps.py#L359) | 
| `utils3d.masked_area_resize`<br>Resize 2D map by area sampling with mask awareness. | [`utils3d.np.masked_area_resize(image, mask, size)`](utils3d/numpy/maps.py#L516) | [`utils3d.th.masked_area_resize(image, mask, size)`](utils3d/torch/maps.py#L477) | 
| `utils3d.masked_nearest_resize`<br>Resize image(s) by nearest sampling with mask awareness.  | [`utils3d.np.masked_nearest_resize(image, mask, size, return_index)`](utils3d/numpy/maps.py#L446) | [`utils3d.th.masked_nearest_resize(image, mask, size, return_index)`](utils3d/torch/maps.py#L406) | 
| `utils3d.normal_map_edge`<br>Compute the edge mask from normal map. | [`utils3d.np.normal_map_edge(normals, tol, kernel_size, mask)`](utils3d/numpy/maps.py#L282) | - | 
| `utils3d.pixel_coord_map`<br>Get image pixel coordinates map, where (0, 0) is the top-left corner of the top-left pixel, and (width, height) is the bottom-right corner of the bottom-right pixel. | [`utils3d.np.pixel_coord_map(size, top, left, definition, dtype)`](utils3d/numpy/maps.py#L71) | [`utils3d.th.pixel_coord_map(size, top, left, definition, dtype, device)`](utils3d/torch/maps.py#L75) | 
| `utils3d.point_map_to_normal_map`<br>Calculate normal map from point map. Value range is [-1, 1].  | [`utils3d.np.point_map_to_normal_map(point, mask, edge_threshold)`](utils3d/numpy/maps.py#L320) | [`utils3d.th.point_map_to_normal_map(point, mask)`](utils3d/torch/maps.py#L299) | 
| `utils3d.screen_coord_map`<br>Get screen space coordinate map, where (0., 0.) is the bottom-left corner of the image, and (1., 1.) is the top-right corner of the image. | [`utils3d.np.screen_coord_map(size, top, left, bottom, right, dtype)`](utils3d/numpy/maps.py#L119) | [`utils3d.th.screen_coord_map(size, top, left, bottom, right, dtype, device)`](utils3d/torch/maps.py#L124) | 
| `utils3d.uv_map`<br>Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image. | [`utils3d.np.uv_map(size, top, left, bottom, right, dtype)`](utils3d/numpy/maps.py#L30) | [`utils3d.th.uv_map(size, top, left, bottom, right, dtype, device)`](utils3d/torch/maps.py#L32) | 


### Mesh

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.calc_quad_candidates`<br>Calculate the candidate quad faces. | [`utils3d.np.calc_quad_candidates(edges, face2edge, edge2face)`](utils3d/numpy/mesh.py#L557) | - | 
| `utils3d.calc_quad_direction`<br>Calculate the direction of each candidate quad face. | [`utils3d.np.calc_quad_direction(vertices, quads)`](utils3d/numpy/mesh.py#L669) | - | 
| `utils3d.calc_quad_distortion`<br>Calculate the distortion of each candidate quad face. | [`utils3d.np.calc_quad_distortion(vertices, quads)`](utils3d/numpy/mesh.py#L618) | - | 
| `utils3d.calc_quad_smoothness`<br>Calculate the smoothness of each candidate quad face connection. | [`utils3d.np.calc_quad_smoothness(quad2edge, quad2adj, quads_direction)`](utils3d/numpy/mesh.py#L710) | - | 
| `utils3d.camera_frustum`<br>Get x triangle mesh of camera frustum. | [`utils3d.np.camera_frustum(extrinsics, intrinsics, depth)`](utils3d/numpy/mesh.py#L490) | - | 
| `utils3d.compute_boundaries`<br> | - | [`utils3d.th.compute_boundaries(faces, edges, face2edge, edge_degrees)`](utils3d/torch/mesh.py#L401) | 
| `utils3d.compute_connected_components`<br> | - | [`utils3d.th.compute_connected_components(faces, edges, face2edge)`](utils3d/torch/mesh.py#L320) | 
| `utils3d.compute_dual_graph`<br> | - | [`utils3d.th.compute_dual_graph(face2edge)`](utils3d/torch/mesh.py#L457) | 
| `utils3d.compute_edge_connected_components`<br> | - | [`utils3d.th.compute_edge_connected_components(edges)`](utils3d/torch/mesh.py#L362) | 
| `utils3d.compute_edges`<br> | - | [`utils3d.th.compute_edges(faces)`](utils3d/torch/mesh.py#L298) | 
| `utils3d.compute_face_corner_angles`<br>Compute face corner angles of a mesh | [`utils3d.np.compute_face_corner_angles(vertices, faces)`](utils3d/numpy/mesh.py#L84) | [`utils3d.th.compute_face_corner_angles(vertices, faces)`](utils3d/torch/mesh.py#L120) | 
| `utils3d.compute_face_corner_normals`<br>Compute the face corner normals of a mesh | [`utils3d.np.compute_face_corner_normals(vertices, faces, normalized)`](utils3d/numpy/mesh.py#L105) | [`utils3d.th.compute_face_corner_normals(vertices, faces, normalized)`](utils3d/torch/mesh.py#L142) | 
| `utils3d.compute_face_corner_tangents`<br>Compute the face corner tangent (and bitangent) vectors of a mesh | [`utils3d.np.compute_face_corner_tangents(vertices, uv, faces_vertices, faces_uv, normalize)`](utils3d/numpy/mesh.py#L130) | [`utils3d.th.compute_face_corner_tangents(vertices, uv, faces_vertices, faces_uv, normalize)`](utils3d/torch/mesh.py#L166) | 
| `utils3d.compute_face_normals`<br>Compute face normals of a mesh | [`utils3d.np.compute_face_normals(vertices, faces)`](utils3d/numpy/mesh.py#L166) | [`utils3d.th.compute_face_normals(vertices, faces)`](utils3d/torch/mesh.py#L202) | 
| `utils3d.compute_face_tangents`<br>Compute the face corner tangent (and bitangent) vectors of a mesh | [`utils3d.np.compute_face_tangents(vertices, uv, faces_vertices, faces_uv, normalize)`](utils3d/numpy/mesh.py#L194) | [`utils3d.th.compute_face_tangents(vertices, uv, faces_vertices, faces_uv, normalize)`](utils3d/torch/mesh.py#L230) | 
| `utils3d.compute_face_tbn`<br> | - | [`utils3d.th.compute_face_tbn(tri_vertices, tri_uvs, eps)`](utils3d/torch/mesh.py#L643) | 
| `utils3d.compute_vertex_normals`<br>Compute vertex normals of a triangular mesh by averaging neighboring face normals | [`utils3d.np.compute_vertex_normals(vertices, faces, weighted)`](utils3d/numpy/mesh.py#L230) | - | 
| `utils3d.compute_vertex_tbn`<br> | - | [`utils3d.th.compute_vertex_tbn(faces_topo, tri_vertices, tri_uvs)`](utils3d/torch/mesh.py#L664) | 
| `utils3d.cube`<br>Get x cube mesh of size 1 centered at origin. | [`utils3d.np.cube(tri)`](utils3d/numpy/mesh.py#L459) | - | 
| `utils3d.flatten_mesh_indices`<br> | [`utils3d.np.flatten_mesh_indices(args)`](utils3d/numpy/mesh.py#L428) | - | 
| `utils3d.icosahedron`<br> | [`utils3d.np.icosahedron()`](utils3d/numpy/mesh.py#L516) | - | 
| `utils3d.laplacian`<br> | - | [`utils3d.th.laplacian(vertices, faces, weight)`](utils3d/torch/mesh.py#L687) | 
| `utils3d.laplacian_hc_smooth_mesh`<br> | - | [`utils3d.th.laplacian_hc_smooth_mesh(vertices, faces, times, alpha, beta, weight)`](utils3d/torch/mesh.py#L746) | 
| `utils3d.laplacian_smooth_mesh`<br> | - | [`utils3d.th.laplacian_smooth_mesh(vertices, faces, weight, times)`](utils3d/torch/mesh.py#L716) | 
| `utils3d.merge_duplicate_vertices`<br>Merge duplicate vertices of a triangular mesh.  | [`utils3d.np.merge_duplicate_vertices(vertices, faces, tol)`](utils3d/numpy/mesh.py#L278) | [`utils3d.th.merge_duplicate_vertices(vertices, faces, tol)`](utils3d/torch/mesh.py#L521) | 
| `utils3d.merge_meshes`<br>Merge multiple meshes into one mesh. Vertices will be no longer shared. | [`utils3d.np.merge_meshes(meshes)`](utils3d/numpy/mesh.py#L532) | - | 
| `utils3d.mesh_relations`<br>Calculate the relation between vertices and faces. | [`utils3d.np.mesh_relations(faces)`](utils3d/numpy/mesh.py#L368) | - | 
| `utils3d.remove_corrupted_faces`<br>Remove corrupted faces (faces with duplicated vertices) | [`utils3d.np.remove_corrupted_faces(faces)`](utils3d/numpy/mesh.py#L264) | [`utils3d.th.remove_corrupted_faces(faces)`](utils3d/torch/mesh.py#L505) | 
| `utils3d.remove_isolated_pieces`<br> | - | [`utils3d.th.remove_isolated_pieces(vertices, faces, connected_components, thresh_num_faces, thresh_radius, thresh_boundary_ratio, remove_unreferenced)`](utils3d/torch/mesh.py#L546) | 
| `utils3d.remove_unused_vertices`<br>Remove unreferenced vertices of a mesh.  | [`utils3d.np.remove_unused_vertices(faces, vertice_attrs, return_indices)`](utils3d/numpy/mesh.py#L303) | [`utils3d.th.remove_unused_vertices(faces, vertice_attrs, return_indices)`](utils3d/torch/mesh.py#L476) | 
| `utils3d.solve_quad`<br>Solve the quad mesh from the candidate quad faces. | [`utils3d.np.solve_quad(face2edge, edge2face, quad2adj, quads_distortion, quads_smoothness, quads_valid)`](utils3d/numpy/mesh.py#L739) | - | 
| `utils3d.solve_quad_qp`<br>Solve the quad mesh from the candidate quad faces. | [`utils3d.np.solve_quad_qp(face2edge, edge2face, quad2adj, quads_distortion, quads_smoothness, quads_valid)`](utils3d/numpy/mesh.py#L852) | - | 
| `utils3d.square`<br>Get a square mesh of area 1 centered at origin in the xy-plane. | [`utils3d.np.square(tri)`](utils3d/numpy/mesh.py#L441) | - | 
| `utils3d.subdivide_mesh`<br>Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles. | [`utils3d.np.subdivide_mesh(vertices, faces, level)`](utils3d/numpy/mesh.py#L332) | [`utils3d.th.subdivide_mesh(vertices, faces, n)`](utils3d/torch/mesh.py#L612) | 
| `utils3d.taubin_smooth_mesh`<br> | - | [`utils3d.th.taubin_smooth_mesh(vertices, faces, lambda_, mu_)`](utils3d/torch/mesh.py#L729) | 
| `utils3d.tri_to_quad`<br>Convert a triangle mesh to a quad mesh. | [`utils3d.np.tri_to_quad(vertices, faces)`](utils3d/numpy/mesh.py#L938) | - | 
| `utils3d.triangulate_mesh`<br>Triangulate a polygonal mesh. | [`utils3d.np.triangulate_mesh(faces, vertices, method)`](utils3d/numpy/mesh.py#L36) | [`utils3d.th.triangulate_mesh(faces, vertices, method)`](utils3d/torch/mesh.py#L71) | 


### Rasterization

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.RastContext`<br> | [`utils3d.np.RastContext(args, kwargs)`](utils3d/numpy/rasterization.py#L51) | [`utils3d.th.RastContext(nvd_ctx, backend, device)`](utils3d/torch/rasterization.py#L21) | 
| `utils3d.rasterize_lines`<br>Rasterize lines. | [`utils3d.np.rasterize_lines(ctx, width, height, vertices, lines, attributes, attributes_domain, view, projection, line_width, return_depth, return_interpolation, background_image, background_depth, background_interpolation_id, background_interpolation_uv)`](utils3d/numpy/rasterization.py#L897) | - | 
| `utils3d.rasterize_point_cloud`<br>Rasterize point cloud. | [`utils3d.np.rasterize_point_cloud(ctx, width, height, points, point_sizes, point_size_in, point_shape, attributes, view, projection, return_depth, return_point_id, background_image, background_depth, background_point_id)`](utils3d/numpy/rasterization.py#L1117) | - | 
| `utils3d.rasterize_triangles`<br>Rasterize triangles. | [`utils3d.np.rasterize_triangles(ctx, width, height, vertices, attributes, attributes_domain, faces, view, projection, cull_backface, return_depth, return_interpolation, background_image, background_depth, background_interpolation_id, background_interpolation_uv)`](utils3d/numpy/rasterization.py#L443) | [`utils3d.th.rasterize_triangles(ctx, width, height, vertices, faces, attr, uv, texture, model, view, projection, antialiasing, diff_attrs)`](utils3d/torch/rasterization.py#L38) | 
| `utils3d.rasterize_triangles_peeling`<br>Rasterize triangles with depth peeling. | [`utils3d.np.rasterize_triangles_peeling(ctx, width, height, vertices, attributes, attributes_domain, faces, view, projection, cull_backface, return_depth, return_interpolation)`](utils3d/numpy/rasterization.py#L665) | [`utils3d.th.rasterize_triangles_peeling(ctx, vertices, faces, width, height, max_layers, attr, uv, texture, model, view, projection, antialiasing, diff_attrs)`](utils3d/torch/rasterization.py#L145) | 
| `utils3d.sample_texture`<br>Sample from a texture map with a UV map. | [`utils3d.np.sample_texture(ctx, uv_map, texture_map, interpolation, mipmap_level, repeat, anisotropic)`](utils3d/numpy/rasterization.py#L1308) | [`utils3d.th.sample_texture(texture, uv, uv_da)`](utils3d/torch/rasterization.py#L271) | 
| `utils3d.test_rasterization`<br>Test if rasterization works. It will render a cube with random colors and save it as a CHECKME.png file. | [`utils3d.np.test_rasterization(ctx)`](utils3d/numpy/rasterization.py#L1459) | - | 
| `utils3d.texture_composite`<br> | - | [`utils3d.th.texture_composite(texture, uv, uv_da, background)`](utils3d/torch/rasterization.py#L291) | 
| `utils3d.warp_image_by_depth`<br> | - | [`utils3d.th.warp_image_by_depth(ctx, depth, image, mask, width, height, extrinsics_src, extrinsics_tgt, intrinsics_src, intrinsics_tgt, near, far, antialiasing, backslash, padding, return_uv, return_dr)`](utils3d/torch/rasterization.py#L332) | 
| `utils3d.warp_image_by_forward_flow`<br> | - | [`utils3d.th.warp_image_by_forward_flow(ctx, image, flow, depth, antialiasing, backslash)`](utils3d/torch/rasterization.py#L502) | 


### Utils

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.lookup`<br>Look up `query` in `key` like a dictionary. | [`utils3d.np.lookup(key, query, value, default_value)`](utils3d/numpy/utils.py#L122) | [`utils3d.th.lookup(key, query)`](utils3d/torch/utils.py#L111) | 
| `utils3d.masked_max`<br> | - | [`utils3d.th.masked_max(input, mask, dim, keepdim)`](utils3d/torch/utils.py#L102) | 
| `utils3d.masked_min`<br> | - | [`utils3d.th.masked_min(input, mask, dim, keepdim)`](utils3d/torch/utils.py#L93) | 
| `utils3d.max_pool_1d`<br> | [`utils3d.np.max_pool_1d(x, kernel_size, stride, padding, axis)`](utils3d/numpy/utils.py#L94) | - | 
| `utils3d.max_pool_2d`<br> | [`utils3d.np.max_pool_2d(x, kernel_size, stride, padding, axis)`](utils3d/numpy/utils.py#L111) | - | 
| `utils3d.max_pool_nd`<br> | [`utils3d.np.max_pool_nd(x, kernel_size, stride, padding, axis)`](utils3d/numpy/utils.py#L105) | - | 
| `utils3d.sliding_window`<br>Get a sliding window of the input array. | [`utils3d.np.sliding_window(x, window_size, stride, pad_size, pad_mode, pad_value, axis)`](utils3d/numpy/utils.py#L17) | [`utils3d.th.sliding_window(x, window_size, stride, pad_size, pad_mode, pad_value, dim)`](utils3d/torch/utils.py#L21) | 
