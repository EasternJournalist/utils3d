# utils3d

<img src="doc/teaser.jpg" width="100%">

A collection of useful functions for 3D computer vision and graphics researchers in Python.

- **NumPy / PyTorch pairs**: most functions have both implementations.
- **Flat & non-modular**: standalone functions only, no classes, no hierarchies.
- **Native types**: always use raw Python / NumPy / PyTorch / SciPy.sparse types.
- **Vectorized only**: no Python loops beyond O(log N).

> ⚠️ *This repo changes quickly*. *Functions may be added, removed, or modified at any time.*
> - Copy code if you only need a single function.
> - Use commit id or fork if you need stability.

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

## Documentation

- Use `utils3d.{function}` to call the function automatically selecting the backend based on the input type (Numpy ndarray or Pytorch tensor).
- Use `utils3d.{np/pt}.{function}` to specifically call the Numpy or Pytorch version.

The links below will take you to the source code of each function with detailed documentation and type hints.

### Transforms

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.angle_between`<br>Calculate the angle between two (batches of) vectors. | [`utils3d.np.angle_between(v1, v2)`](utils3d/numpy/transforms.py#L1429) | [`utils3d.pt.angle_between(v1, v2, eps)`](utils3d/torch/transforms.py#L1399) | 
| `utils3d.axis_angle_to_matrix`<br>Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation | [`utils3d.np.axis_angle_to_matrix(axis_angle)`](utils3d/numpy/transforms.py#L1033) | [`utils3d.pt.axis_angle_to_matrix(axis_angle, eps)`](utils3d/torch/transforms.py#L936) | 
| `utils3d.axis_angle_to_quaternion`<br>Convert axis-angle representation (rotation vector) to quaternion (w, x, y, z) | [`utils3d.np.axis_angle_to_quaternion(axis_angle)`](utils3d/numpy/transforms.py#L1060) | [`utils3d.pt.axis_angle_to_quaternion(axis_angle, eps)`](utils3d/torch/transforms.py#L993) | 
| `utils3d.crop_intrinsics`<br>Evaluate the new intrinsics after cropping the image | [`utils3d.np.crop_intrinsics(intrinsics, size, cropped_top, cropped_left, cropped_height, cropped_width)`](utils3d/numpy/transforms.py#L445) | [`utils3d.pt.crop_intrinsics(intrinsics, size, cropped_top, cropped_left, cropped_height, cropped_width)`](utils3d/torch/transforms.py#L431) | 
| `utils3d.denormalize_intrinsics`<br>Denormalize intrinsics from uv cooridnates to pixel coordinates | [`utils3d.np.denormalize_intrinsics(intrinsics, size, pixel_convention)`](utils3d/numpy/transforms.py#L405) | [`utils3d.pt.denormalize_intrinsics(intrinsics, size, pixel_convention)`](utils3d/torch/transforms.py#L390) | 
| `utils3d.depth_buffer_to_linear`<br>OpenGL depth buffer to linear depth | [`utils3d.np.depth_buffer_to_linear(depth_buffer, near, far)`](utils3d/numpy/transforms.py#L581) | [`utils3d.pt.depth_buffer_to_linear(depth, near, far)`](utils3d/torch/transforms.py#L561) | 
| `utils3d.depth_linear_to_buffer`<br>Project linear depth to depth value in screen space | [`utils3d.np.depth_linear_to_buffer(depth, near, far)`](utils3d/numpy/transforms.py#L561) | [`utils3d.pt.depth_linear_to_buffer(depth, near, far)`](utils3d/torch/transforms.py#L542) | 
| `utils3d.euler_angles_to_matrix`<br>Convert rotations given as Euler angles in radians to rotation matrices. | [`utils3d.np.euler_angles_to_matrix(euler_angles, convention)`](utils3d/numpy/transforms.py#L982) | [`utils3d.pt.euler_angles_to_matrix(euler_angles, convention)`](utils3d/torch/transforms.py#L813) | 
| `utils3d.euler_axis_angle_rotation`<br>Return the rotation matrices for one of the rotations about an axis | [`utils3d.np.euler_axis_angle_rotation(axis, angle)`](utils3d/numpy/transforms.py#L952) | [`utils3d.pt.euler_axis_angle_rotation(axis, angle)`](utils3d/torch/transforms.py#L783) | 
| `utils3d.extrinsics_look_at`<br>Get OpenCV extrinsics matrix looking at something | [`utils3d.np.extrinsics_look_at(eye, look_at, up)`](utils3d/numpy/transforms.py#L250) | [`utils3d.pt.extrinsics_look_at(eye, look_at, up)`](utils3d/torch/transforms.py#L255) | 
| `utils3d.extrinsics_to_essential`<br>extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0` | [`utils3d.np.extrinsics_to_essential(extrinsics)`](utils3d/numpy/transforms.py#L930) | [`utils3d.pt.extrinsics_to_essential(extrinsics)`](utils3d/torch/transforms.py#L1180) | 
| `utils3d.extrinsics_to_view`<br>OpenCV camera extrinsics to OpenGL view matrix | [`utils3d.np.extrinsics_to_view(extrinsics)`](utils3d/numpy/transforms.py#L338) | [`utils3d.pt.extrinsics_to_view(extrinsics)`](utils3d/torch/transforms.py#L323) | 
| `utils3d.focal_to_fov`<br> | [`utils3d.np.focal_to_fov(focal)`](utils3d/numpy/transforms.py#L209) | [`utils3d.pt.focal_to_fov(focal)`](utils3d/torch/transforms.py#L213) | 
| `utils3d.fov_to_focal`<br> | [`utils3d.np.fov_to_focal(fov)`](utils3d/numpy/transforms.py#L213) | [`utils3d.pt.fov_to_focal(fov)`](utils3d/torch/transforms.py#L217) | 
| `utils3d.interpolate_se3_matrix`<br>Linear interpolation between two SE(3) matrices. | [`utils3d.np.interpolate_se3_matrix(T1, T2, t)`](utils3d/numpy/transforms.py#L1273) | [`utils3d.pt.interpolate_se3_matrix(T1, T2, t)`](utils3d/torch/transforms.py#L1158) | 
| `utils3d.intrinsics_from_focal_center`<br>Get OpenCV intrinsics matrix | [`utils3d.np.intrinsics_from_focal_center(fx, fy, cx, cy)`](utils3d/numpy/transforms.py#L139) | [`utils3d.pt.intrinsics_from_focal_center(fx, fy, cx, cy)`](utils3d/torch/transforms.py#L139) | 
| `utils3d.intrinsics_from_fov`<br>Get normalized OpenCV intrinsics matrix from given field of view. | [`utils3d.np.intrinsics_from_fov(fov_x, fov_y, fov_max, fov_min, aspect_ratio)`](utils3d/numpy/transforms.py#L165) | [`utils3d.pt.intrinsics_from_fov(fov_x, fov_y, fov_max, fov_min, aspect_ratio)`](utils3d/torch/transforms.py#L169) | 
| `utils3d.intrinsics_to_fov`<br>NOTE: approximate FOV by assuming centered principal point | [`utils3d.np.intrinsics_to_fov(intrinsics)`](utils3d/numpy/transforms.py#L217) | [`utils3d.pt.intrinsics_to_fov(intrinsics)`](utils3d/torch/transforms.py#L221) | 
| `utils3d.intrinsics_to_perspective`<br>OpenCV intrinsics to OpenGL perspective matrix | [`utils3d.np.intrinsics_to_perspective(intrinsics, near, far)`](utils3d/numpy/transforms.py#L311) | [`utils3d.pt.intrinsics_to_perspective(intrinsics, near, far)`](utils3d/torch/transforms.py#L294) | 
| `utils3d.lerp`<br>Linear interpolation between two vectors. | [`utils3d.np.lerp(x1, x2, t)`](utils3d/numpy/transforms.py#L1209) | [`utils3d.pt.lerp(v1, v2, t)`](utils3d/torch/transforms.py#L1095) | 
| `utils3d.make_affine_matrix`<br>Make an affine transformation matrix from a linear matrix and a translation vector. | [`utils3d.np.make_affine_matrix(M, t)`](utils3d/numpy/transforms.py#L1190) | [`utils3d.pt.make_affine_matrix(M, t)`](utils3d/torch/transforms.py#L1202) | 
| `utils3d.matrix_to_axis_angle`<br>Convert a batch of 3x3 rotation matrices to axis-angle representation (rotation vector) | [`utils3d.np.matrix_to_axis_angle(rot_mat)`](utils3d/numpy/transforms.py#L916) | [`utils3d.pt.matrix_to_axis_angle(rot_mat, eps)`](utils3d/torch/transforms.py#L963) | 
| `utils3d.matrix_to_euler_angles`<br>Convert rotations given as rotation matrices to Euler angles in radians. | [`utils3d.np.matrix_to_euler_angles(matrix, convention)`](utils3d/numpy/transforms.py#L1108) | [`utils3d.pt.matrix_to_euler_angles(matrix, convention)`](utils3d/torch/transforms.py#L898) | 
| `utils3d.matrix_to_quaternion`<br>Convert 3x3 rotation matrix to quaternion (w, x, y, z) | [`utils3d.np.matrix_to_quaternion(rot_mat)`](utils3d/numpy/transforms.py#L854) | [`utils3d.pt.matrix_to_quaternion(rot_mat, eps)`](utils3d/torch/transforms.py#L1008) | 
| `utils3d.normalize_intrinsics`<br>Normalize intrinsics from pixel cooridnates to uv coordinates | [`utils3d.np.normalize_intrinsics(intrinsics, size, pixel_convention)`](utils3d/numpy/transforms.py#L364) | [`utils3d.pt.normalize_intrinsics(intrinsics, size, pixel_convention)`](utils3d/torch/transforms.py#L349) | 
| `utils3d.perspective_from_fov`<br>Get OpenGL perspective matrix from field of view  | [`utils3d.np.perspective_from_fov(fov_x, fov_y, fov_min, fov_max, aspect_ratio, near, far)`](utils3d/numpy/transforms.py#L70) | [`utils3d.pt.perspective_from_fov(fov_x, fov_y, fov_min, fov_max, aspect_ratio, near, far)`](utils3d/torch/transforms.py#L68) | 
| `utils3d.perspective_from_window`<br>Get OpenGL perspective matrix from the window of z=-1 projection plane | [`utils3d.np.perspective_from_window(left, right, bottom, top, near, far)`](utils3d/numpy/transforms.py#L112) | [`utils3d.pt.perspective_from_window(left, right, bottom, top, near, far)`](utils3d/torch/transforms.py#L112) | 
| `utils3d.perspective_to_intrinsics`<br>OpenGL perspective matrix to OpenCV intrinsics | [`utils3d.np.perspective_to_intrinsics(perspective)`](utils3d/numpy/transforms.py#L283) | [`utils3d.pt.perspective_to_intrinsics(perspective)`](utils3d/torch/transforms.py#L276) | 
| `utils3d.perspective_to_near_far`<br>Get near and far planes from OpenGL perspective matrix | [`utils3d.np.perspective_to_near_far(perspective)`](utils3d/numpy/transforms.py#L300) | - | 
| `utils3d.piecewise_interpolate_se3_matrix`<br>Linear spline interpolation for SE(3) matrices. | [`utils3d.np.piecewise_interpolate_se3_matrix(T, t, s, extrapolation_mode)`](utils3d/numpy/transforms.py#L1322) | - | 
| `utils3d.piecewise_lerp`<br>Linear spline interpolation. | [`utils3d.np.piecewise_lerp(x, t, s, extrapolation_mode)`](utils3d/numpy/transforms.py#L1293) | - | 
| `utils3d.pixel_to_ndc`<br>Convert pixel coordinates to NDC (Normalized Device Coordinates). | [`utils3d.np.pixel_to_ndc(pixel, size, pixel_convention)`](utils3d/numpy/transforms.py#L532) | [`utils3d.pt.pixel_to_ndc(pixel, size, pixel_convention)`](utils3d/torch/transforms.py#L515) | 
| `utils3d.pixel_to_uv`<br>Convert pixel space coordiantes to UV space coordinates. | [`utils3d.np.pixel_to_uv(pixel, size, pixel_convention)`](utils3d/numpy/transforms.py#L481) | [`utils3d.pt.pixel_to_uv(pixel, size, pixel_convention)`](utils3d/torch/transforms.py#L467) | 
| `utils3d.procrustes`<br>Procrustes analysis to solve for scale `s`, rotation `R` and translation `t` such that `y_i ~= s R x_i + t`. | [`utils3d.np.procrustes(cov_yx, cov_xx, cov_yy, mean_x, mean_y, niter)`](utils3d/numpy/transforms.py#L1456) | - | 
| `utils3d.project`<br>Calculate projection.  | [`utils3d.np.project(points, intrinsics, extrinsics, view, projection)`](utils3d/numpy/transforms.py#L746) | [`utils3d.pt.project(points, intrinsics, extrinsics, view, projection)`](utils3d/torch/transforms.py#L700) | 
| `utils3d.project_cv`<br>Project 3D points to 2D following the OpenCV convention | [`utils3d.np.project_cv(points, intrinsics, extrinsics)`](utils3d/numpy/transforms.py#L630) | [`utils3d.pt.project_cv(points, intrinsics, extrinsics)`](utils3d/torch/transforms.py#L609) | 
| `utils3d.project_gl`<br>Project 3D points to 2D following the OpenGL convention (except for row major matrices) | [`utils3d.np.project_gl(points, projection, view)`](utils3d/numpy/transforms.py#L601) | [`utils3d.pt.project_gl(points, projection, view)`](utils3d/torch/transforms.py#L580) | 
| `utils3d.quaternion_to_axis_angle`<br>Convert a batch of quaternions (w, x, y, z) to axis-angle representation (rotation vector) | [`utils3d.np.quaternion_to_axis_angle(quaternion)`](utils3d/numpy/transforms.py#L900) | [`utils3d.pt.quaternion_to_axis_angle(quaternion, eps)`](utils3d/torch/transforms.py#L977) | 
| `utils3d.quaternion_to_matrix`<br>Converts a batch of quaternions (w, x, y, z) to rotation matrices | [`utils3d.np.quaternion_to_matrix(quaternion)`](utils3d/numpy/transforms.py#L829) | [`utils3d.pt.quaternion_to_matrix(quaternion, eps)`](utils3d/torch/transforms.py#L1054) | 
| `utils3d.random_rotation_matrix`<br>Generate random 3D rotation matrix. | [`utils3d.np.random_rotation_matrix(size, dtype)`](utils3d/numpy/transforms.py#L1146) | [`utils3d.pt.random_rotation_matrix(size, dtype, device)`](utils3d/torch/transforms.py#L1079) | 
| `utils3d.ray_intersection`<br>Compute the intersection/closest point of two D-dimensional rays | [`utils3d.np.ray_intersection(p1, d1, p2, d2)`](utils3d/numpy/transforms.py#L1162) | - | 
| `utils3d.rotate_2d`<br>3x3 matrix for 2D rotation around a center | - | [`utils3d.pt.rotate_2d(theta, center)`](utils3d/torch/transforms.py#L1238) | 
| `utils3d.rotation_matrix_2d`<br>2x2 matrix for 2D rotation | - | [`utils3d.pt.rotation_matrix_2d(theta)`](utils3d/torch/transforms.py#L1220) | 
| `utils3d.rotation_matrix_from_vectors`<br>Rotation matrix that rotates v1 to v2 | [`utils3d.np.rotation_matrix_from_vectors(v1, v2)`](utils3d/numpy/transforms.py#L1021) | [`utils3d.pt.rotation_matrix_from_vectors(v1, v2)`](utils3d/torch/transforms.py#L853) | 
| `utils3d.scale_2d`<br>Scale matrix for 2D scaling | - | [`utils3d.pt.scale_2d(scale, center)`](utils3d/torch/transforms.py#L1292) | 
| `utils3d.screen_coord_to_view_coord`<br>Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices) | [`utils3d.np.screen_coord_to_view_coord(screen_coord, projection)`](utils3d/numpy/transforms.py#L692) | - | 
| `utils3d.skew_symmetric`<br>Skew symmetric matrix from a 3D vector | [`utils3d.np.skew_symmetric(v)`](utils3d/numpy/transforms.py#L1009) | [`utils3d.pt.skew_symmetric(v)`](utils3d/torch/transforms.py#L841) | 
| `utils3d.slerp`<br>Spherical linear interpolation between two (unit) vectors. | [`utils3d.np.slerp(v1, v2, t)`](utils3d/numpy/transforms.py#L1227) | [`utils3d.pt.slerp(v1, v2, t, eps)`](utils3d/torch/transforms.py#L1112) | 
| `utils3d.slerp_rotation_matrix`<br>Spherical linear interpolation between two rotation matrices. | [`utils3d.np.slerp_rotation_matrix(R1, R2, t)`](utils3d/numpy/transforms.py#L1253) | [`utils3d.pt.slerp_rotation_matrix(R1, R2, t)`](utils3d/torch/transforms.py#L1138) | 
| `utils3d.solve_pose`<br>Solve for the pose (transformation from p to q) given weighted point correspondences. | [`utils3d.np.solve_pose(p, q, w, offsets, mode, lam, niter)`](utils3d/numpy/transforms.py#L1554) | - | 
| `utils3d.solve_poses_sequential`<br>Given trajectories of points over time, sequentially solve for the poses (transformations from canonical to each frame) of each body at each frame. | [`utils3d.np.solve_poses_sequential(trajectories, weights, offsets, accum, min_valid_size, mode, lam, niter)`](utils3d/numpy/transforms.py#L1633) | - | 
| `utils3d.transform_points`<br>Apply transformation(s) to a point or a set of points. | [`utils3d.np.transform_points(x, Ts)`](utils3d/numpy/transforms.py#L1351) | [`utils3d.pt.transform_points(x, Ts)`](utils3d/torch/transforms.py#L1322) | 
| `utils3d.translate_2d`<br>Translation matrix for 2D translation | - | [`utils3d.pt.translate_2d(translation)`](utils3d/torch/transforms.py#L1269) | 
| `utils3d.unproject`<br>Calculate inverse projection.  | [`utils3d.np.unproject(uv, depth, intrinsics, extrinsics, projection, view)`](utils3d/numpy/transforms.py#L787) | [`utils3d.pt.unproject(uv, depth, intrinsics, extrinsics, projection, view)`](utils3d/torch/transforms.py#L741) | 
| `utils3d.unproject_cv`<br>Unproject uv coordinates to 3D view space following the OpenCV convention | [`utils3d.np.unproject_cv(uv, depth, intrinsics, extrinsics)`](utils3d/numpy/transforms.py#L713) | [`utils3d.pt.unproject_cv(uv, depth, intrinsics, extrinsics)`](utils3d/torch/transforms.py#L669) | 
| `utils3d.unproject_gl`<br>Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices) | [`utils3d.np.unproject_gl(uv, depth, projection, view)`](utils3d/numpy/transforms.py#L662) | [`utils3d.pt.unproject_gl(uv, depth, projection, view)`](utils3d/torch/transforms.py#L639) | 
| `utils3d.uv_to_pixel`<br>Convert UV space coordinates to pixel space coordinates. | [`utils3d.np.uv_to_pixel(uv, size, pixel_convention)`](utils3d/numpy/transforms.py#L507) | [`utils3d.pt.uv_to_pixel(uv, size, pixel_convention)`](utils3d/torch/transforms.py#L491) | 
| `utils3d.vector_outer`<br> | [`utils3d.np.vector_outer(x, y)`](utils3d/numpy/transforms.py#L1450) | - | 
| `utils3d.view_look_at`<br>Get OpenGL view matrix looking at something | [`utils3d.np.view_look_at(eye, look_at, up)`](utils3d/numpy/transforms.py#L223) | [`utils3d.pt.view_look_at(eye, look_at, up)`](utils3d/torch/transforms.py#L228) | 
| `utils3d.view_to_extrinsics`<br>OpenGL view matrix to OpenCV camera extrinsics | [`utils3d.np.view_to_extrinsics(view)`](utils3d/numpy/transforms.py#L351) | [`utils3d.pt.view_to_extrinsics(view)`](utils3d/torch/transforms.py#L336) | 


### Maps

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.bounding_rect_from_mask`<br>Get bounding rectangle of a mask | - | [`utils3d.pt.bounding_rect_from_mask(mask)`](utils3d/torch/maps.py#L368) | 
| `utils3d.build_mesh_from_depth_map`<br>Get a mesh by lifting depth map to 3D, while removing depths of large depth difference. | [`utils3d.np.build_mesh_from_depth_map(depth, other_maps, intrinsics, extrinsics, atol, rtol, tri)`](utils3d/numpy/maps.py#L189) | [`utils3d.pt.build_mesh_from_depth_map(depth, other_maps, intrinsics, extrinsics, atol, rtol, tri)`](utils3d/torch/maps.py#L199) | 
| `utils3d.build_mesh_from_map`<br>Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces. | [`utils3d.np.build_mesh_from_map(maps, mask, tri)`](utils3d/numpy/maps.py#L157) | [`utils3d.pt.build_mesh_from_map(maps, mask, tri)`](utils3d/torch/maps.py#L160) | 
| `utils3d.chessboard`<br>Get a chessboard image | [`utils3d.np.chessboard(size, grid_size, color_a, color_b)`](utils3d/numpy/maps.py#L445) | [`utils3d.pt.chessboard(size, grid_size, color_a, color_b)`](utils3d/torch/maps.py#L385) | 
| `utils3d.colorize_depth_map`<br>Colorize depth map for visualization. | [`utils3d.np.colorize_depth_map(depth, mask, near, far, cmap)`](utils3d/numpy/maps.py#L606) | - | 
| `utils3d.colorize_normal_map`<br>Colorize normal map for visualization. Value range is [-1, 1]. | [`utils3d.np.colorize_normal_map(normal, mask, flip_yz)`](utils3d/numpy/maps.py#L636) | - | 
| `utils3d.depth_map_aliasing`<br>Compute the map that indicates the aliasing of x depth map, identifying pixels which neither close to the maximum nor the minimum of its neighbors. | [`utils3d.np.depth_map_aliasing(depth, atol, rtol, kernel_size, mask)`](utils3d/numpy/maps.py#L276) | [`utils3d.pt.depth_map_aliasing(depth, atol, rtol, kernel_size, mask)`](utils3d/torch/maps.py#L267) | 
| `utils3d.depth_map_edge`<br>Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth. | [`utils3d.np.depth_map_edge(depth, atol, rtol, ltol, kernel_size, mask)`](utils3d/numpy/maps.py#L231) | [`utils3d.pt.depth_map_edge(depth, atol, rtol, kernel_size, mask)`](utils3d/torch/maps.py#L241) | 
| `utils3d.depth_map_to_normal_map`<br>Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenCV identity camera's coordinate system. | [`utils3d.np.depth_map_to_normal_map(depth, intrinsics, mask, edge_threshold)`](utils3d/numpy/maps.py#L399) | [`utils3d.pt.depth_map_to_normal_map(depth, intrinsics, mask)`](utils3d/torch/maps.py#L346) | 
| `utils3d.depth_map_to_point_map`<br>Unproject depth map to 3D points. | [`utils3d.np.depth_map_to_point_map(depth, intrinsics, extrinsics)`](utils3d/numpy/maps.py#L419) | [`utils3d.pt.depth_map_to_point_map(depth, intrinsics, extrinsics)`](utils3d/torch/maps.py#L361) | 
| `utils3d.masked_area_resize`<br>Resize 2D map by area sampling with mask awareness. | [`utils3d.np.masked_area_resize(image, mask, size)`](utils3d/numpy/maps.py#L538) | [`utils3d.pt.masked_area_resize(image, mask, size)`](utils3d/torch/maps.py#L479) | 
| `utils3d.masked_nearest_resize`<br>Resize image(s) by nearest sampling with mask awareness.  | [`utils3d.np.masked_nearest_resize(image, mask, size, return_index)`](utils3d/numpy/maps.py#L468) | [`utils3d.pt.masked_nearest_resize(image, mask, size, return_index)`](utils3d/torch/maps.py#L408) | 
| `utils3d.normal_map_edge`<br>Compute the edge mask from normal map. | [`utils3d.np.normal_map_edge(normals, tol, kernel_size, mask)`](utils3d/numpy/maps.py#L304) | - | 
| `utils3d.pixel_coord_map`<br>Get image pixel coordinates map, where (0, 0) is the top-left corner of the top-left pixel, and (width, height) is the bottom-right corner of the bottom-right pixel. | [`utils3d.np.pixel_coord_map(size, top, left, convention, dtype)`](utils3d/numpy/maps.py#L74) | [`utils3d.pt.pixel_coord_map(size, top, left, convention, dtype, device)`](utils3d/torch/maps.py#L75) | 
| `utils3d.point_map_to_normal_map`<br>Calculate normal map from point map. Value range is [-1, 1].  | [`utils3d.np.point_map_to_normal_map(point, mask, edge_threshold)`](utils3d/numpy/maps.py#L342) | [`utils3d.pt.point_map_to_normal_map(point, mask)`](utils3d/torch/maps.py#L301) | 
| `utils3d.screen_coord_map`<br>Get screen space coordinate map, where (0., 0.) is the bottom-left corner of the image, and (1., 1.) is the top-right corner of the image. | [`utils3d.np.screen_coord_map(size, top, left, bottom, right, dtype)`](utils3d/numpy/maps.py#L124) | [`utils3d.pt.screen_coord_map(size, top, left, bottom, right, dtype, device)`](utils3d/torch/maps.py#L126) | 
| `utils3d.uv_map`<br>Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image. | [`utils3d.np.uv_map(size, top, left, bottom, right, dtype)`](utils3d/numpy/maps.py#L33) | [`utils3d.pt.uv_map(size, top, left, bottom, right, dtype, device)`](utils3d/torch/maps.py#L33) | 


### Mesh

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.compute_boundaries`<br>Compute boundary edges of a mesh. | - | [`utils3d.pt.compute_boundaries(faces, edges, face2edge, edge_degrees)`](utils3d/torch/mesh.py#L507) | 
| `utils3d.compute_face_corner_angles`<br>Compute face corner angles of a mesh | [`utils3d.np.compute_face_corner_angles(vertices, faces)`](utils3d/numpy/mesh.py#L91) | [`utils3d.pt.compute_face_corner_angles(vertices, faces)`](utils3d/torch/mesh.py#L82) | 
| `utils3d.compute_face_corner_normals`<br>Compute the face corner normals of a mesh | [`utils3d.np.compute_face_corner_normals(vertices, faces, normalize)`](utils3d/numpy/mesh.py#L112) | [`utils3d.pt.compute_face_corner_normals(vertices, faces, normalize)`](utils3d/torch/mesh.py#L104) | 
| `utils3d.compute_face_corner_tangents`<br>Compute the face corner tangent (and bitangent) vectors of a mesh | [`utils3d.np.compute_face_corner_tangents(vertices, uv, faces_vertices, faces_uv, normalize)`](utils3d/numpy/mesh.py#L137) | [`utils3d.pt.compute_face_corner_tangents(vertices, uv, faces_vertices, faces_uv, normalize)`](utils3d/torch/mesh.py#L129) | 
| `utils3d.compute_face_normals`<br>Compute face normals of a mesh | [`utils3d.np.compute_face_normals(vertices, faces)`](utils3d/numpy/mesh.py#L173) | [`utils3d.pt.compute_face_normals(vertices, faces)`](utils3d/torch/mesh.py#L165) | 
| `utils3d.compute_face_tangents`<br>Compute the face corner tangent (and bitangent) vectors of a mesh | [`utils3d.np.compute_face_tangents(vertices, uv, faces_vertices, faces_uv, normalize)`](utils3d/numpy/mesh.py#L201) | [`utils3d.pt.compute_face_tangents(vertices, uv, faces_vertices, faces_uv, normalize)`](utils3d/torch/mesh.py#L193) | 
| `utils3d.compute_mesh_laplacian`<br>Laplacian smooth with cotangent weights | - | [`utils3d.pt.compute_mesh_laplacian(vertices, faces, weight)`](utils3d/torch/mesh.py#L752) | 
| `utils3d.compute_vertex_normals`<br>Compute vertex normals of a triangular mesh by averaging neighboring face normals | [`utils3d.np.compute_vertex_normals(vertices, faces, weighted)`](utils3d/numpy/mesh.py#L237) | - | 
| `utils3d.create_camera_frustum_mesh`<br>Create a triangle mesh of camera frustum. | [`utils3d.np.create_camera_frustum_mesh(extrinsics, intrinsics, depth)`](utils3d/numpy/mesh.py#L457) | - | 
| `utils3d.create_cube_mesh`<br>Create a cube mesh of size 1 centered at origin. | [`utils3d.np.create_cube_mesh(tri)`](utils3d/numpy/mesh.py#L426) | - | 
| `utils3d.create_icosahedron_mesh`<br>Create an icosahedron mesh of centered at origin. | [`utils3d.np.create_icosahedron_mesh()`](utils3d/numpy/mesh.py#L483) | - | 
| `utils3d.create_square_mesh`<br>Create a square mesh of area 1 centered at origin in the xy-plane. | [`utils3d.np.create_square_mesh(tri)`](utils3d/numpy/mesh.py#L408) | - | 
| `utils3d.flatten_mesh_indices`<br> | [`utils3d.np.flatten_mesh_indices(args)`](utils3d/numpy/mesh.py#L395) | - | 
| `utils3d.graph_connected_components`<br>Compute connected components of an undirected graph. | [`utils3d.np.graph_connected_components(edges, num_vertices)`](utils3d/numpy/mesh.py#L724) | [`utils3d.pt.graph_connected_components(edges, num_vertices)`](utils3d/torch/mesh.py#L458) | 
| `utils3d.laplacian_hc_smooth_mesh`<br>HC algorithm from Improved Laplacian Smoothing of Noisy Surface Meshes by J.Vollmer et al. | - | [`utils3d.pt.laplacian_hc_smooth_mesh(vertices, faces, times, alpha, beta, weight)`](utils3d/torch/mesh.py#L811) | 
| `utils3d.laplacian_smooth_mesh`<br>Laplacian smooth with cotangent weights | - | [`utils3d.pt.laplacian_smooth_mesh(vertices, faces, weight, times)`](utils3d/torch/mesh.py#L781) | 
| `utils3d.merge_duplicate_vertices`<br>Merge duplicate vertices of a triangular mesh.  | [`utils3d.np.merge_duplicate_vertices(vertices, faces, tol)`](utils3d/numpy/mesh.py#L285) | [`utils3d.pt.merge_duplicate_vertices(vertices, faces, tol)`](utils3d/torch/mesh.py#L630) | 
| `utils3d.merge_meshes`<br>Merge multiple meshes into one mesh. Vertices will be no longer shared. | [`utils3d.np.merge_meshes(meshes)`](utils3d/numpy/mesh.py#L502) | - | 
| `utils3d.mesh_adjacency_graph`<br>Get adjacency graph of a mesh. | [`utils3d.np.mesh_adjacency_graph(adjacency, faces, edges, num_vertices, self_loop)`](utils3d/numpy/mesh.py#L764) | - | 
| `utils3d.mesh_connected_components`<br>Compute connected faces of a mesh. | [`utils3d.np.mesh_connected_components(faces, num_vertices)`](utils3d/numpy/mesh.py#L698) | [`utils3d.pt.mesh_connected_components(faces, num_vertices)`](utils3d/torch/mesh.py#L435) | 
| `utils3d.mesh_dual_graph`<br>Get dual graph of a mesh. (Mesh face as dual graph's vertex, adjacency by edge sharing) | - | [`utils3d.pt.mesh_dual_graph(faces)`](utils3d/torch/mesh.py#L563) | 
| `utils3d.mesh_edges`<br>Get undirected edges of a mesh. Optionally return additional mappings. | [`utils3d.np.mesh_edges(faces, return_face2edge, return_edge2face, return_counts)`](utils3d/numpy/mesh.py#L526) | [`utils3d.pt.mesh_edges(faces, return_face2edge, return_edge2face, return_counts)`](utils3d/torch/mesh.py#L260) | 
| `utils3d.mesh_half_edges`<br>Get half edges of a mesh. Optionally return additional mappings. | [`utils3d.np.mesh_half_edges(faces, return_face2edge, return_edge2face, return_twin, return_next, return_prev, return_counts)`](utils3d/numpy/mesh.py#L597) | [`utils3d.pt.mesh_half_edges(faces, return_face2edge, return_edge2face, return_twin, return_next, return_prev, return_counts)`](utils3d/torch/mesh.py#L334) | 
| `utils3d.remove_corrupted_faces`<br>Remove corrupted faces (faces with duplicated vertices) | [`utils3d.np.remove_corrupted_faces(faces)`](utils3d/numpy/mesh.py#L271) | [`utils3d.pt.remove_corrupted_faces(faces)`](utils3d/torch/mesh.py#L614) | 
| `utils3d.remove_isolated_pieces`<br>Remove isolated pieces of a mesh.  | - | [`utils3d.pt.remove_isolated_pieces(vertices, faces, connected_components, thresh_num_faces, thresh_radius, thresh_boundary_ratio, remove_unreferenced)`](utils3d/torch/mesh.py#L655) | 
| `utils3d.remove_unused_vertices`<br>Remove unreferenced vertices of a mesh.  | [`utils3d.np.remove_unused_vertices(faces, vertice_attrs, return_indices)`](utils3d/numpy/mesh.py#L310) | [`utils3d.pt.remove_unused_vertices(faces, vertice_attrs, return_indices)`](utils3d/torch/mesh.py#L585) | 
| `utils3d.subdivide_mesh`<br>Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles. | [`utils3d.np.subdivide_mesh(vertices, faces, level)`](utils3d/numpy/mesh.py#L339) | [`utils3d.pt.subdivide_mesh(vertices, faces, n)`](utils3d/torch/mesh.py#L721) | 
| `utils3d.taubin_smooth_mesh`<br>Taubin smooth mesh | - | [`utils3d.pt.taubin_smooth_mesh(vertices, faces, lambda_, mu_)`](utils3d/torch/mesh.py#L794) | 
| `utils3d.triangulate_mesh`<br>Triangulate a polygonal mesh. | [`utils3d.np.triangulate_mesh(faces, vertices, method)`](utils3d/numpy/mesh.py#L43) | [`utils3d.pt.triangulate_mesh(faces, vertices, method)`](utils3d/torch/mesh.py#L34) | 


### Rasterization

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.RastContext`<br>Context for numpy-side rasterization. Based on moderngl. | [`utils3d.np.RastContext(args, kwargs)`](utils3d/numpy/rasterization.py#L257) | [`utils3d.pt.RastContext(nvd_ctx, backend, device)`](utils3d/torch/rasterization.py#L20) | 
| `utils3d.rasterize_lines`<br>Rasterize lines. | [`utils3d.np.rasterize_lines(size, vertices, lines, attributes, attributes_domain, view, projection, extrinsics, intrinsics, near, far, line_width, return_depth, return_interpolation, background, ctx)`](utils3d/numpy/rasterization.py#L937) | - | 
| `utils3d.rasterize_point_cloud`<br>Rasterize point cloud. | [`utils3d.np.rasterize_point_cloud(size, points, point_sizes, point_size_in, point_shape, attributes, view, projection, extrinsics, intrinsics, near, far, return_depth, return_point_id, background, ctx)`](utils3d/numpy/rasterization.py#L1187) | - | 
| `utils3d.rasterize_triangles`<br>Rasterize triangles. | [`utils3d.np.rasterize_triangles(size, vertices, attributes, attributes_domain, faces, view, projection, extrinsics, intrinsics, near, far, cull_backface, return_depth, return_interpolation, background, ctx)`](utils3d/numpy/rasterization.py#L413) | [`utils3d.pt.rasterize_triangles(size, vertices, attributes, faces, view, projection, extrinsics, intrinsics, near, far, return_image_derivatives, return_depth, return_interpolation, antialiasing, ctx)`](utils3d/torch/rasterization.py#L53) | 
| `utils3d.rasterize_triangles_peeling`<br>Rasterize triangles with depth peeling. | [`utils3d.np.rasterize_triangles_peeling(size, vertices, attributes, attributes_domain, faces, view, projection, extrinsics, intrinsics, near, far, cull_backface, return_depth, return_interpolation, ctx)`](utils3d/numpy/rasterization.py#L666) | [`utils3d.pt.rasterize_triangles_peeling(size, vertices, attributes, faces, view, projection, extrinsics, intrinsics, near, far, return_image_derivatives, return_depth, return_interpolation, antialiasing, ctx)`](utils3d/torch/rasterization.py#L189) | 
| `utils3d.sample_texture`<br>Sample from a texture map with a UV map. | [`utils3d.np.sample_texture(uv_map, texture_map, interpolation, mipmap_level, repeat, anisotropic, ctx)`](utils3d/numpy/rasterization.py#L1414) | [`utils3d.pt.sample_texture(texture, uv, uv_dr)`](utils3d/torch/rasterization.py#L347) | 
| `utils3d.test_rasterization`<br>Test if rasterization works. It will render a cube with random colors and save it as a CHECKME.png file. | [`utils3d.np.test_rasterization(ctx)`](utils3d/numpy/rasterization.py#L1579) | - | 
| `utils3d.texture_composite`<br>Composite textures with depth peeling output. | - | [`utils3d.pt.texture_composite(texture, uv, uv_da, background)`](utils3d/torch/rasterization.py#L367) | 


### Utils

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.csr_eliminate_zeros`<br>Remove zero elements from a sparse CSR tensor. | - | [`utils3d.pt.csr_eliminate_zeros(input)`](utils3d/torch/utils.py#L231) | 
| `utils3d.csr_matrix_from_dense_indices`<br>Convert a regular indices array to a sparse CSR adjacency matrix format | [`utils3d.np.csr_matrix_from_dense_indices(indices, n_cols)`](utils3d/numpy/utils.py#L325) | [`utils3d.pt.csr_matrix_from_dense_indices(indices, n_cols)`](utils3d/torch/utils.py#L212) | 
| `utils3d.group`<br>Split the data into groups based on the provided labels. | [`utils3d.np.group(labels, data)`](utils3d/numpy/utils.py#L342) | [`utils3d.pt.group(labels, data)`](utils3d/torch/utils.py#L266) | 
| `utils3d.group_as_segments`<br>Group as segments by labels | [`utils3d.np.group_as_segments(labels, data)`](utils3d/numpy/utils.py#L365) | [`utils3d.pt.group_as_segments(labels, data)`](utils3d/torch/utils.py#L288) | 
| `utils3d.lookup`<br>Look up `query` in `key` like a dictionary. Useful for COO indexing. | [`utils3d.np.lookup(key, query)`](utils3d/numpy/utils.py#L200) | [`utils3d.pt.lookup(key, query)`](utils3d/torch/utils.py#L119) | 
| `utils3d.lookup_get`<br>Dictionary-like get for arrays | [`utils3d.np.lookup_get(key, value, get_key, default_value)`](utils3d/numpy/utils.py#L223) | [`utils3d.pt.lookup_get(key, value, get_key, default_value)`](utils3d/torch/utils.py#L143) | 
| `utils3d.lookup_set`<br>Dictionary-like set for arrays. | [`utils3d.np.lookup_set(key, value, set_key, set_value, append, inplace)`](utils3d/numpy/utils.py#L242) | [`utils3d.pt.lookup_set(key, value, set_key, set_value, append, inplace)`](utils3d/torch/utils.py#L162) | 
| `utils3d.masked_max`<br>Similar to torch.max, but with mask | - | [`utils3d.pt.masked_max(input, mask, dim, keepdim)`](utils3d/torch/utils.py#L110) | 
| `utils3d.masked_min`<br>Similar to torch.min, but with mask | - | [`utils3d.pt.masked_min(input, mask, dim, keepdim)`](utils3d/torch/utils.py#L101) | 
| `utils3d.max_pool_2d`<br> | [`utils3d.np.max_pool_2d(x, kernel_size, stride, padding, axis)`](utils3d/numpy/utils.py#L189) | - | 
| `utils3d.pooling`<br>Compute the pooling of the input array.  | [`utils3d.np.pooling(x, kernel_size, stride, padding, axis, mode)`](utils3d/numpy/utils.py#L104) | - | 
| `utils3d.segment_concatenate`<br>Concatenate a list of segmented arrays into a single segmented array | [`utils3d.np.segment_concatenate(segments)`](utils3d/numpy/utils.py#L301) | - | 
| `utils3d.segment_roll`<br>Roll the data within each segment. | [`utils3d.np.segment_roll(data, offsets, shift)`](utils3d/numpy/utils.py#L271) | [`utils3d.pt.segment_roll(data, offsets, shift)`](utils3d/torch/utils.py#L191) | 
| `utils3d.segment_take`<br>Take some segments from a segmented array | [`utils3d.np.segment_take(data, offsets, taking)`](utils3d/numpy/utils.py#L281) | [`utils3d.pt.segment_take(data, offsets, taking)`](utils3d/torch/utils.py#L201) | 
| `utils3d.sliding_window`<br>Get a sliding window of the input array. Window axis(axes) will be appended as the last dimension(s). | [`utils3d.np.sliding_window(x, window_size, stride, pad_size, pad_mode, pad_value, axis)`](utils3d/numpy/utils.py#L27) | [`utils3d.pt.sliding_window(x, window_size, stride, pad_size, pad_mode, pad_value, dim)`](utils3d/torch/utils.py#L29) | 


### Io

| Function | Numpy | Pytorch |
| ---- | ---- | ---- |
| `utils3d.read_extrinsics_from_colmap`<br>Read extrinsics from colmap `images.txt` file.  | [`utils3d.np.read_extrinsics_from_colmap(file)`](utils3d/numpy/io/colmap.py#L65) | - | 
| `utils3d.read_intrinsics_from_colmap`<br>Read intrinsics from colmap `cameras.txt` file. | [`utils3d.np.read_intrinsics_from_colmap(file, normalize)`](utils3d/numpy/io/colmap.py#L103) | - | 
| `utils3d.read_obj`<br>Read wavefront .obj file, without preprocessing. | [`utils3d.np.read_obj(file, encoding, ignore_unknown)`](utils3d/numpy/io/obj.py#L12) | - | 
| `utils3d.write_extrinsics_as_colmap`<br>Write extrinsics to colmap `images.txt` file. | [`utils3d.np.write_extrinsics_as_colmap(file, extrinsics, image_names, camera_ids)`](utils3d/numpy/io/colmap.py#L11) | - | 
| `utils3d.write_intrinsics_as_colmap`<br>Write intrinsics to colmap `cameras.txt` file. Currently only support PINHOLE model (no distortion) | [`utils3d.np.write_intrinsics_as_colmap(file, intrinsics, width, height, normalized)`](utils3d/numpy/io/colmap.py#L43) | - | 
| `utils3d.write_obj`<br> | [`utils3d.np.write_obj(file, obj, encoding)`](utils3d/numpy/io/obj.py#L112) | - | 
| `utils3d.write_simple_obj`<br>Write wavefront .obj file, without preprocessing. | [`utils3d.np.write_simple_obj(file, vertices, faces, encoding)`](utils3d/numpy/io/obj.py#L127) | - | 
