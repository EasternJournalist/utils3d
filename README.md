# utils3d
Easy rasterization and useful tool functions for researchers.

Some functionalities included:

* **rasterize** with OpenGL but in one single line
* **warp** an image as either a 2D or 3D mesh (eg. optical-flow-based warping)
* **project** following either OpenCV or OpenGL conventions
* Some mesh processing (e.g. computing vertex normal, computing laplacian smooth), which can be differentiable with pytorch.
* Convert OpenCV extrinsics and intrinsics from or to OpenGL view matrix and projection matrix

## Install

```bash
pip install -e https://github.com/EasternJournalist/utils3d.git 
```

## Rasterization 

At first, one step to initialize a OpenGL context. It depends on your platform and machine.
```python
import utils3d

ctx = utils3d.GLContext(standalone=True)                   # Recommanded for a standalone python program. The machine must have a display device (virtual display like X11 is also okay)
ctx = utils3d.GLContext(standalone=False)                  # Recommanded for a nested python script running in a windowed opengl program to share the OpenGL context, eg. Blender.
ctx = utils3d.GLContext(standalone=True, backend='egl')    # Recommanded for a program running on a headless linux server (without any display device)
```

A number of rasterization functions of `ctx` can be used. The functions the most probably you would like to use

* `ctx.rasterize_barycentric(...)`: rasterize trianglular mesh and get the barycentric coordinates map
* `ctx.rasterize_attribute(...)`: rasterize trianglular mesh with vertex attributes
* `ctx.rasterize_texture(...)`: rasterize trianglular mesh with texture
* `ctx.texture(uv, texture)`: sample texture by a UV image. Exactly the same as grid sample, but an OpenGL shader implementation.

Some other functions that could be helpful for certain purposes

* `ctx.rasterize_flow(...)`: render an optical flow image given source and target geometry
* `ctx.warp_image_3d(image, pixel_positions, transform_matrix)`
* `ctx.warp_image_by_flow(image, flow, occlusion_mask)`

## Useful tool functions

Most of the functions have both Numpy and Pytorch implementations, but not all of them are differentiable. Please check the table below. For column Pytorch, "yes/diff" means that the function is differentiable; "yes/indiff" means that it is indifferentiable; "yes/-" means it is naturally indifferentiable.

|  function             | Numpy     | Pytorch   |
|  ----                 | ----      | ----      | 
| triangulate           | yes       | yes/-     | 
| compute_face_normal   | yes       | yes/diff  |
| compute_vertex_normal | yes       | yes/diff  | 
| compute_face_tbn      | yes       | yes/diff  | 
| compute_vertex_tbn    | yes       | yes/diff  |
| compute_vertex_tbn    | yes       | yes/diff  |
| laplacian             | yes       | yes/diff  |
| laplacian_smooth_mesh | yes       | yes/diff  |
| taubin_smooth_mesh    | yes       | yes/diff  |
| laplacian_hc_smooth_mesh | yes    | yes/diff  |
| _axis_angle_rotation  | yes       | yes/diff  |
| euler_angles_to_matrix| yes       | yes/diff  |
| rodrigues             | yes       | yes/diff  |
| perspective_from_fov  | yes       | yes/indiff|
| perspective_from_fov_yx| yes      | yes/indiff|
| perspective_to_intrinsic| yes     | yes/diff  |
| intrinsic_to_perspective| yes     | yes/diff  |
| view_to_extrinsic     | yes       | yes/diff  |
| extrinsic_to_view     | yes       | yes/diff  |
| camera_cv_to_gl       | yes       | yes/diff  |
| camera_gl_to_cv       | yes       | yes/diff  |
| normalize_intrinsic   | yes       | yes/diff  |
| crop_intrinsic        | yes       | yes/diff  |
| projection            | yes       | yes/diff  |
| projection_ndc        | yes       | yes/diff  |
| to_linear_depth       | yes       | yes/diff  |
| to_screen_depth       | yes       | yes/diff  |
| view_look_at          | yes       | yes/diff  | 
| image_uv              | yes       | yes/-     |
| image_mesh            | yes       | yes/-     |
| chess_board           | yes       | yes/-     | 