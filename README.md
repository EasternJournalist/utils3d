# utils3d
Easy rasterization and useful tool functions for researchers.

It could be helpful when you want to:

* **rasterize** a simple mesh but don't want get into OpenGL chores
* **warp** an image as either a 2D or 3D mesh (eg. optical-flow-based warping)
* **project** following either OpenCV or OpenGL conventions
* render a optical **flow** image

This tool sets could help you finish them in a few lines, just two save time and keep code clean.

It is **NOT** what you are looking for when you want:

* a differentiable rasterization tool. You should turn to `nvdiffrast`, `pytorch3d`, `SoftRas`.
* a real-time graphics application. Though as fast as it could be, the expected performance of `util3d` rasterization is to be around 10 ~ 100 ms. It is not expected to fully make use of GPU performance because of the overhead of buffering every time calling rasterzation. If the best performance withou any overhead is demanded, You will have to manage buffer objects like VBO, VAO and FBO. I personally recommand `moderngl` as an alternative python OpenGL library. 


## Install

The folder of repo is a package. Clone the repo.

```bash
git clone https://github.com/EasternJournalist/utils3d.git 
```

Install requirements

```bash
pip install numpy
pip install moderngl
```

## Rasterization 

At first, one step to initialize a OpenGL context. It depends on your platform and machine.
```python
import utils3d

ctx = utils3d.gl.Context(standalone=True)                   # Recommanded for a standalone python program. The machine must have a display device (virtual display like X11 is also okay)
ctx = utils3d.gl.Context(standalone=False)                  # Recommanded for a nested python script running in a windowed opengl program to share the OpenGL context, eg. Blender.
ctx = utils3d.gl.Context(standalone=True, backend='egl')    # Recommanded for a program running on a headless linux server (without any display device)
```

Then a number of rasterization functions of `ctx` can be used. The functions the most probably you would like to use

* `ctx.rasterize_barycentric(...)`: rasterize trianglular mesh and get the barycentric coordinates map
* `ctx.rasterize_attribute(...)`: rasterize trianglular mesh with vertex attributes
* `ctx.rasterize_texture(...)`: rasterize trianglular mesh with texture
* `ctx.texture(uv, texture)`: sample texture by a UV image. Exactly the same as grid sample, but an OpenGL shader implementation.

Some other functions that could be helpful for certain purposes

* `ctx.render_flow(...)`: render an optical flow image given source and target geometry
* `ctx.warp_image_3d(image, pixel_positions, transform_matrix)`
* `ctx.warp_image_by_flow(image, flow, occlusion_mask)`

## Useful tool functions

* Image based tool functions
    * `image_uv(...)` : return a numpy array of shape `[height, width, 2]`, the image uv of each pixel. 
    * `image_mesh(...)` : return a quad mesh connecting all neighboring pixels as vertices. A boolean array of shape `[height, width]` or  `[height, width, 1]` mask is optional. If a mask is provided, only pixels where mask value is `True` are involved in the mesh.
* Geometric tools
    * `triangulate(...)` : convert a polygonal mesh into a triangular mesh (naively).
    * `compute_face_normal(...)`
    * `compute_vertex_normal(...)` 
    * `compute_face_tbn()`
    * `laplacian(...)`: 
    * `laplacian_smooth_mesh(...)`
    * `laplacian_hc_smooth()`
* OpenGL convention projection
    * `perspective_from_fov(...)`
    * `perspective_from_fov_xy(...)`
    * `projection(...)`: project 3D points to 2D screen space following the OpenGL convention (except for using row major matrix). 
    * `inverse_projection(...)`: reconstruct 3D position in view space given screen space coordinates and linear depth. to be updated (2020-04-29)
* OpenCV convention projection
    * `intrinsic_from_fov(...)`
    * `intrinsic_from_fov_xy(...)`
    * `projection_cv(...)`
    * `inverse_projection_cv(...)`: to be updated (2020-04-29)
* OpenGL↔OpenCV convention camera conversion
    * `extrinsic_to_view(...)`
    * `view_to_extrinsic(...)`
    * `intrinsic_to_perspective(...)`
    * `perspective_to_extrinsic(...)`
    * `camera_cv_to_gl(...)`
    * `camera_gl_to_cv(...)`

