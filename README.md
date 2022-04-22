# utils3d
 Rasterize and do image-based 3D transforms with the least efforts for researchers. Code fast and run fast. 
## Install

Clone the repo.

```bash
git clone https://github.com/EasternJournalist/utils3d.git 
```

Install requirements

```bash
pip install numpy
pip install moderngl
```

The folder of repo is a package. 

## Usage
At first, one step to initialize a OpenGL context. It depends on your platform and machine.
```python
import utils3d

ctx = utils3d.Context()                                 # Recommanded for a nested python script running in a windowed opengl program to share the OpenGL context, eg. Blender.
ctx = utils3d.Context(standalone=True)                  # Recommanded for a standalone python program. The machine must have a display device (virtual display like X11 is also okay)
ctx = utils3d.Context(standalone=True, backend='egl')   # Recommanded for a program running on a headless linux server (without any display device)
```
The two functions the most probably you would like to use

* `ctx.rasterize(...)`: rasterize trianglular mesh with vertex attributes.
* `ctx.texture(uv, texture)`: sample texture by a UV image. Exactly the same as grid sample, but an OpenGL shader implementation.
* `ctx.rasterize_texture(...)`: rasterize trianglular mesh with texture

Some other functions that could be helpful for certain purposes

* `ctx.render_flow(...)`: render an optical flow image given source and target geometry
* `ctx.warp_image_3d(image, pixel_positions, transform_matrix)`
* `ctx.warp_image_by_flow(image, flow, occlusion_mask)`

## Useful tool functions

* `image_uv(width, height)` : return a numpy array of shape `[height, width, 2]`, the image uv of each pixel. 
* `image_mesh(width, height, mask=None)` : return a quad mesh connecting all neighboring pixels as vertices. A boolean array of shape `[height, width]` or  `[height, width, 1]` mask is optional. If a mask is provided, only pixels where mask value is `True` are involved in te mesh.
* `triangulate(indices)` : convert a polygonal mesh into a triangular mesh (naively).
* `perspective_from_image()`
* `perspective_from_fov_xy()`
* `projection()`: project 3D points to 2D following the OpenGL convention (except for using row major matrix)