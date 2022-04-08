# utils3d
 Rasterize and do image-based 3D transforms with the least efforts for researchers. Code fast and run fast. 
## Requirements
* numpy ```pip install numpy```
* moderngl ``` pip install moderngl ```
## How to rasterize
At first, initialize a OpenGL context, for either window display or a standalone context without any display device. (TODO in future: default context)

```
ctx = rasterizer.Context()
```
The two functions the most probably you would like to use

* `ctx.rasterize(...)`: rasterize triangle vertex attributes.
  
* `ctx.texture(uv, texture)`: sample texture by a UV image. Exactly the same as grid sample, but an OpenGL shader implementation.

Some other functions that could be helpful for certain purposes

* `ctx.render_flow(...)`: render an optical flow image given source and target geometry.

* `ctx.warp_image_3d(image, pixel_positions, transform_matrix)`
  
* `ctx.warp_image_by_flow(image, flow, occlusion_mask)`

## Useful tool functions

* `image_uv(width, height)` : return a numpy array of shape `[height, width, 2]`, the image uv of each pixel. 

* `image_mesh(width, height, mask=None)` : return a quad mesh connecting all neighboring pixels as vertices. A boolean array of shape `[height, width]` or  `[height, width, 1]` mask is optional. If a mask is provided, only pixels where mask value is `True` are involved in te mesh.

* `triangulate(indices)` : convert a polygonal mesh into a triangular mesh (naively).

* `perspective_from_image()`

* `perspective_from_fov_xy()`