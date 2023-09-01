import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import imageio

def run():
    depth = np.ones((128, 128), dtype=np.float32) * 2
    depth[32:96, 32:96] = 1
    perspective = utils3d.numpy.transforms.perspective(1.0, 1.0, 0.1, 10).astype(np.float32)
    view_src = utils3d.numpy.transforms.view_look_at([0, 0, 1], [0, 0, 0], [0, 1, 0]).astype(np.float32)
    view_tgt = utils3d.numpy.transforms.view_look_at([1, 0, 1], [0, 0, 0], [0, 1, 0]).astype(np.float32)
    ctx = utils3d.rastctx.GLContext(
        standalone=True,
        backend='egl',
        device_index=0,
    )
    uv = utils3d.numpy.rasterization.warp_image_by_depth(
        ctx,
        depth,
        perspective=perspective,
        view_src=view_src,
        view_tgt=view_tgt,
    )
    uv = (np.concatenate([uv, np.zeros((128, 128, 1), dtype=np.float32)], axis=-1) * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(os.path.dirname(__file__), '..', '..', 'results_to_check', 'warp_image_uv.png'), uv)

if __name__ == '__main__':
    run()
