import math
import unittest

import numpy as np
import torch
import utils3d



class TestScreenGridPixelCenters(unittest.TestCase):
    def test_default_bounds_and_custom_affine_viewports(self):
        for size in [(3, 5), (1, 1), (1, 4), (5, 1)]:
            for bounds in [(1, 0, 0, 1), (7, -2, -3, 5), (-3, 5, 7, -2)]:
                top, left, bottom, right = bounds
                height, width = size
                x = left + (np.arange(width, dtype=np.float64) + 0.5) * (right - left) / width
                y = top + (np.arange(height, dtype=np.float64) + 0.5) * (bottom - top) / height
                expected = np.stack(np.meshgrid(x, y, indexing='xy'), axis=-1)
                for backend in [utils3d.np, utils3d.pt]:
                    with self.subTest(size=size, bounds=bounds, backend=backend.__name__):
                        dtype = np.float64 if backend is utils3d.np else torch.float64
                        actual = backend.screen_coord_map(size, top=top, left=left, bottom=bottom, right=right, dtype=dtype)
                        actual = actual if isinstance(actual, np.ndarray) else actual.numpy()
                        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    def test_vertical_flip_matches_uv_pixel_centers(self):
        expected = utils3d.np.uv_map(3, 5).copy()
        expected[..., 1] = 1 - expected[..., 1]
        np.testing.assert_allclose(utils3d.np.screen_coord_map(3, 5), expected, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
