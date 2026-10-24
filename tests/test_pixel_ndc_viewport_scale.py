import unittest

import numpy as np
import torch
import utils3d


def as_numpy(value):
    return value if isinstance(value, np.ndarray) else value.detach().numpy()



class TestPixelNdcViewportScale(unittest.TestCase):
    def test_viewport_edges_and_pixel_centers(self):
        for height, width in [(5, 9), (1, 7), (6, 1), (1, 1)]:
            for convention in ['integer-corner', 'integer-center']:
                for dtype in [np.float32, np.float64]:
                    shift = 0.5 if convention == 'integer-center' else 0.0
                    pixels = np.array([[0.0, 0.0], [width / 2 - shift, height / 2 - shift],
                                       [width - shift, height - shift]], dtype=dtype)
                    expected = np.array([[2 * (x + shift) / width - 1,
                                          1 - 2 * (y + shift) / height] for x, y in pixels])
                    for backend in [utils3d.np, utils3d.pt]:
                        with self.subTest(size=(height, width), convention=convention, dtype=dtype, backend=backend.__name__):
                            args = pixels if backend is utils3d.np else torch.from_numpy(pixels)
                            actual = backend.pixel_to_ndc(args, (height, width), pixel_convention=convention)
                            np.testing.assert_allclose(as_numpy(actual), expected, rtol=2e-7, atol=2e-7)
                            np.testing.assert_allclose(as_numpy(actual)[1], [0, 0], atol=1e-15)

    def test_distinct_batched_viewports_and_uv_convention(self):
        sizes = np.array([[[5, 9]], [[8, 4]]], dtype=np.float64)
        pixels = np.array([[[0.0, 0.0], [4.0, 2.0], [8.0, 4.0]],
                           [[0.0, 0.0], [1.5, 3.5], [3.0, 7.0]]], dtype=np.float64)
        expected = np.array([[[2 * (x + .5) / width - 1, 1 - 2 * (y + .5) / height]
                              for x, y in group] for group, (height, width) in zip(pixels, sizes[:, 0])])
        for backend in [utils3d.np, utils3d.pt]:
            with self.subTest(backend=backend.__name__):
                args = (pixels, sizes) if backend is utils3d.np else tuple(torch.from_numpy(x) for x in [pixels, sizes])
                actual = backend.pixel_to_ndc(*args)
                np.testing.assert_allclose(as_numpy(actual), expected, rtol=1e-14, atol=1e-14)
                uv = backend.pixel_to_uv(*args)
                np.testing.assert_allclose(as_numpy(actual), as_numpy(uv) * [2, -2] + [-1, 1], atol=1e-14)
                empty = pixels[:, :0]
                empty = empty if backend is utils3d.np else torch.from_numpy(empty)
                self.assertEqual(tuple(backend.pixel_to_ndc(empty, args[1]).shape), (2, 0, 2))

    def test_torch_pixel_jacobian_is_the_viewport_affine_map(self):
        pixels = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(lambda x: utils3d.pt.pixel_to_ndc(x, (8, 10)), pixels)
        torch.testing.assert_close(jacobian, torch.diag(torch.tensor([.2, -.25], dtype=torch.float64)),
                                   rtol=0, atol=1e-15)
        sizes = torch.tensor([8.0, 10.0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda p, s: utils3d.pt.pixel_to_ndc(p, s), (pixels, sizes))


if __name__ == '__main__':
    unittest.main()
