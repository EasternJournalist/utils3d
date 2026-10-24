import unittest

import numpy as np
import torch
import utils3d


def as_numpy(value):
    return value if isinstance(value, np.ndarray) else value.detach().numpy()



def camera_data(count):
    matrices = np.array([[[1.0 + .1*i, .02*i, .4 + .01*i],
                         [0.0, 1.2 + .2*i, .6 - .03*i], [0.0, 0.0, 1.0]] for i in range(count)])
    sizes = np.array([[60 + 12*i, 80 + 10*i] for i in range(count)], dtype=np.float64)
    top = np.arange(count, dtype=np.float64) + 3
    left = 2*np.arange(count, dtype=np.float64) + 5
    heights = sizes[:, 0] - 2 * top
    widths = sizes[:, 1] - 2 * left
    expected = matrices.copy()
    for i, ((height, width), t, l, h, w) in enumerate(zip(sizes, top, left, heights, widths)):
        expected[i, 0] = (width * matrices[i, 0] - l * matrices[i, 2]) / w
        expected[i, 1] = (height * matrices[i, 1] - t * matrices[i, 2]) / h
    return (matrices, sizes, top, left, heights, widths), expected


class TestCropIntrinsicsCameraBatch(unittest.TestCase):
    def test_distinct_camera_batches_preserve_each_crop(self):
        for count in [2, 3, 4]:
            arrays, expected = camera_data(count)
            for backend in [utils3d.np, utils3d.pt]:
                with self.subTest(count=count, backend=backend.__name__):
                    args = arrays if backend is utils3d.np else tuple(torch.from_numpy(x) for x in arrays)
                    result = backend.crop_intrinsics(*args)
                    self.assertEqual(tuple(result.shape), (count, 3, 3))
                    np.testing.assert_allclose(as_numpy(result), expected, rtol=1e-14, atol=1e-14)
                    np.testing.assert_array_equal(as_numpy(args[0]), arrays[0])

    def test_crop_projection_and_leading_broadcast(self):
        arrays, expected = camera_data(4)
        arrays = tuple(x.reshape((2, 2) + x.shape[1:]) for x in arrays)
        expected = expected.reshape(2, 2, 3, 3)
        rays = np.array([[.2, -.3, 1.0], [-.1, .4, 1.0]])
        for backend in [utils3d.np, utils3d.pt]:
            with self.subTest(backend=backend.__name__):
                args = arrays if backend is utils3d.np else tuple(torch.from_numpy(x) for x in arrays)
                result = as_numpy(backend.crop_intrinsics(*args))
                np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
                original_uv = (arrays[0] @ rays.T).swapaxes(-1, -2)[..., :2]
                sizes, top, left, heights, widths = arrays[1:]
                cropped_uv = (result @ rays.T).swapaxes(-1, -2)[..., :2]
                pixel_size = sizes[..., None, ::-1]
                offsets = np.stack([left, top], axis=-1)[..., None, :]
                cropped_size = np.stack([widths, heights], axis=-1)[..., None, :]
                np.testing.assert_allclose(cropped_uv, (original_uv * pixel_size - offsets) / cropped_size,
                                           rtol=1e-14, atol=1e-14)
                single = tuple(x[0, 0] for x in args)
                np.testing.assert_allclose(as_numpy(backend.crop_intrinsics(*single)), expected[0, 0], atol=1e-14)

    def test_torch_batched_crop_gradcheck(self):
        arrays, _ = camera_data(2)
        args = tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in arrays)
        assert torch.autograd.gradcheck(lambda *values: utils3d.pt.crop_intrinsics(*values), args)


if __name__ == '__main__':
    unittest.main()
