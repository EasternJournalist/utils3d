import math
import unittest

import numpy as np
import torch
import utils3d



def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


class TestEssentialMatrixGeometry(unittest.TestCase):
    def test_real_camera_correspondences_and_matrix_structure(self):
        theta = [0.2, -0.5, 1.1, -0.8]
        matrices = np.tile(np.eye(4), (4, 1, 1))
        for i, angle in enumerate(theta):
            c, s = math.cos(angle), math.sin(angle)
            matrices[i, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        matrices[:, :3, 3] = [[1, 2, 3], [-1, 0.5, 2], [0.2, -3, 1], [2, 1, -0.7]]
        points = np.array([[0.1, 0.2, 4], [0.5, -0.3, 5], [-0.2, 0.7, 3]], dtype=np.float64)
        expected = np.stack([skew(t[:3, 3]) @ t[:3, :3] for t in matrices])
        for shape in [(4, 4, 4), (2, 2, 4, 4)]:
            for backend in [utils3d.np, utils3d.pt]:
                with self.subTest(shape=shape, backend=backend.__name__):
                    arg = matrices.reshape(shape)
                    actual = backend.extrinsics_to_essential(arg if backend is utils3d.np else torch.from_numpy(arg))
                    actual = actual if isinstance(actual, np.ndarray) else actual.detach().numpy()
                    self.assertEqual(actual.shape, (*shape[:-2], 3, 3))
                    actual = actual.reshape(4, 3, 3)
                    np.testing.assert_allclose(actual, expected, atol=1e-14)
                    projected = points @ matrices[:, :3, :3].swapaxes(-1, -2) + matrices[:, None, :3, 3]
                    residual = np.einsum('bni,bij,nj->bn', projected, actual, points)
                    np.testing.assert_allclose(residual, 0, atol=2e-14)
                    singular = np.linalg.svd(actual, compute_uv=False)
                    np.testing.assert_allclose(singular[:, 0], singular[:, 1], atol=2e-14)
                    np.testing.assert_allclose(singular[:, 2], 0, atol=2e-14)

    def test_single_transform_zero_translation_and_autograd(self):
        transform = torch.eye(4, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(utils3d.pt.extrinsics_to_essential(transform), torch.zeros(3, 3, dtype=torch.float64))
        input = torch.tensor([[1.0, 0.2, -0.3, 2.0], [-0.1, 0.8, 0.4, -1.0], [0.2, -0.3, 1.1, 0.5], [0, 0, 0, 1]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(utils3d.pt.extrinsics_to_essential, (input,)))


if __name__ == '__main__':
    unittest.main()
