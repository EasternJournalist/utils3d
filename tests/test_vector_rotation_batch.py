import math
import unittest

import numpy as np
import torch
import utils3d



class TestVectorRotationBatch(unittest.TestCase):
    def test_alignment_orthogonality_and_determinant(self):
        a = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3], [2, -1, 1]], dtype=np.float64)
        b = np.array([[1, 2, 0], [1, -1, 2], [3, 1, -1], [1, 3, 2]], dtype=np.float64)
        for count in [2, 3, 4]:
            for backend in ['numpy', 'torch', 'dispatch']:
                with self.subTest(count=count, backend=backend):
                    if backend == 'numpy':
                        rotation = utils3d.np.rotation_matrix_from_vectors(a[:count], b[:count])
                    else:
                        fn = utils3d.pt.rotation_matrix_from_vectors if backend == 'torch' else utils3d.rotation_matrix_from_vectors
                        rotation = fn(torch.from_numpy(a[:count]), torch.from_numpy(b[:count])).numpy()
                    self.assertEqual(rotation.shape, (count, 3, 3))
                    ua = a[:count] / np.linalg.norm(a[:count], axis=-1, keepdims=True)
                    ub = b[:count] / np.linalg.norm(b[:count], axis=-1, keepdims=True)
                    np.testing.assert_allclose((rotation @ ua[..., None])[..., 0], ub, atol=2e-14)
                    np.testing.assert_allclose(rotation.swapaxes(-1, -2) @ rotation, np.broadcast_to(np.eye(3), rotation.shape), atol=2e-14)
                    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=2e-14)

    def test_leading_broadcast_and_noncontiguous(self):
        a = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)[:, None, :]
        b = np.array([[1, 2, 0], [2, 3, 4], [-2, 1, 3]], dtype=np.float64)[None, :, :]
        for backend in [utils3d.np, utils3d.pt]:
            with self.subTest(backend=backend.__name__):
                rotation = backend.rotation_matrix_from_vectors(a if backend is utils3d.np else torch.from_numpy(a), b if backend is utils3d.np else torch.from_numpy(b))
                actual = rotation if isinstance(rotation, np.ndarray) else rotation.numpy()
                self.assertEqual(actual.shape, (2, 3, 3, 3))
                np.testing.assert_allclose((actual @ a[..., None])[..., 0], np.broadcast_to(b / np.linalg.norm(b, axis=-1, keepdims=True), (2, 3, 3)), atol=2e-14)

    def test_torch_first_derivative(self):
        a = torch.tensor([[1.0, 0.2, 0.3], [0.4, 1.0, 0.5]], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([[0.3, 0.8, 1.1], [-0.4, 0.1, 1.0]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(utils3d.pt.rotation_matrix_from_vectors, (a, b)))


if __name__ == '__main__':
    unittest.main()
