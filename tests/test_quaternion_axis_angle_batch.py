import math
import unittest

import numpy as np
import torch
import utils3d



class TestQuaternionAxisAngleBatch(unittest.TestCase):
    def test_analytic_rotations_and_batch_shapes(self):
        angles = np.array([0.2, 0.7, 1.1, 2.4], dtype=np.float64)
        axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, -1, 3]], dtype=np.float64)
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        q = np.concatenate([np.cos(angles[:, None] / 2), axes * np.sin(angles[:, None] / 2)], axis=-1)
        expected = axes * angles[:, None]
        for shape in [(4, 4), (2, 2, 4)]:
            with self.subTest(shape=shape):
                actual = utils3d.np.quaternion_to_axis_angle(q.reshape(shape))
                self.assertEqual(actual.shape, (*shape[:-1], 3))
                np.testing.assert_allclose(actual.reshape(-1, 3), expected, rtol=1e-14, atol=1e-14)
                torch.testing.assert_close(torch.from_numpy(actual), utils3d.pt.quaternion_to_axis_angle(torch.from_numpy(q.reshape(shape))))

    def test_noncontiguous_two_and_three_quaternions(self):
        angle = np.array([0.3, 0.9, 1.5], dtype=np.float64)
        q = np.stack([np.cos(angle / 2), np.sin(angle / 2), np.zeros(3), np.zeros(3)], axis=0).T
        self.assertFalse(q.flags.c_contiguous)
        for n in [2, 3]:
            with self.subTest(n=n):
                expected = np.stack([angle[:n], np.zeros(n), np.zeros(n)], axis=-1)
                np.testing.assert_allclose(utils3d.quaternion_to_axis_angle(q[:n]), expected, atol=1e-14)

    def test_matrix_conversion_and_zero_angle(self):
        angles = np.array([0.0, 0.4], dtype=np.float64)
        matrices = np.zeros((2, 3, 3), dtype=np.float64)
        matrices[:, 0, 0] = matrices[:, 1, 1] = np.cos(angles)
        matrices[:, 0, 1] = -np.sin(angles)
        matrices[:, 1, 0] = np.sin(angles)
        matrices[:, 2, 2] = 1
        expected = np.stack([np.zeros(2), np.zeros(2), angles], axis=-1)
        np.testing.assert_allclose(utils3d.np.matrix_to_axis_angle(matrices), expected, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
