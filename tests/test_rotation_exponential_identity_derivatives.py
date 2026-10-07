import math
import unittest

import numpy as np
import torch
import utils3d


GENERATORS = np.array([[[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                       [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                       [[0, -1, 0], [1, 0, 0], [0, 0, 0]]], dtype=np.float64)


def matrix_series(vector):
    # Independent matrix-exponential definition, without axis normalization.
    generator = sum(float(v) * g for v, g in zip(vector, GENERATORS))
    result, term = np.eye(3), np.eye(3)
    for k in range(1, 40):
        term = term @ generator / k
        result = result + term
    return result


class TestRotationExponentialIdentityDerivatives(unittest.TestCase):
    def test_quaternion_identity_jacobian_and_hessian(self):
        zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(utils3d.pt.axis_angle_to_quaternion, zero)
        expected = torch.cat([torch.zeros(1, 3, dtype=torch.float64),
                              torch.eye(3, dtype=torch.float64) / 2])
        torch.testing.assert_close(jacobian, expected, atol=1e-14, rtol=0)
        for component in range(4):
            with self.subTest(component=component):
                hessian = torch.autograd.functional.hessian(
                    lambda v: utils3d.pt.axis_angle_to_quaternion(v)[component], zero)
                expected_hessian = -torch.eye(3, dtype=torch.float64) / 4 if component == 0 else torch.zeros(3, 3, dtype=torch.float64)
                torch.testing.assert_close(hessian, expected_hessian, atol=1e-14, rtol=0)

    def test_matrix_identity_first_and_second_differentials(self):
        zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(utils3d.pt.axis_angle_to_matrix, zero)
        np.testing.assert_allclose(jacobian.detach().numpy(), np.moveaxis(GENERATORS, 0, -1), atol=1e-14)
        expected = np.empty((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                expected[..., i, j] = (GENERATORS[i] @ GENERATORS[j] + GENERATORS[j] @ GENERATORS[i]) / 2
        for row in range(3):
            for column in range(3):
                with self.subTest(row=row, column=column):
                    hessian = torch.autograd.functional.hessian(
                        lambda v: utils3d.pt.axis_angle_to_matrix(v)[row, column], zero)
                    np.testing.assert_allclose(hessian.detach().numpy(), expected[row, column], atol=1e-13)

    def test_small_float32_rotations_retain_second_order_terms(self):
        vector = np.array([[1e-4, 2e-4, 0], [0, -3e-4, 2e-4]], dtype=np.float32)
        expected = np.stack([matrix_series(v) for v in vector]).astype(np.float32)
        for backend in [utils3d.np, utils3d.pt]:
            with self.subTest(backend=backend.__name__):
                arg = vector if backend is utils3d.np else torch.from_numpy(vector)
                result = backend.axis_angle_to_matrix(arg)
                result = result if isinstance(result, np.ndarray) else result.detach().numpy()
                np.testing.assert_allclose(result, expected, rtol=2e-7, atol=1e-10)
                self.assertEqual(result.dtype, np.float32)

    def test_ordinary_batched_rotations_and_gradgradcheck(self):
        vectors = np.array([[[0, 0, 0], [0.2, -0.4, 0.3]],
                            [[math.pi, 0, 0], [-0.2, 0.8, -1.4]]], dtype=np.float64)
        expected = np.stack([matrix_series(v) for v in vectors.reshape(-1, 3)]).reshape(2, 2, 3, 3)
        for backend in [utils3d.np, utils3d.pt]:
            with self.subTest(backend=backend.__name__):
                arg = vectors if backend is utils3d.np else torch.from_numpy(vectors)
                result = backend.axis_angle_to_matrix(arg)
                result = result if isinstance(result, np.ndarray) else result.detach().numpy()
                np.testing.assert_allclose(result, expected, atol=3e-14)
                q = backend.axis_angle_to_quaternion(arg)
                q = q if isinstance(q, np.ndarray) else q.detach().numpy()
                np.testing.assert_allclose(np.sum(q * q, axis=-1), 1, atol=1e-14)
        for function in [utils3d.pt.axis_angle_to_matrix, utils3d.pt.axis_angle_to_quaternion]:
            for values in [[0., 0., 0.], [1e-6, -2e-6, 3e-6]]:
                with self.subTest(function=function.__name__, values=values):
                    arg = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                    self.assertTrue(torch.autograd.gradcheck(function, (arg,)))
                    self.assertTrue(torch.autograd.gradgradcheck(function, (arg,)))


if __name__ == '__main__':
    unittest.main()
