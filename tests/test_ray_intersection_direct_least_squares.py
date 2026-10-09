import math
import unittest

import numpy as np
import utils3d


class TestRayIntersectionDirectLeastSquares(unittest.TestCase):
    def test_arbitrary_direction_scales_preserve_intersection_and_parameters(self):
        point = np.array([2., -1., 4.])
        p1, p2 = point - [3., 0., 0.], point - [0., 2., 0.]
        for scale1 in [1e-9, 1., 1e9]:
            for scale2 in [1e-9, 1., 1e9]:
                with self.subTest(scale1=scale1, scale2=scale2):
                    d1, d2 = np.array([scale1, 0., 0.]), np.array([0., scale2, 0.])
                    actual, (t1, t2) = utils3d.ray_intersection(p1, d1, p2, d2)
                    np.testing.assert_allclose(actual, point, rtol=0, atol=1e-13)
                    np.testing.assert_allclose(p1 + t1 * d1, point, rtol=0, atol=1e-13)
                    np.testing.assert_allclose(p2 + t2 * d2, point, rtol=0, atol=1e-13)

    def test_small_parallax_uses_the_unsquared_least_squares_system(self):
        expected = np.array([2., 1., 3.])
        d1 = np.array([1., 0., 0.])
        for angle in [1e-3, 1e-5, 1e-6]:
            with self.subTest(angle=angle):
                d2 = np.array([math.cos(angle), math.sin(angle), 0.])
                p1, p2 = expected - 2 * d1, expected - 3 * d2
                actual, (t1, t2) = utils3d.np.ray_intersection(p1, d1, p2, d2)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
                np.testing.assert_allclose([t1, t2], [2., 3.], rtol=0, atol=2e-8)

    def test_skew_rays_return_the_midpoint_and_are_rigid_frame_equivariant(self):
        p1, p2 = np.array([-1., 0., 0.]), np.array([0., -1., 2.])
        d1, d2 = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        expected = np.array([0., 0., 1.])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = np.array([40., -30., 50.])
        actual, parameters = utils3d.np.ray_intersection(p1, d1, p2, d2)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_allclose(parameters, [1, 1], atol=1e-14)
        shifted, shifted_parameters = utils3d.np.ray_intersection(
            rotation @ p1 + translation, rotation @ d1,
            rotation @ p2 + translation, rotation @ d2)
        np.testing.assert_allclose(shifted, rotation @ expected + translation, atol=1e-12)
        np.testing.assert_allclose(shifted_parameters, parameters, atol=1e-12)

    def test_broadcasted_noncontiguous_inputs_and_different_dimensions(self):
        for dimension in [2, 3, 4]:
            for dtype in [np.float32, np.float64]:
                with self.subTest(dimension=dimension, dtype=dtype):
                    points = np.arange(4 * dimension, dtype=dtype).reshape(4, dimension)[::2]
                    first = np.zeros(dimension, dtype=dtype)
                    second = first.copy()
                    first[0], second[1] = 1, 1
                    actual, (t1, t2) = utils3d.np.ray_intersection(points - first, first, points - second, second)
                    self.assertEqual(actual.shape, (2, dimension))
                    np.testing.assert_allclose(actual, points, atol=2e-6 if dtype == np.float32 else 1e-13)
                    np.testing.assert_allclose(t1, 1, atol=2e-6 if dtype == np.float32 else 1e-13)
                    np.testing.assert_allclose(t2, 1, atol=2e-6 if dtype == np.float32 else 1e-13)
                    empty, _ = utils3d.np.ray_intersection(points[:0], first, points[:0], second)
                    self.assertEqual(empty.shape, (0, dimension))

    def test_zero_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            utils3d.np.ray_intersection(np.zeros(3), np.zeros(3), np.ones(3), np.ones(3))


if __name__ == '__main__':
    unittest.main()
