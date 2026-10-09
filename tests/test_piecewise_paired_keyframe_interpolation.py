import bisect
import math
import unittest

import numpy as np

import utils3d


def _reference(values, knots, queries, mode):
    output = []
    for query in queries:
        if mode == 'constant' and query <= knots[0]:
            output.append(values[0])
        elif mode == 'constant' and query >= knots[-1]:
            output.append(values[-1])
        else:
            lower = min(max(bisect.bisect_left(knots, query) - 1, 0), len(knots) - 2)
            weight = (query - knots[lower]) / (knots[lower + 1] - knots[lower])
            output.append(values[lower] + weight * (values[lower + 1] - values[lower]))
    return np.asarray(output)


def _z_pose(angle, translation):
    pose = np.eye(4)
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    pose[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    pose[:3, 3] = translation
    return pose


class TestPiecewisePairedInterpolation(unittest.TestCase):
    def test_unequal_query_and_feature_counts_reproduce_an_affine_path(self):
        knots = np.array([0., 1., 3.])
        queries = np.array([0., 0.4, 1., 1.5, 3.])
        offset, velocity = np.array([0.2, -0.3, 0.4]), np.array([1., 2., -1.])
        values = offset + knots[:, None] * velocity
        actual = utils3d.numpy.piecewise_lerp(values, knots, queries)
        self.assertEqual(actual.shape, (5, 3))
        np.testing.assert_allclose(actual, offset + queries[:, None] * velocity, rtol=0, atol=1e-14)

    def test_equal_counts_do_not_mix_distinct_query_intervals(self):
        knots = np.array([0., 1., 3.])
        queries = np.array([0.25, 1.5, 2.5])
        values = np.array([[0.2, 0.8, -0.3], [1.5, -0.2, 0.4], [0.3, 1.1, 2.]])
        actual = utils3d.numpy.piecewise_lerp(values, knots, queries)
        self.assertEqual(actual.shape, (3, 3))
        np.testing.assert_allclose(actual, _reference(values, knots, queries, 'constant'), rtol=0, atol=1e-14)

    def test_constant_boundaries_are_exact_for_ordinary_query_times(self):
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=dtype)
                knots = np.array([0., 1.], dtype=dtype)
                queries = np.array([-2., 0., 1., 3.], dtype=dtype)
                actual = utils3d.numpy.piecewise_lerp(values, knots, queries)
                expected = np.stack([values[0], values[0], values[1], values[1]])
                self.assertEqual(actual.shape, (4, 3))
                np.testing.assert_array_equal(actual, expected)

    def test_se3_pairs_match_an_independent_z_axis_camera_trajectory(self):
        knots = np.array([0., 1., 3.])
        angles = np.deg2rad([20., 65., 100.])
        positions = np.array([[0., 0., 0.], [1., -0.5, 0.4], [0.2, 1.5, 2.]])
        poses = np.stack([_z_pose(angle, point) for angle, point in zip(angles, positions)])
        queries = np.array([0., 0.25, 1., 1.7, 3.])
        actual = utils3d.numpy.piecewise_interpolate_se3_matrix(poses, knots, queries)
        expected_angles = _reference(angles[:, None], knots, queries, 'constant')[:, 0]
        expected_positions = _reference(positions, knots, queries, 'constant')
        expected = np.stack([_z_pose(angle, point) for angle, point in zip(expected_angles, expected_positions)])
        self.assertEqual(actual.shape, (5, 4, 4))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)

    def test_single_keyframe_constant_trajectory_is_unchanged(self):
        queries = np.array([-2., 0.5, 1., 3.])
        knots = np.array([0.5])
        pose = _z_pose(0.7, [0.1, -0.2, 0.3])
        actual_pose = utils3d.numpy.piecewise_interpolate_se3_matrix(pose[None], knots, queries)
        self.assertEqual(actual_pose.shape, (4, 4, 4))
        np.testing.assert_allclose(actual_pose, np.broadcast_to(pose, (4, 4, 4)), rtol=0, atol=1e-11)
        point = np.array([[0.1, 0.2, 0.3]])
        actual_point = utils3d.numpy.piecewise_lerp(point, knots, queries)
        self.assertEqual(actual_point.shape, (4, 3))
        np.testing.assert_array_equal(actual_point, np.broadcast_to(point[0], (4, 3)))

    def test_linear_extrapolation_keeps_the_endpoint_velocity(self):
        knots = np.array([0., 1.])
        queries = np.array([-0.5, 0., 0.5, 1., 1.5])
        points = np.array([[0.2, -0.1, 0.3], [1.2, 0.4, -0.7]])
        expected_points = _reference(points, knots, queries, 'linear')
        actual_points = utils3d.numpy.piecewise_lerp(points, knots, queries, 'linear')
        self.assertEqual(actual_points.shape, (5, 3))
        np.testing.assert_allclose(actual_points, expected_points, rtol=0, atol=1e-14)
        poses = np.stack([_z_pose(math.radians(10), points[0]), _z_pose(math.radians(60), points[1])])
        actual_poses = utils3d.numpy.piecewise_interpolate_se3_matrix(poses, knots, queries, 'linear')
        expected_poses = np.stack([_z_pose(math.radians(10 + 50 * query), point)
                                   for query, point in zip(queries, expected_points)])
        self.assertEqual(actual_poses.shape, (5, 4, 4))
        np.testing.assert_allclose(actual_poses, expected_poses, rtol=0, atol=1e-11)
