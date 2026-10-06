import math
import unittest

import numpy as np
import torch
import utils3d



def z_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


class TestRotationSlerpShortestPath(unittest.TestCase):
    def test_wrap_across_pi_uses_short_geodesic(self):
        for degrees in [170.0, 179.0, 179.9]:
            for backend in [utils3d.np, utils3d.pt]:
                with self.subTest(degrees=degrees, backend=backend.__name__):
                    start = math.radians(degrees)
                    r1, r2 = z_rotation(start), z_rotation(-start)
                    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
                    expected = np.stack([z_rotation(start + u * (2 * math.pi - 2 * start)) for u in t])
                    args = (r1, r2, t) if backend is utils3d.np else tuple(torch.from_numpy(x) for x in [r1, r2, t])
                    actual = backend.slerp_rotation_matrix(*args)
                    actual = actual if isinstance(actual, np.ndarray) else actual.numpy()
                    np.testing.assert_allclose(actual, expected, atol=3e-12)
                    np.testing.assert_allclose(actual.swapaxes(-1, -2) @ actual, np.broadcast_to(np.eye(3), actual.shape), atol=3e-12)

    def test_unwrapped_control_and_se3_translation(self):
        r1, r2 = z_rotation(0.2), z_rotation(0.8)
        t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        for backend in [utils3d.np, utils3d.pt]:
            with self.subTest(backend=backend.__name__):
                args = (r1, r2, t) if backend is utils3d.np else tuple(torch.from_numpy(x) for x in [r1, r2, t])
                actual = backend.slerp_rotation_matrix(*args)
                np.testing.assert_allclose(actual if isinstance(actual, np.ndarray) else actual.numpy(), np.stack([z_rotation(0.2 + 0.6 * u) for u in t]), atol=3e-12)
                transforms = np.stack([np.eye(4), np.eye(4)])
                transforms[0, :3, :3], transforms[1, :3, :3] = z_rotation(math.radians(170)), z_rotation(math.radians(-170))
                transforms[1, :3, 3] = [2, 4, 6]
                ts = (transforms[0], transforms[1], t) if backend is utils3d.np else tuple(torch.from_numpy(x) for x in [transforms[0], transforms[1], t])
                result = backend.interpolate_se3_matrix(*ts)
                result = result if isinstance(result, np.ndarray) else result.numpy()
                np.testing.assert_allclose(result[:, :3, 3], t[:, None] * [2, 4, 6], atol=1e-14)
                np.testing.assert_allclose(result[1, :3, :3], z_rotation(math.pi), atol=3e-12)


    def test_close_rotations_across_largest_quaternion_component_branch(self):
        # Extraction fixes the sign of the largest component, which switches
        # from positive X to negative Y for these nearby, ordinary rotations.
        for rotation_degrees in [120.0, 170.0]:
            half_angle = math.radians(rotation_degrees) / 2
            qs = [np.array([math.cos(half_angle),
                            math.sin(half_angle) * math.cos(math.radians(axis_degrees)),
                            -math.sin(half_angle) * math.sin(math.radians(axis_degrees)), 0.0])
                  for axis_degrees in [44.0, 46.0]]

            def matrix(q):
                w, x, y, z = q
                return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

            r1, r2 = [matrix(q) for q in qs]
            omega = math.acos(float(qs[0] @ qs[1]))
            t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
            expected = np.stack([matrix((math.sin((1 - u) * omega) * qs[0]
                                         + math.sin(u * omega) * qs[1]) / math.sin(omega)) for u in t])
            for backend in [utils3d.np, utils3d.pt]:
                with self.subTest(rotation_degrees=rotation_degrees, backend=backend.__name__):
                    args = (r1, r2, t) if backend is utils3d.np else tuple(torch.from_numpy(x) for x in [r1, r2, t])
                    actual = backend.slerp_rotation_matrix(*args)
                    actual = actual if isinstance(actual, np.ndarray) else actual.numpy()
                    np.testing.assert_allclose(actual, expected, atol=3e-12)
                    np.testing.assert_allclose(actual.swapaxes(-1, -2) @ actual,
                                               np.broadcast_to(np.eye(3), actual.shape), atol=3e-12)


if __name__ == '__main__':
    unittest.main()
