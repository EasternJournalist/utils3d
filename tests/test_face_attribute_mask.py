import math
import unittest

import numpy as np
import torch
import utils3d



class TestFaceAttributeMask(unittest.TestCase):
    def test_face_domain_mask_preserves_every_attribute(self):
        mask = np.array([[True, False, True], [False, True, False]])
        colors = np.arange(18).reshape(2, 3, 3)
        scalar = np.arange(6).reshape(2, 3)
        for tri in [False, True]:
            with self.subTest(tri=tri):
                faces, out_color, out_scalar = utils3d.np.build_mesh_from_map(colors, scalar, mask=mask, domain='face', tri=tri)
                repeats = 2 if tri else 1
                self.assertEqual(faces.shape, (int(mask.sum()) * repeats, 3 if tri else 4))
                if tri:
                    expected_color = np.repeat(colors[mask], 2, axis=0)
                    expected_scalar = np.repeat(scalar[mask], 2, axis=0)
                else:
                    expected_color, expected_scalar = colors[mask], scalar[mask]
                np.testing.assert_array_equal(out_color, expected_color)
                np.testing.assert_array_equal(out_scalar, expected_scalar)

    def test_empty_selection_and_no_attributes(self):
        mask = np.zeros((2, 3), dtype=bool)
        colors = np.zeros((2, 3, 3))
        for tri in [False, True]:
            with self.subTest(tri=tri):
                faces, result = utils3d.np.build_mesh_from_map(colors, mask=mask, domain='face', tri=tri)
                self.assertEqual(faces.shape, (0, 3 if tri else 4))
                self.assertEqual(result.shape, (0, 3))
                self.assertEqual(len(utils3d.np.build_mesh_from_map(mask=mask, domain='face', tri=tri)), 1)


if __name__ == '__main__':
    unittest.main()
