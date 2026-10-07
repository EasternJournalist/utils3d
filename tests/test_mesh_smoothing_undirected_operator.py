import unittest

import numpy as np
import torch
import utils3d


def uniform_operator(count, faces):
    # Independent graph definition: one unit weight per unique neighbor.
    neighbors = [set() for _ in range(count)]
    for face in faces:
        for a, b in zip(face, np.roll(face, -1)):
            neighbors[a].add(b)
            neighbors[b].add(a)
    result = np.zeros((count, count), dtype=np.float64)
    for vertex, row in enumerate(neighbors):
        if not row:
            result[vertex, vertex] = 1
        else:
            result[vertex, list(row)] = 1 / len(row)
    return result


class TestMeshSmoothingUndirectedOperator(unittest.TestCase):
    def setUp(self):
        self.vertices = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.],
                                      [2., 1., 0.], [7., -3., 2.]], dtype=torch.float64)
        self.faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
        self.operator = uniform_operator(5, self.faces.numpy())

    def test_uniform_neighbors_are_winding_and_face_multiplicity_invariant(self):
        expected = self.operator @ self.vertices.numpy()
        variants = [self.faces, self.faces.flip(-1), self.faces.flip(0),
                    torch.cat([self.faces, self.faces[:1]])]
        for faces in variants:
            with self.subTest(faces=faces.tolist()):
                actual = utils3d.pt.compute_mesh_laplacian(self.vertices, faces)
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14)
                torch.testing.assert_close(actual[-1], self.vertices[-1], rtol=0, atol=0)

    def test_cotangent_weights_match_independent_right_triangle(self):
        vertices = self.vertices[[0, 1, 2, 4]]
        faces = torch.tensor([[0, 1, 2]])
        # Opposite cotangents are 0, 2, and 1/2 for this right triangle.
        expected = torch.tensor([[0.4, 0.8, 0], [0, 0, 0], [0, 0, 0],
                                 [7, -3, 2]], dtype=torch.float64)
        actual = utils3d.pt.compute_mesh_laplacian(vertices, faces, 'cotangent')
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
        translation = torch.tensor([100., -50., 20.])
        shifted = utils3d.pt.compute_mesh_laplacian(vertices + translation, faces, 'cotangent')
        torch.testing.assert_close(shifted, expected + translation, rtol=0, atol=2e-14)

    def test_laplacian_iterations_and_translation_equivariance(self):
        for times in [0, 1, 2, 4]:
            with self.subTest(times=times):
                expected = np.linalg.matrix_power(self.operator, times) @ self.vertices.numpy()
                actual = utils3d.laplacian_smooth_mesh(self.vertices, self.faces, times=times)
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14)
                translation = torch.tensor([4., -7., 2.])
                shifted = utils3d.pt.laplacian_smooth_mesh(self.vertices + translation, self.faces, times=times)
                torch.testing.assert_close(shifted, actual + translation, rtol=0, atol=2e-14)

    def test_taubin_matches_two_laplacian_displacement_passes(self):
        delta = self.operator - np.eye(5)
        for positive, negative in [(0.5, -0.51), (0.17, -0.34)]:
            with self.subTest(positive=positive, negative=negative):
                expected = (np.eye(5) + negative * delta) @ (np.eye(5) + positive * delta) @ self.vertices.numpy()
                actual = utils3d.pt.taubin_smooth_mesh(self.vertices, self.faces, positive, negative)
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14)
                torch.testing.assert_close(actual[-1], self.vertices[-1], rtol=0, atol=0)

    def test_hc_correction_uses_current_iterate_and_one_neighbor_pass(self):
        original = self.vertices.numpy()
        for times in [0, 1, 3]:
            with self.subTest(times=times):
                expected = original.copy()
                for _ in range(times):
                    previous = expected
                    smoothed = self.operator @ previous
                    correction = smoothed - (0.3 * original + 0.7 * previous)
                    expected = smoothed - (0.6 * correction + 0.4 * (self.operator @ correction))
                actual = utils3d.pt.laplacian_hc_smooth_mesh(self.vertices, self.faces, times=times, alpha=0.3, beta=0.6)
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14)

    def test_batched_vertices_and_true_autograd(self):
        values = torch.stack([self.vertices, self.vertices + 2])
        for weight in ['uniform', 'cotangent']:
            with self.subTest(weight=weight):
                expected = torch.stack([utils3d.pt.compute_mesh_laplacian(v, self.faces, weight) for v in values])
                torch.testing.assert_close(utils3d.pt.compute_mesh_laplacian(values, self.faces, weight), expected)
                inputs = self.vertices.detach().clone().requires_grad_()
                self.assertTrue(torch.autograd.gradcheck(
                    lambda v: utils3d.pt.compute_mesh_laplacian(v, self.faces, weight), (inputs,)))

    def test_empty_faces_preserve_positions_and_empty_mesh_shapes(self):
        faces = torch.empty((0, 3), dtype=torch.int64)
        for function in [utils3d.pt.compute_mesh_laplacian, utils3d.pt.laplacian_smooth_mesh,
                         utils3d.pt.taubin_smooth_mesh, utils3d.pt.laplacian_hc_smooth_mesh]:
            with self.subTest(function=function.__name__):
                torch.testing.assert_close(function(self.vertices, faces), self.vertices, rtol=0, atol=0)
                self.assertEqual(tuple(function(self.vertices[:0], faces).shape), (0, 3))


if __name__ == '__main__':
    unittest.main()
