import math
import unittest

import numpy as np
import torch
import utils3d



class TestSlerpQueryBatch(unittest.TestCase):
    def test_distinct_arcs_with_multiple_query_times(self):
        v1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        theta = torch.tensor([0.4, 1.2], dtype=torch.float64)
        v2 = torch.stack([torch.stack([theta[0].cos(), theta[0].sin(), theta[0] * 0]), torch.stack([theta[1] * 0, theta[1].cos(), theta[1].sin()])])
        t = torch.tensor([[0.0, 0.25, 0.5, 1.0], [0.0, 0.2, 0.6, 1.0]], dtype=torch.float64)
        angle = theta[:, None] * t
        expected = torch.stack([torch.stack([angle[0].cos(), angle[0].sin(), angle[0] * 0], dim=-1), torch.stack([angle[1] * 0, angle[1].cos(), angle[1].sin()], dim=-1)])
        actual = utils3d.pt.slerp(v1, v2, t, eps=0.0)
        self.assertEqual(actual.shape, (2, 4, 3))
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(actual.numpy(), utils3d.np.slerp(v1.numpy(), v2.numpy(), t.numpy()), atol=1e-14)

    def test_equal_batch_and_query_count_does_not_mix_arcs(self):
        v1 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        theta = torch.tensor([0.3, 1.1], dtype=torch.float64)
        v2 = torch.stack([theta.cos(), theta.sin()], dim=-1)
        t = torch.tensor([[0.2, 0.8], [0.4, 0.9]], dtype=torch.float64)
        actual = utils3d.slerp(v1, v2, t, eps=0.0)
        expected = torch.stack([(theta[:, None] * t).cos(), (theta[:, None] * t).sin()], dim=-1)
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)

    def test_gradients_for_batched_endpoints_and_queries(self):
        v1 = torch.tensor([[1.0, 0.2, -0.1], [-0.2, 1.0, 0.3]], dtype=torch.float64, requires_grad=True)
        v2 = torch.tensor([[0.1, 0.8, 1.2], [0.7, 0.2, 1.0]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[0.2, 0.3, 0.7], [0.1, 0.4, 0.8]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(utils3d.pt.slerp, (v1, v2, t)))


if __name__ == '__main__':
    unittest.main()
