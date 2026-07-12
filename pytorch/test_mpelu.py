import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpelu import MPELU  # noqa: E402


def reference_mpelu(input, alpha, beta):
    alpha = alpha.to(input.dtype).view(1, -1, 1, 1)
    beta = beta.to(input.dtype).view(1, -1, 1, 1)
    return torch.where(
        input > 0,
        input,
        alpha * (torch.exp(beta * input) - 1),
    )


@unittest.skipUnless(torch.cuda.is_available(), "MPELU extension requires CUDA")
class MPELUCudaTest(unittest.TestCase):
    def test_forward_and_backward_match_reference(self):
        torch.manual_seed(7)
        device = torch.device("cuda")
        custom = MPELU(5).to(device=device, dtype=torch.float64)
        alpha_ref = custom.alpha.detach().clone().requires_grad_()
        beta_ref = custom.beta.detach().clone().requires_grad_()
        input_custom = torch.randn(
            3, 5, 7, 9, device=device, dtype=torch.float64, requires_grad=True
        )
        input_ref = input_custom.detach().clone().requires_grad_()
        upstream = torch.randn_like(input_custom)

        output_custom = custom(input_custom)
        output_ref = reference_mpelu(input_ref, alpha_ref, beta_ref)
        output_custom.backward(upstream)
        output_ref.backward(upstream)

        torch.testing.assert_close(output_custom, output_ref)
        torch.testing.assert_close(input_custom.grad, input_ref.grad)
        torch.testing.assert_close(custom.alpha.grad, alpha_ref.grad)
        torch.testing.assert_close(custom.beta.grad, beta_ref.grad)

    def test_channels_last_input(self):
        device = torch.device("cuda")
        module = MPELU(4).to(device)
        input = torch.randn(2, 4, 8, 8, device=device).to(
            memory_format=torch.channels_last
        )
        input.requires_grad_()

        output = module(input)
        output.sum().backward()

        self.assertEqual(output.shape, input.shape)
        self.assertIsNotNone(input.grad)
        self.assertTrue(torch.isfinite(input.grad).all())

    def test_zero_alpha_has_finite_gradients(self):
        device = torch.device("cuda")
        module = MPELU(3).to(device)
        module.alpha.data.zero_()
        input = -torch.rand(2, 3, 4, 4, device=device)
        input.requires_grad_()

        module(input).sum().backward()

        self.assertTrue(torch.isfinite(module.alpha.grad).all())
        self.assertTrue(torch.isfinite(module.beta.grad).all())
        self.assertTrue(torch.isfinite(input.grad).all())

    def test_half_input_accumulates_parameter_gradients_in_float(self):
        device = torch.device("cuda")
        module = MPELU(3).to(device)
        input = torch.randn(
            4, 3, 8, 8, device=device, dtype=torch.float16, requires_grad=True
        )

        module(input).float().sum().backward()

        self.assertEqual(module.alpha.grad.dtype, torch.float32)
        self.assertEqual(module.beta.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(module.alpha.grad).all())
        self.assertTrue(torch.isfinite(module.beta.grad).all())

    def test_uses_current_stream(self):
        device = torch.device("cuda")
        module = MPELU(3).to(device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            input = torch.randn(4, 3, 16, 16, device=device, requires_grad=True)
            output = module(input)
            output.sum().backward()
        torch.cuda.current_stream().wait_stream(stream)

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(input.grad).all())


if __name__ == "__main__":
    unittest.main()
