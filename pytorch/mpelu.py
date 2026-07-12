import torch
import mpelu_cuda
from torch.cuda.amp import custom_fwd, custom_bwd

class MPELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, alpha, beta):
        input_contiguous = input.contiguous()
        alpha_compute = alpha.to(dtype=input.dtype).contiguous()
        beta_compute = beta.to(dtype=input.dtype).contiguous()
        output = mpelu_cuda.mpelu_forward(
            input_contiguous, alpha_compute, beta_compute
        )
        ctx.save_for_backward(input_contiguous, alpha_compute, beta_compute)
        ctx.alpha_dtype = alpha.dtype
        ctx.beta_dtype = beta.dtype

        return output


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, alpha, beta = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(input)
        accumulation_dtype = (
            torch.float64 if input.dtype == torch.float64 else torch.float32
        )
        grad_a = torch.zeros_like(alpha, dtype=accumulation_dtype)
        grad_b = torch.zeros_like(beta, dtype=accumulation_dtype)

        mpelu_cuda.mpelu_backward(
            input, alpha, beta, grad_output, grad_input, grad_a, grad_b
        )
        
        return (
            grad_input,
            grad_a.to(dtype=ctx.alpha_dtype),
            grad_b.to(dtype=ctx.beta_dtype),
        )


class MPELU(torch.nn.Module):
    def __init__(self, num_channels):
        super(MPELU, self).__init__()
        self.alpha = torch.nn.Parameter(torch.Tensor(num_channels))
        self.beta = torch.nn.Parameter(torch.Tensor(num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.alpha, 0.25)
        torch.nn.init.ones_(self.beta)

    def forward(self, input):
        return MPELUFunction.apply(input, self.alpha, self.beta)
