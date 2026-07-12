#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor mpelu_forward_cuda(
    const torch::Tensor input,
    const torch::Tensor a,
    const torch::Tensor b
);


void mpelu_backward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input,
    torch::Tensor& grad_a,
    torch::Tensor& grad_b
);
