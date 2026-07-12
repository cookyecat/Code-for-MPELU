#include "mpelu.h"

namespace {

void check_mpelu_arguments(
    const torch::Tensor& input,
    const torch::Tensor& a,
    const torch::Tensor& b
) {
    CHECK_INPUT(input);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(input.dim() == 4, "input must be a 4D NCHW tensor");
    TORCH_CHECK(a.dim() == 1 && b.dim() == 1,
                "alpha and beta must be 1D channel-wise tensors");
    TORCH_CHECK(a.numel() == input.size(1) && b.numel() == input.size(1),
                "alpha and beta must contain one value per input channel");
    TORCH_CHECK(input.scalar_type() == a.scalar_type() &&
                input.scalar_type() == b.scalar_type(),
                "input, alpha, and beta must have the same dtype");
    TORCH_CHECK(input.device() == a.device() && input.device() == b.device(),
                "input, alpha, and beta must be on the same CUDA device");
}

}  // namespace

torch::Tensor mpelu_forward(
    torch::Tensor input,
    torch::Tensor a,
    torch::Tensor b
) {
    check_mpelu_arguments(input, a, b);

    return mpelu_forward_cuda(input, a, b);
}

void mpelu_backward(
    const torch::Tensor& input,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input,
    torch::Tensor& grad_a,
    torch::Tensor& grad_b
) {
    check_mpelu_arguments(input, a, b);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_input);
    CHECK_INPUT(grad_a);
    CHECK_INPUT(grad_b);

    TORCH_CHECK(grad_output.sizes() == input.sizes(),
                "grad_output must have the same shape as input");
    TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(),
                "grad_output must have the same dtype as input");
    TORCH_CHECK(grad_output.device() == input.device(),
                "grad_output must be on the same CUDA device as input");
    TORCH_CHECK(grad_input.sizes() == input.sizes(),
                "grad_input must have the same shape as input");
    TORCH_CHECK(grad_a.sizes() == a.sizes() && grad_b.sizes() == b.sizes(),
                "parameter gradients must match alpha and beta");
    const auto accumulation_type = input.scalar_type() == at::kDouble
        ? at::kDouble
        : at::kFloat;
    TORCH_CHECK(grad_input.scalar_type() == input.scalar_type(),
                "grad_input must have the input dtype");
    TORCH_CHECK(grad_a.scalar_type() == accumulation_type &&
                grad_b.scalar_type() == accumulation_type,
                "parameter gradients must use the accumulation dtype");
    TORCH_CHECK(grad_input.device() == input.device() &&
                grad_a.device() == input.device() &&
                grad_b.device() == input.device(),
                "all output gradients must be on the input CUDA device");

    mpelu_backward_cuda(input, a, b, grad_output, grad_input, grad_a, grad_b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("mpelu_forward", &mpelu_forward);
    m.def("mpelu_backward", &mpelu_backward);
}
