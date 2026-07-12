#include <torch/extension.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void mpelu_forward_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ output,
    int64_t num_elements,
    int64_t channels,
    int64_t spatial_size
) {
    using acc_t = at::acc_type<scalar_t, true>;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < num_elements;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t channel = (index / spatial_size) % channels;
        const acc_t value = static_cast<acc_t>(input[index]);
        const acc_t result = value > acc_t(0)
            ? value
            : static_cast<acc_t>(alpha[channel]) *
                (exp(static_cast<acc_t>(beta[channel]) * value) - acc_t(1));
        output[index] = static_cast<scalar_t>(result);
    }
}

// Each block owns one (batch, channel) plane. Threads calculate grad_input
// while reducing both channel-wise parameter gradients in shared memory.
// Consequently, each block performs only two global atomic additions instead
// of two atomic additions for every input element.
template <typename scalar_t>
__global__ void mpelu_backward_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    at::acc_type<scalar_t, true>* __restrict__ grad_alpha,
    at::acc_type<scalar_t, true>* __restrict__ grad_beta,
    int64_t channels,
    int64_t spatial_size
) {
    using acc_t = at::acc_type<scalar_t, true>;
    extern __shared__ unsigned char shared_bytes[];
    acc_t* shared_alpha = reinterpret_cast<acc_t*>(shared_bytes);
    acc_t* shared_beta = shared_alpha + blockDim.x;

    const int64_t batch_channel = blockIdx.x;
    const int64_t channel = batch_channel % channels;
    const int64_t offset = batch_channel * spatial_size;
    const acc_t alpha_value = static_cast<acc_t>(alpha[channel]);
    const acc_t beta_value = static_cast<acc_t>(beta[channel]);
    acc_t alpha_sum = acc_t(0);
    acc_t beta_sum = acc_t(0);

    for (int64_t pixel = threadIdx.x; pixel < spatial_size;
         pixel += blockDim.x) {
        const int64_t index = offset + pixel;
        const acc_t value = static_cast<acc_t>(input[index]);
        const acc_t upstream = static_cast<acc_t>(grad_output[index]);

        if (value <= acc_t(0)) {
            const acc_t exponential = exp(beta_value * value);
            alpha_sum += upstream * (exponential - acc_t(1));
            beta_sum += upstream * alpha_value * value * exponential;
            grad_input[index] = static_cast<scalar_t>(
                upstream * alpha_value * beta_value * exponential
            );
        } else {
            grad_input[index] = static_cast<scalar_t>(upstream);
        }
    }

    shared_alpha[threadIdx.x] = alpha_sum;
    shared_beta[threadIdx.x] = beta_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_alpha[threadIdx.x] += shared_alpha[threadIdx.x + stride];
            shared_beta[threadIdx.x] += shared_beta[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        gpuAtomicAdd(grad_alpha + channel, shared_alpha[0]);
        gpuAtomicAdd(grad_beta + channel, shared_beta[0]);
    }
}

int get_forward_blocks(int64_t num_elements) {
    // A capped grid works with the grid-stride loop and avoids launching a
    // needlessly large number of blocks for very large feature maps.
    constexpr int kMaxBlocks = 4096;
    return static_cast<int>(std::min<int64_t>(
        (num_elements + kThreads - 1) / kThreads, kMaxBlocks
    ));
}

}  // namespace

torch::Tensor mpelu_forward_cuda(
    const torch::Tensor input,
    const torch::Tensor alpha,
    const torch::Tensor beta
) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto output = torch::empty_like(input);
    if (input.numel() == 0) {
        return output;
    }

    const int64_t channels = input.size(1);
    const int64_t spatial_size = input.size(2) * input.size(3);
    const int blocks = get_forward_blocks(input.numel());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "mpelu_forward_cuda", [&] {
            mpelu_forward_cuda_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                alpha.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                input.numel(),
                channels,
                spatial_size
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void mpelu_backward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& alpha,
    const torch::Tensor& beta,
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input,
    torch::Tensor& grad_alpha,
    torch::Tensor& grad_beta
) {
    c10::cuda::CUDAGuard device_guard(input.device());
    if (input.numel() == 0) {
        return;
    }

    const int64_t channels = input.size(1);
    const int64_t spatial_size = input.size(2) * input.size(3);
    const int64_t batch_channels = input.size(0) * channels;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "mpelu_backward_cuda", [&] {
            using acc_t = at::acc_type<scalar_t, true>;
            const size_t shared_memory = 2 * kThreads * sizeof(acc_t);
            mpelu_backward_cuda_kernel<scalar_t>
                <<<batch_channels, kThreads, shared_memory, stream>>>(
                    input.data_ptr<scalar_t>(),
                    alpha.data_ptr<scalar_t>(),
                    beta.data_ptr<scalar_t>(),
                    grad_output.data_ptr<scalar_t>(),
                    grad_input.data_ptr<scalar_t>(),
                    grad_alpha.data_ptr<acc_t>(),
                    grad_beta.data_ptr<acc_t>(),
                    channels,
                    spatial_size
                );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
