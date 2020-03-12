#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024


inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}


template <typename scalar_t>
__global__ void PropTcfgForward(const int nthreads,
                                const scalar_t *input,
                                const int channels,
                                const int tscale,
                                const int start_num,
                                const int center_num,
                                const int end_num,
                                scalar_t *output) {
    const int feature_num = start_num + center_num + end_num;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int w = index % tscale;
        const int h = (index / tscale) % tscale;
        const int t = (index / tscale / tscale) % feature_num;
        const int c = (index / tscale / tscale / feature_num) % channels;
        const int n = index / tscale / tscale/ feature_num / channels;
        if (h > w) {
            output[index] = 0;
        } else {
            float flen = w - h;
            float start, step, region_len;
            float idx, l_w, r_w;
            int l_idx, r_idx;
            scalar_t target, l_val = 0, r_val = 0;
            if (t < start_num) {
                // start region
                flen = flen + 1;
                start = h - flen / 5.0;
                region_len = 2.0 * flen / 5.0;
                step = t;
                idx = start + step * region_len / (start_num - 1);
                l_idx = static_cast<int>(floorf(idx));
                r_idx = l_idx + 1;
                l_w = r_idx - idx;
                r_w = idx - l_idx;
                l_idx = max(min(l_idx, tscale - 1), 0);
                r_idx = max(min(r_idx, tscale - 1), 0);
                l_val = input[n * channels * tscale + c * tscale + l_idx];
                r_val = input[n * channels * tscale + c * tscale + r_idx];
                target = l_w * l_val + r_w * r_val;
            } else if (t >= start_num + center_num) {
                // end region
                flen = flen + 1;
                start = w - flen / 5.0;
                region_len = 2.0 * flen / 5.0;
                step = t - (start_num + center_num);
                idx = start + step * region_len / (end_num - 1);
                l_idx = static_cast<int>(floorf(idx));
                r_idx = l_idx + 1;
                l_w = r_idx - idx;
                r_w = idx - l_idx;
                l_idx = max(min(l_idx, tscale - 1), 0);
                r_idx = max(min(r_idx, tscale - 1), 0);
                l_val = input[n * channels * tscale + c * tscale + l_idx];
                r_val = input[n * channels * tscale + c * tscale + r_idx];
                target = l_w * l_val + r_w * r_val;
            } else {
                // action region
                start = h;
                step = t - start_num;
                region_len = flen;
                idx = start + step * region_len / (center_num - 1);
                l_idx = static_cast<int>(floorf(idx));
                r_idx = l_idx + 1;
                l_w = r_idx - idx;
                r_w = idx - l_idx;
                l_idx = max(min(l_idx, tscale - 1), 0);
                r_idx = max(min(r_idx, tscale - 1), 0);
                l_val = input[n * channels * tscale + c * tscale + l_idx];
                r_val = input[n * channels * tscale + c * tscale + r_idx];
                target = l_w * l_val + r_w * r_val;
            }
            output[index] = target;
        }
    }
}

template <typename scalar_t>
__global__ void PropTcfgBackward(const int nthreads,
                                 const scalar_t *grad_output,
                                 const int channels,
                                 const int tscale,
                                 const int start_num,
                                 const int center_num,
                                 const int end_num,
                                 scalar_t *grad_input) {
    const int feature_num = start_num + center_num + end_num;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int w = index % tscale;
        const int h = (index / tscale) % tscale;
        const int t = (index / tscale / tscale) % feature_num;
        const int c = (index / tscale / tscale / feature_num) % channels;
        const int n = index / tscale / tscale/ feature_num / channels;
        if (h <= w) {
            float flen = w - h;
            float start, step, region_len;
            float idx, l_w, r_w;
            int l_idx, r_idx;
            scalar_t grad = grad_output[index];
            if (t < start_num) {
                // start region
                flen = flen + 1;
                start = h - flen / 5.0;
                region_len = 2.0 * flen / 5.0;
                step = t;
                idx = start + step * region_len / (start_num - 1);
                l_idx = static_cast<int>(floorf(idx));
                r_idx = l_idx + 1;
                l_w = r_idx - idx;
                r_w = idx - l_idx;
                l_idx = max(min(l_idx, tscale - 1), 0);
                r_idx = max(min(r_idx, tscale - 1), 0);
                atomicAdd(grad_input + n * channels * tscale + c * tscale + l_idx, l_w * grad);
                atomicAdd(grad_input + n * channels * tscale + c * tscale + r_idx, r_w * grad);
            }
            else if (t >= start_num + center_num) {
                // end region
                flen = flen + 1;
                start = w - flen / 5.0;
                region_len = 2.0 * flen / 5.0;
                step = t - (start_num + center_num);
                idx = start + step * region_len / (end_num - 1);
                l_idx = static_cast<int>(floorf(idx));
                r_idx = l_idx + 1;
                l_w = r_idx - idx;
                r_w = idx - l_idx;
                l_idx = max(min(l_idx, tscale - 1), 0);
                r_idx = max(min(r_idx, tscale - 1), 0);
                atomicAdd(grad_input + n * channels * tscale + c * tscale + l_idx, l_w * grad);
                atomicAdd(grad_input + n * channels * tscale + c * tscale + r_idx, r_w * grad);
            }
            else {
                // action region
                start = h;
                step = t - start_num;
                region_len = flen;
                idx = start + step * region_len / (center_num - 1);
                l_idx = static_cast<int>(floorf(idx));
                r_idx = l_idx + 1;
                l_w = r_idx - idx;
                r_w = idx - l_idx;
                l_idx = max(min(l_idx, tscale - 1), 0);
                r_idx = max(min(r_idx, tscale - 1), 0);
                atomicAdd(grad_input + n * channels * tscale + c * tscale + l_idx, l_w * grad);
                atomicAdd(grad_input + n * channels * tscale + c * tscale + r_idx, r_w * grad);
            }
        }
    }
}
int prop_tcfg_cuda_forward(
        const at::Tensor input,
        at::Tensor output,
        int start_num,
        int center_num,
        int end_num) {
    const int batch_size = output.size(0);
    const int channels = output.size(1);
    const int number_dim = output.size(2);
    const int tscale = output.size(3);
    const int output_size = batch_size * channels * number_dim * tscale * tscale;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "PropTcfgForward", ([&] {
        PropTcfgForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            output_size, input.data<scalar_t>(),
            channels, tscale, start_num, center_num, end_num,
            output.data<scalar_t>());
    }));

    THCudaCheck(cudaGetLastError());
    return 1;
}

int prop_tcfg_cuda_backward(
        const at::Tensor grad_output,
        at::Tensor grad_input,
        int start_num,
        int center_num,
        int end_num) {
    const int batch_size = grad_output.size(0);
    const int channels = grad_output.size(1);
    const int feature_num = grad_output.size(2);
    const int tscale = grad_output.size(3);
    const int output_size = batch_size * channels * feature_num * tscale * tscale;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.type(), "PropTcfgBackward", ([&] {
        PropTcfgBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            output_size, grad_output.data<scalar_t>(),
            channels, tscale, start_num, center_num, end_num,
            grad_input.data<scalar_t>());
    }));

    THCudaCheck(cudaGetLastError());
    return 1;
}