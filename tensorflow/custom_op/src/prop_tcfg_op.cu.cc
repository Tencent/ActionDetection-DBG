#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "prop_tcfg.h"

namespace tensorflow {
	typedef Eigen::GpuDevice GPUDevice;
	namespace functor {
		template <typename T>
		__global__ void PropTcfg(
			const int nthreads,
			const int batch, const int tscale, const int in_channel,
			const T* input, T* out,
			const int start_num, const int center_num, const int end_num) {

			const int feature_num = start_num + center_num + end_num;
			CUDA_1D_KERNEL_LOOP(index, nthreads) {
				const int c = index % feature_num;
				const int w = (index / feature_num) % tscale;
				const int h = (index / feature_num / tscale) % tscale;
				const int t = (index / feature_num / tscale / tscale) % in_channel;
				const int n = index / feature_num / tscale / tscale / in_channel;

				if (h > w) {
					out[index] = 0;
				}
				else {
					float flen = w - h;
					float start, step, region_len;
					float idx, l_w, r_w;
					int l_idx, r_idx;
					T target, l_val = 0, r_val = 0;
					if (c < start_num) {
						// start region
						flen = flen + 1;
						start = h - flen / 5.0;
						region_len = 2.0 * flen / 5.0;
						step = c;
						idx = start + step * region_len / (start_num - 1);
						l_idx = static_cast<int>(floorf(idx));
						r_idx = l_idx + 1;
						l_w = r_idx - idx;
						r_w = idx - l_idx;
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						l_val = ldg(input + n * tscale * in_channel + l_idx * in_channel + t);
						r_val = ldg(input + n * tscale * in_channel + r_idx * in_channel + t);
						target = l_w * l_val + r_w * r_val;
					}
					else if (c >= start_num + center_num) {
						// end region
						flen = flen + 1;
						start = w - flen / 5.0;
						region_len = 2.0 * flen / 5.0;
						step = c - (start_num + center_num);
						idx = start + step * region_len / (end_num - 1);
						l_idx = static_cast<int>(floorf(idx));
						r_idx = l_idx + 1;
						l_w = r_idx - idx;
						r_w = idx - l_idx;
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						l_val = ldg(input + n * tscale * in_channel + l_idx * in_channel + t);
						r_val = ldg(input + n * tscale * in_channel + r_idx * in_channel + t);
						target = l_w * l_val + r_w * r_val;
					}
					else {
						// center region
						start = h;
						step = c - start_num;
						region_len = flen;
						idx = start + step * region_len / (center_num - 1);
						l_idx = static_cast<int>(floorf(idx));
						r_idx = l_idx + 1;
						l_w = r_idx - idx;
						r_w = idx - l_idx;
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						l_val = ldg(input + n * tscale * in_channel + l_idx * in_channel + t);
						r_val = ldg(input + n * tscale * in_channel + r_idx * in_channel + t);
						target = l_w * l_val + r_w * r_val;
					}
					out[index] = target;
				}
			}
		}

		template <typename T>
		__global__ void PropTcfgV2(
			const int nthreads,
			const int batch, const int tscale, const int in_channel,
			const T* input, T* out, 
			const int start_num, const int center_num, const int end_num) {

			const int feature_num = start_num + center_num + end_num;
			CUDA_1D_KERNEL_LOOP(index, nthreads) {
				const int c = index % in_channel;
				const int w = (index / in_channel) % tscale;
				const int h = (index / in_channel / tscale) % tscale;
				const int t = (index / in_channel / tscale / tscale) % feature_num;
				const int n = index / in_channel / tscale / tscale / feature_num;

				if (h > w) {
					out[index] = 0;
				}
				else {
					float flen = w - h;
					float start, step, region_len;
					float idx, l_w, r_w;
					int l_idx, r_idx;
					T target, l_val = 0, r_val = 0;
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
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						l_val = ldg(input + n * tscale * in_channel + l_idx * in_channel + c);
						r_val = ldg(input + n * tscale * in_channel + r_idx * in_channel + c);
						target = l_w * l_val + r_w * r_val;
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
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						l_val = ldg(input + n * tscale * in_channel + l_idx * in_channel + c);
						r_val = ldg(input + n * tscale * in_channel + r_idx * in_channel + c);
						target = l_w * l_val + r_w * r_val;
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
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						l_val = ldg(input + n * tscale * in_channel + l_idx * in_channel + c);
						r_val = ldg(input + n * tscale * in_channel + r_idx * in_channel + c);
						target = l_w * l_val + r_w * r_val;
					}
					out[index] = target;
				}
			}
		}

		template <typename T>
		struct PropTcfgFunctor<GPUDevice, T> {
			void operator()(const GPUDevice& d,
				int batch, int tscale, int in_channel,
				const T* input, T* out, int mode,
				int start_num, int center_num, int end_num) {
				CudaLaunchConfig config;
				int size;
				int feature_num = start_num + center_num + end_num;
				size = batch * in_channel * tscale * tscale * feature_num;
				config = GetCudaLaunchConfig(size, d);
				if (mode == 0) {
					PropTcfg<T>
						<< <config.block_count, config.thread_per_block, 0, d.stream() >> > (
							config.virtual_thread_count,
							batch, tscale, in_channel, input, out,
							start_num, center_num, end_num);
				}
				else {
					PropTcfgV2<T>
						<< <config.block_count, config.thread_per_block, 0, d.stream() >> > (
							config.virtual_thread_count,
							batch, tscale, in_channel, input, out,
							start_num, center_num, end_num);
				}

			}
		};

		template <typename T>
		__global__ void PropTcfgGrad(
			const int nthreads,
			const int batch, const int tscale, const int in_channel,
			const T* output_grad, T* input_grad,
			const int start_num, const int center_num, const int end_num) {
			
			const int feature_num = start_num + center_num + end_num;
			CUDA_1D_KERNEL_LOOP(index, nthreads) {
				const int c = index % feature_num;
				const int w = (index / feature_num) % tscale;
				const int h = (index / feature_num / tscale) % tscale;
				const int t = (index / feature_num / tscale / tscale) % in_channel;
				const int n = index / feature_num / tscale / tscale / in_channel;
				if (h <= w) {
					float flen = w - h;
					float start, step, region_len;
					float idx, l_w, r_w;
					int l_idx, r_idx;
					T l_val = 0, r_val = 0;
					T grad = ldg(output_grad + index);
					if (c < start_num) {
						// start region
						flen = flen + 1;
						start = h - flen / 5.0;
						region_len = 2.0 * flen / 5.0;
						step = c;
						idx = start + step * region_len / (start_num - 1);
						l_idx = static_cast<int>(floorf(idx));
						r_idx = l_idx + 1;
						l_w = r_idx - idx;
						r_w = idx - l_idx;
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + l_idx * in_channel + t, l_w * grad);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + r_idx * in_channel + t, r_w * grad);
					}
					else if (c >= start_num + center_num) {
						// end region
						flen = flen + 1;
						start = w - flen / 5.0;
						region_len = 2.0 * flen / 5.0;
						step = c - (start_num + center_num);
						idx = start + step * region_len / (end_num - 1);
						l_idx = static_cast<int>(floorf(idx));
						r_idx = l_idx + 1;
						l_w = r_idx - idx;
						r_w = idx - l_idx;
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + l_idx * in_channel + t, l_w * grad);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + r_idx * in_channel + t, r_w * grad);
					}
					else {
						// action region
						start = h;
						step = c - start_num;
						region_len = flen;
						idx = start + step * region_len / (center_num - 1);
						l_idx = static_cast<int>(floorf(idx));
						r_idx = l_idx + 1;
						l_w = r_idx - idx;
						r_w = idx - l_idx;
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + l_idx * in_channel + t, l_w * grad);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + r_idx * in_channel + t, r_w * grad);
					}
				}
			}
		}

		template <typename T>
		__global__ void PropTcfgGradV2(
			const int nthreads,
			const int batch, const int tscale, const int in_channel,
			const T* output_grad, T* input_grad,
			const int start_num, const int center_num, const int end_num) {

			const int feature_num = start_num + center_num + end_num;
			CUDA_1D_KERNEL_LOOP(index, nthreads) {
				const int c = index % in_channel;
				const int w = (index / in_channel) % tscale;
				const int h = (index / in_channel / tscale) % tscale;
				const int t = (index / in_channel / tscale / tscale) % feature_num;
				const int n = index / in_channel / tscale / tscale / feature_num;
				if (h <= w) {
					float flen = w - h;
					float start, step, region_len;
					float idx, l_w, r_w;
					int l_idx, r_idx;
					T l_val = 0, r_val = 0;
					T grad = ldg(output_grad + index);
					if (t < start_num) {
						flen = flen + 1;
						// start region
						start = h - flen / 5.0;
						region_len = 2.0 * flen / 5.0;
						step = t;
						idx = start + step * region_len / (start_num - 1);
						l_idx = static_cast<int>(floorf(idx));
						r_idx = l_idx + 1;
						l_w = r_idx - idx;
						r_w = idx - l_idx;
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + l_idx * in_channel + c, l_w * grad);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + r_idx * in_channel + c, r_w * grad);
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
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + l_idx * in_channel + c, l_w * grad);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + r_idx * in_channel + c, r_w * grad);
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
						l_idx = tf_max(tf_min(l_idx, tscale - 1), 0);
						r_idx = tf_max(tf_min(r_idx, tscale - 1), 0);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + l_idx * in_channel + c, l_w * grad);
						CudaAtomicAdd(input_grad + n * tscale * in_channel + r_idx * in_channel + c, r_w * grad);
					}
				}
			}
		}

		template <typename T>
		struct PropTcfgGradFunctor<GPUDevice, T> {
			void operator()(const GPUDevice& d,
				int batch, int tscale, int in_channel,
				const T* output_grad, T* input_grad, int mode,
				int start_num, int center_num, int end_num) {

				CudaLaunchConfig config;
				int size;
				size = batch * tscale * in_channel;
				config = GetCudaLaunchConfig(size, d);
				SetZero << <config.block_count, config.thread_per_block, 0, d.stream() >> >(
					config.virtual_thread_count, input_grad);

				int feature_num = start_num + center_num + end_num;
				size = batch * in_channel * tscale * tscale * feature_num;
				config = GetCudaLaunchConfig(size, d);
				
				if (mode == 0) {
					PropTcfgGrad<T>
						<< <config.block_count, config.thread_per_block, 0, d.stream() >> > (
							config.virtual_thread_count,
							batch, tscale, in_channel, output_grad, input_grad,
							start_num, center_num, end_num);
				}
				else {
					PropTcfgGradV2<T>
						<< <config.block_count, config.thread_per_block, 0, d.stream() >> > (
							config.virtual_thread_count,
							batch, tscale, in_channel, output_grad, input_grad,
							start_num, center_num, end_num);
				}

			}
		};
		// Specify the kernels
		template struct PropTcfgFunctor<GPUDevice, float>;
		template struct PropTcfgGradFunctor<GPUDevice, float>;
	}	// namespace functor
}	// namespace tensorflow
#endif	// GOOGLE_CUDA