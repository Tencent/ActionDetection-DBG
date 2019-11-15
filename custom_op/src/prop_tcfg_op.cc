#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/types.h"
#include <stdio.h>
#include "prop_tcfg.h"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("PropTcfg")
.Attr("T: {float, double} = DT_FLOAT")
.Input("input: T")
.Attr("mode: int = 0")
.Attr("start_num: int = 8")
.Attr("center_num: int = 16")
.Attr("end_num: int = 8")
.Output("output: T")
.SetShapeFn([](InferenceContext* c) {
	ShapeHandle input;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
	int mode;
	TF_RETURN_IF_ERROR(c->GetAttr("mode", &mode));
	int start_num, center_num, end_num;
	TF_RETURN_IF_ERROR(c->GetAttr("start_num", &start_num));
	TF_RETURN_IF_ERROR(c->GetAttr("center_num", &center_num));
	TF_RETURN_IF_ERROR(c->GetAttr("end_num", &end_num));

	DimensionHandle batch_dim = c->Dim(input, 0);
	DimensionHandle t_dim = c->Dim(input, 1);
	DimensionHandle channel_dim = c->Dim(input, 2);
	DimensionHandle number_dim = c->MakeDim(start_num + center_num + end_num);

	if (mode == 0) {
		c->set_output(0, c->MakeShape({ batch_dim, channel_dim, t_dim, t_dim, number_dim }));
	}
	else {
		c->set_output(0, c->MakeShape({ batch_dim, number_dim, t_dim, t_dim, channel_dim }));
	}
	
	return Status::OK();
});

REGISTER_OP("PropTcfgGrad")
.Attr("T: {float, double} = DT_FLOAT")
.Input("input: T")
.Input("output_grad: T")
.Attr("mode: int = 0")
.Attr("start_num: int = 8")
.Attr("center_num: int = 16")
.Attr("end_num: int = 8")
.Output("input_grad: T")
.SetShapeFn([](InferenceContext* c) {
	c->set_output(0, c->input(0));
	return Status::OK();
});

namespace tensorflow {
	typedef Eigen::ThreadPoolDevice CPUDevice;
	typedef Eigen::GpuDevice GPUDevice;

	// define class PropTcfgOp
	template <typename Device, typename T>
	class PropTcfgOp : public OpKernel {
		int mode;
		int start_num, center_num, end_num;
	public:
		explicit PropTcfgOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));
			OP_REQUIRES_OK(context, context->GetAttr("start_num", &start_num));
			OP_REQUIRES_OK(context, context->GetAttr("center_num", &center_num));
			OP_REQUIRES_OK(context, context->GetAttr("end_num", &end_num));
		}

		void Compute(OpKernelContext* context) override {
			const Tensor& input_tensor = context->input(0);
			OP_REQUIRES(context, input_tensor.dims() == 3,
				errors::InvalidArgument("input must be at 3-D, got shape", input_tensor.shape().DebugString()));
			int dims[3];
			for (int i = 0; i < 3; i++) {
				dims[i] = static_cast<int>(input_tensor.dim_size(i));
			}

			int feature_num = start_num + center_num + end_num;
			OP_REQUIRES(context, feature_num > 0,
				errors::InvalidArgument("output feature number must > 0"));

			int output_dims[] = { dims[0], dims[2], dims[1], dims[1], feature_num };
			if (mode != 0) {
				output_dims[1] = feature_num;
				output_dims[4] = dims[2];
			}
			
			TensorShape output_shape;
			TensorShapeUtils::MakeShape(output_dims, 5, &output_shape);
			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
			if (input_tensor.NumElements() > 0) {
				functor::PropTcfgFunctor<Device, T>()(
					context->eigen_device<Device>(),
					dims[0], dims[1], dims[2],
					input_tensor.flat<T>().data(),
					output_tensor->flat<T>().data(),
					mode,
					start_num, center_num, end_num);
			}
		}

	};	// class PropTcfgOp

		// define class PropTcfgGradOp
	template <typename Device, typename T>
	class PropTcfgGradOp : public OpKernel {
		int mode;
		int start_num, center_num, end_num;
	public:
		explicit PropTcfgGradOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));
			OP_REQUIRES_OK(context, context->GetAttr("start_num", &start_num));
			OP_REQUIRES_OK(context, context->GetAttr("center_num", &center_num));
			OP_REQUIRES_OK(context, context->GetAttr("end_num", &end_num));
		}

		void Compute(OpKernelContext* context) override {
			const Tensor& input_tensor = context->input(0);
			const Tensor& output_grad_tensor = context->input(1);
			Tensor* input_grad_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &input_grad_tensor));
			int dims[3];
			for (int i = 0; i < 3; i++) {
				dims[i] = static_cast<int>(input_tensor.dim_size(i));
			}
			if (input_tensor.NumElements() > 0) {
				functor::PropTcfgGradFunctor<Device, T>()(
					context->eigen_device<Device>(),
					dims[0], dims[1], dims[2],
					output_grad_tensor.flat<T>().data(),
					input_grad_tensor->flat<T>().data(),
					mode,
					start_num, center_num, end_num);
			}
		}
	};	// class PropTcfgGradOp

#if GOOGLE_CUDA
	namespace functor {
		extern template struct PropTcfgFunctor<GPUDevice, float>;
	}
	REGISTER_KERNEL_BUILDER(Name("PropTcfg").Device(DEVICE_GPU).TypeConstraint<float>("T"),
		PropTcfgOp<GPUDevice, float>);

	namespace functor {
		extern template struct PropTcfgGradFunctor<GPUDevice, float>;
	}
	REGISTER_KERNEL_BUILDER(Name("PropTcfgGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),
		PropTcfgGradOp<GPUDevice, float>);
#endif	// GOOGLE CUDA
}	// namespace tensorflow