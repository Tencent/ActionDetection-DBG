#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int prop_tcfg_cuda_forward(
        const at::Tensor input,
        at::Tensor output,
        int start_num,
        int center_num,
        int end_num);

int prop_tcfg_cuda_backward(
        const at::Tensor grad_output,
        at::Tensor grad_input,
        int start_num,
        int center_num,
        int end_num);

at::Tensor prop_tcfg_forward(
        at::Tensor input,
        int start_num,
        int center_num,
        int end_num) {
    CHECK_INPUT(input);
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int t_dim = input.size(2);
    const int number_dim = start_num + center_num + end_num;

    auto output = torch::zeros({batch_size, channels, number_dim, t_dim, t_dim}, input.options());

    prop_tcfg_cuda_forward(input, output, start_num, center_num, end_num);
    return output;
}

at::Tensor prop_tcfg_backward(
        at::Tensor grad_output,
        int start_num,
        int center_num,
        int end_num) {
    CHECK_INPUT(grad_output);
    const int batch_size = grad_output.size(0);
    const int channels = grad_output.size(1);
    const int t_dim = grad_output.size(3);

    auto grad_input = torch::zeros({batch_size, channels, t_dim}, grad_output.options());

    prop_tcfg_cuda_backward(grad_output, grad_input, start_num, center_num, end_num);
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &prop_tcfg_forward, "Proposal feature generation forward (CUDA)");
  m.def("backward", &prop_tcfg_backward, "Proposal feature generation backward (CUDA)");
}