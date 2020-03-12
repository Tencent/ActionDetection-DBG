import torch.nn as nn
from torch.autograd import Function

import prop_tcfg_cuda

class PropTcfgFunction(Function):
    @staticmethod
    def forward(ctx, input, start_num=8, center_num=16, end_num=8):
        output = prop_tcfg_cuda.forward(input, start_num, center_num, end_num)
        ctx.start_num = start_num
        ctx.center_num = center_num
        ctx.end_num = end_num
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = prop_tcfg_cuda.backward(grad_output,
                                             ctx.start_num,
                                             ctx.center_num,
                                             ctx.end_num)
        return grad_input, None, None, None


class PropTcfg(nn.Module):
    def __init__(self, start_num=8, center_num=16, end_num=8):
        super(PropTcfg, self).__init__()
        self.start_num = start_num
        self.center_num = center_num
        self.end_num = end_num

    def forward(self, input):
        return PropTcfgFunction.apply(input,
                                      self.start_num, self.center_num, self.end_num)
