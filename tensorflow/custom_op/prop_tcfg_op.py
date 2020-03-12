import tensorflow as tf
from tensorflow.python.framework import ops
import os.path as osp

""" Load operation library """
filename = osp.join(osp.dirname(__file__), 'prop_tcfg.so')
_prop_tcfg_module = tf.load_op_library(filename)
_prop_tcfg = _prop_tcfg_module.prop_tcfg
_prop_tcfg_grad = _prop_tcfg_module.prop_tcfg_grad

def prop_tcfg(x, mode=0, start_num=8, center_num=16, end_num=8):
    """
    Define proposal generation layer
    :param x: input tensor
    :param mode: 0 or 1, 0: output shape is BxCxTxTxN, 1: output shape is BxNxTxTxC
    :param start_num: sample num in starting region
    :param center_num: sample num in center region
    :param end_num: sample num in ending region
    :return: output tensor
    """
    x = _prop_tcfg(x, mode, start_num, center_num, end_num)
    return x

@ops.RegisterGradient("PropTcfg")
def prop_tcfg_grad(op, grad):
    """
    Define backward for proposal generation layer
    :param op: context op
    :param grad: output grad
    :return: list of input grad
    """
    inp = op.inputs[0]
    mode = op.get_attr('mode')
    start_num = op.get_attr('start_num')
    center_num = op.get_attr('center_num')
    end_num = op.get_attr('end_num')
    # compute gradient
    inp_grad = _prop_tcfg_grad(inp, grad, mode, start_num, center_num, end_num)

    return [inp_grad]
