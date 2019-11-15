import tensorflow as tf
from prop_tcfg_op import prop_tcfg
import numpy as np

"""
x = np.ones([1, 100, 3], np.float32)
x[:, :, 1] *= 2.0
x[:, :, 2] *= 3.0
x = tf.convert_to_tensor(x, tf.float32)
y = prop_tcfg(x)
print(y)
sess = tf.Session()
out = sess.run(y)
print(out[0, 0, 1, 99])
print(out[0, 0, 1, 2])
"""

shape = [1, 10, 3]
d = np.random.rand(*shape)
inp = tf.constant(d, tf.float32)
inp = tf.nn.sigmoid(inp)
y = prop_tcfg(inp)
print(y)
with tf.Session() as sess:
	inp_grad_err = tf.test.compute_gradient_error(
		inp, shape, y, [1, 3, 10, 10, 32])
	print(inp_grad_err)

