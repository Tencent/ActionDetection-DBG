import tensorflow as tf

from custom_op.prop_tcfg_op import prop_tcfg


def conv1d(x, out_channel, kernel_size, padding='same', act=tf.nn.relu, l2_scale=1e-3):
    """
    Construct conv1d operation
    :param x: input tensor
    :param out_channel: channel number of output tensor
    :param kernel_size: int
    :param padding: 'same' or 'valid'
    :param act: activate function
    :param l2_scale: the scale of L2 regulation for weights
    :return: output tensor
    """
    x = tf.layers.conv1d(x, out_channel, kernel_size, padding=padding,
                         activation=act,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale))
    return x


def conv2d(x, out_channel, kernel_size, padding='same', act=tf.nn.relu, l2_scale=1e-5):
    """
    Construct conv2d operation
    :param x: input tensor
    :param out_channel: channel number of output tensor
    :param kernel_size: int
    :param padding: 'same' or 'valid'
    :param act: activate function
    :param l2_scale: the scale of L2 regulation for weights
    :return: output tensor
    """
    x = tf.layers.conv2d(x, out_channel, kernel_size, padding=padding,
                         activation=act,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale))
    return x


def model(x, name='model', training=False):
    """
    Construct DBG model
    :param x: input tensor
    :param name: the name of variable scope
    :param training: bool
    :return: output tensors
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        """ Dual Stream BaseNet """
        x1, x2 = tf.split(x, 2, -1)
        x1 = conv1d(x1, 256, 3)
        x1 = conv1d(x1, 128, 3)
        x1_feat = x1
        x1 = conv1d(x1, 1, 1, act=tf.nn.sigmoid)

        x2 = conv1d(x2, 256, 3)
        x2 = conv1d(x2, 128, 3)
        x2_feat = x2
        x2 = conv1d(x2, 1, 1, act=tf.nn.sigmoid)

        xc = x1_feat + x2_feat
        xc_feat = xc
        xc = conv1d(xc, 1, 1, act=tf.nn.sigmoid)
        scores = (x1 + x2 + xc) / 3.0

        """ Action Completeness Regression """
        action_score = scores[:, :, :1]
        action_feat = prop_tcfg(action_score)  # B x 1 x T x T x 32
        action_feat = tf.squeeze(action_feat, axis=[1])  # B x T x T x 32

        net = conv2d(action_feat, 256, 1)
        net = tf.layers.dropout(net, 0.3, training=training)
        net = conv2d(net, 256, 1)
        net = tf.layers.dropout(net, 0.3, training=training)
        iou1 = conv2d(net, 1, 1, act=tf.nn.sigmoid)
        iou_mat = iou1

        """ Temporal Boundary Classification """
        net_feat = prop_tcfg(xc_feat, mode=1)  # B x 32 x T x T x 128
        net2 = tf.layers.conv3d(net_feat, 512, [32, 1, 1], activation=tf.nn.relu,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4))
        net2 = tf.squeeze(net2, [1])  # B x T x T x 512

        p2 = conv2d(net2, 256, 1)
        p2 = tf.layers.dropout(p2, 0.3, training=training)
        p2 = conv2d(p2, 256, 1)
        p2 = tf.layers.dropout(p2, 0.3, training=training)
        p2 = conv2d(p2, 2, 1, act=tf.nn.sigmoid)

        prop_start = p2[:, :, :, :1]
        prop_end = p2[:, :, :, 1:2]

        return scores, iou_mat, x1, x2, xc, prop_start, prop_end
