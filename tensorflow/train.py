import sys
sys.path.append('.')
import os

import numpy as np
import tensorflow as tf
import tqdm

import data_loader
from config_loader import dbg_config
import model
""" Load config
"""
checkpoint_dir = dbg_config.checkpoint_dir
batch_size = dbg_config.batch_size
learning_rate = dbg_config.learning_rate
tscale = dbg_config.tscale
feature_dim = dbg_config.feature_dim
epoch_num = dbg_config.epoch_num

""" Initialize map mask
"""
mask = data_loader.gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), -1)
mask = tf.convert_to_tensor(mask, tf.float32)


def binary_logistic_loss(gt_scores, pred_anchors, label_smoothing=0, weight_balance=True):
    """
    Calculate weighted binary logistic loss
    :param gt_scores: gt scores tensor
    :param pred_anchors: prediction score tensor
    :param label_smoothing: float
    :param weight_balance: bool
    :return: loss output tensor
    """
    gt_scores = tf.reshape(gt_scores, [-1])
    pred_anchors = tf.reshape(pred_anchors, [-1])

    pmask = tf.cast(gt_scores > 0.5, dtype=tf.float32)
    pmask = pmask * (1 - label_smoothing) + 0.5 * label_smoothing
    num_positive = tf.reduce_sum(pmask)

    num_entries = tf.cast(tf.shape(gt_scores)[0], dtype=tf.float32)

    ratio = num_entries / tf.maximum(num_positive, 1)
    coef_0 = 0.5 * (ratio) / (ratio - 1)
    coef_1 = coef_0 * (ratio - 1)
    pred_anchors = tf.clip_by_value(pred_anchors, 1e-10, 1.0)
    neg_pred_anchors = tf.clip_by_value(1.0 - pred_anchors, 1e-10, 1.0)
    if not weight_balance:
        coef_0 = coef_1 = 1.0
    loss = coef_1 * pmask * tf.log(pred_anchors) + \
           coef_0 * (1.0 - pmask) * tf.log(neg_pred_anchors)
    loss = -tf.reduce_mean(loss)
    return loss


def abs_smooth(x):
    """
    Smoothed absolute function. Useful to compute an L1 smooth error.
    :param x: input tensor
    :return: loss output tensor
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def IoU_loss(gt_iou, pred_iou):
    """
    Calculate IoU loss
    :param gt_iou: gt IoU tensor
    :param pred_iou: prediction IoU tensor
    :return: loss output tensor
    """
    u_hmask = tf.cast(gt_iou > 0.6, dtype=tf.float32)
    u_mmask = tf.cast(tf.logical_and(gt_iou <= 0.6, gt_iou > 0.2), dtype=tf.float32)
    u_lmask = tf.cast(gt_iou <= 0.2, dtype=tf.float32) * mask

    u_hmask = tf.reshape(u_hmask, [-1])
    u_mmask = tf.reshape(u_mmask, [-1])
    u_lmask = tf.reshape(u_lmask, [-1])

    num_h = tf.reduce_sum(u_hmask)
    num_m = tf.reduce_sum(u_mmask)
    num_l = tf.reduce_sum(u_lmask)

    r_m = 1.0 * num_h / (num_m)
    r_m = tf.minimum(r_m, 1)
    u_smmask = tf.random_uniform([tf.shape(u_hmask)[0]], dtype=tf.float32)
    u_smmask = u_smmask * u_mmask
    u_smmask = tf.cast(u_smmask > (1. - r_m), dtype=tf.float32)

    r_l = 2.0 * num_h / (num_l)
    r_l = tf.minimum(r_l, 1)
    u_slmask = tf.random_uniform([tf.shape(u_hmask)[0]], dtype=tf.float32)
    u_slmask = u_slmask * u_lmask
    u_slmask = tf.cast(u_slmask > (1. - r_l), dtype=tf.float32)

    iou_weights = (u_hmask + u_smmask + u_slmask)

    gt_iou = tf.reshape(gt_iou, [-1])
    pred_iou = tf.reshape(pred_iou, [-1])

    iou_loss = abs_smooth(gt_iou - pred_iou)

    loss = tf.losses.compute_weighted_loss(iou_loss, iou_weights)
    return loss

def DBG_Train(X_feature, Y_action, Y_start, Y_end, Y_iou, LR, training):
    """ Model and loss function of DBG
    """
    scores, iou_mat, r1, r2, r3, prop_start, prop_end = model.model(X_feature, training=training)

    loss = {}
    # calculate action loss
    loss['loss_action'] = binary_logistic_loss(Y_action, r1[:, :, 0]) + \
                          binary_logistic_loss(Y_action, r2[:, :, 0]) + \
                          binary_logistic_loss(Y_action, r3[:, :, 0])
    loss['loss_action'] /= 3.0

    # expand start and end label to map form
    gt_start = tf.expand_dims(Y_start, 2)
    gt_start = tf.tile(gt_start, [1, 1, tscale])
    gt_end = tf.expand_dims(Y_end, 1)
    gt_end = tf.tile(gt_end, [1, tscale, 1])
    tmp_mask = tf.tile(mask, [batch_size, 1, 1, 1])
    gt_idx = tf.where(tf.reshape(tmp_mask, [-1]) > 0)

    # calculate IoU loss
    iou_losses = []
    for i in range(batch_size):
        iou_loss = IoU_loss(Y_iou[i:i + 1], iou_mat[i:i + 1, :, :, :])
        iou_losses.append(iou_loss)
    iou_loss = tf.add_n(iou_losses) / batch_size
    loss['loss_iou'] = iou_loss * 10

    # calculate starting and ending map loss
    start_loss = binary_logistic_loss(
        tf.gather(tf.reshape(gt_start, [-1]), gt_idx),
        tf.gather(tf.reshape(prop_start, [-1]), gt_idx)
    )
    end_loss = binary_logistic_loss(
        tf.gather(tf.reshape(gt_end, [-1]), gt_idx),
        tf.gather(tf.reshape(prop_end, [-1]), gt_idx)
    )
    st_loss = start_loss + end_loss
    loss['loss_start'] = start_loss
    loss['loss_end'] = end_loss
    loss['loss_st'] = st_loss
    DBG_trainable_variables = tf.trainable_variables()

    # get l2 regularization loss
    l2 = tf.losses.get_regularization_loss()
    loss['l2'] = l2
    # total loss
    cost = 2 * loss["loss_action"] + loss['loss_iou'] + loss['loss_st'] + l2
    loss['cost'] = cost
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.\
            AdamOptimizer(learning_rate=LR).\
            minimize(cost, var_list=DBG_trainable_variables)

    return optimizer, loss, DBG_trainable_variables


if __name__ == "__main__":
    """ define the input and the network"""
    X_feature = tf.placeholder(tf.float32, shape=(batch_size, tscale, feature_dim))
    Y_action = tf.placeholder(tf.float32, shape=(batch_size, tscale))
    Y_start = tf.placeholder(tf.float32, shape=(batch_size, tscale))
    Y_end = tf.placeholder(tf.float32, shape=(batch_size, tscale))
    Y_iou = tf.placeholder(tf.float32, shape=(batch_size, tscale, tscale, 1))
    LR = tf.placeholder(tf.float32)
    train = tf.placeholder(tf.bool)
    optimizer, loss, DBG_trainable_variables = \
        DBG_Train(X_feature, Y_action, Y_start, Y_end, Y_iou, LR, train)

    """ init tf"""
    model_saver = tf.train.Saver(max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()

    """ load dataset """
    train_dict, val_dict, test_dict = data_loader.getDatasetDict(
        dbg_config.video_info_file, dbg_config.video_filter)
    train_data_dict, train_video_mean_len = data_loader.getFullData(train_dict, dbg_config)
    val_data_dict = data_loader.getFullData(val_dict, dbg_config, training=False)

    train_info = {"cost": [], "loss_action": [], "loss_start": [], "loss_end": [], "l2": [],
                  'loss_iou': [], 'loss_st': []}
    val_info = {"cost": [], "loss_action": [], "loss_start": [], "loss_end": [], "l2": [],
                'loss_iou': [], 'loss_st': []}
    info_keys = list(train_info.keys())
    best_val_cost = 1000000

    for epoch in range(0, epoch_num):
        """ Training"""
        batch_video_list = data_loader.getBatchList(train_dict, batch_size,
                                                    training=True,
                                                    train_video_mean_len=train_video_mean_len)

        mini_info = {"cost": [], "loss_action": [], "loss_start": [], "loss_end": [], "l2": [],
                     'loss_iou': [], 'loss_st': []}
        for idx in tqdm.tqdm(range(len(batch_video_list))):
            # get batch data
            batch_label_action, batch_label_start, batch_label_end, \
            batch_anchor_feature, batch_iou_label = data_loader.getBatchData(
                batch_video_list[idx], train_data_dict)

            _, out_loss = sess.run([optimizer, loss],
                                   feed_dict={X_feature: batch_anchor_feature,
                                              Y_action: batch_label_action,
                                              Y_start: batch_label_start,
                                              Y_end: batch_label_end,
                                              Y_iou: batch_iou_label,
                                              LR: learning_rate[epoch],
                                              train: True})
            for key in info_keys:
                mini_info[key].append(out_loss[key])
        for key in info_keys:
            train_info[key].append(np.mean(mini_info[key]))

        """ Validation"""
        batch_video_list = data_loader.getBatchList(val_dict, batch_size, training=False)

        mini_info = {"cost": [], "loss_action": [], "loss_start": [], "loss_end": [], "l2": [],
                     'loss_iou': [], 'loss_st': []}
        for idx in range(len(batch_video_list)):
            batch_label_action, batch_label_start, batch_label_end, \
            batch_anchor_feature, batch_iou_label = data_loader.getBatchData(
                batch_video_list[idx], val_data_dict)
            out_loss = sess.run(loss,
                                feed_dict={X_feature: batch_anchor_feature,
                                           Y_action: batch_label_action,
                                           Y_start: batch_label_start,
                                           Y_end: batch_label_end,
                                           Y_iou: batch_iou_label,
                                           LR: learning_rate[epoch],
                                           train: False})
            for key in info_keys:
                mini_info[key].append(out_loss[key])
        for key in info_keys:
            val_info[key].append(np.mean(mini_info[key]))

        # print loss information for each epoch
        print("Epoch-%d Train Loss: "
              "Action - %.05f, Start - %.05f, End - %.05f, IoU - %.05f, ST - %.05f, L2 - %.02f"
              % (epoch, train_info["loss_action"][-1], train_info["loss_start"][-1],
                 train_info["loss_end"][-1], train_info['loss_iou'][-1], train_info['loss_st'][-1],
                 train_info["l2"][-1]))
        print("Epoch-%d Val Loss: "
              "Action - %.05f, Start - %.05f, End - %.05f, IoU - %.05f, ST - %.05f"
              % (epoch, val_info["loss_action"][-1],
                 val_info["loss_start"][-1], val_info["loss_end"][-1],
                 val_info['loss_iou'][-1], val_info['loss_st'][-1]))

        train_cost = train_info["loss_action"][-1] + \
                     train_info['loss_iou'][-1] + \
                     train_info['loss_st'][-1]
        val_cost = val_info["loss_action"][-1] +\
                   val_info['loss_iou'][-1] + \
                   val_info['loss_st'][-1]
        print("Train Cost: %.05f, Val Cost: %.05f" % (train_cost, val_cost))
        """ save model """
        model_saver.save(sess,
                         os.path.join(checkpoint_dir, 'dbg_model_checkpoint'), global_step=epoch)
        if val_info["cost"][-1] < best_val_cost and epoch >= epoch_num // 2:
            best_val_cost = val_info["cost"][-1]
            print('save current best model, loss: %f' % best_val_cost)
            model_saver.save(sess, os.path.join(checkpoint_dir, 'dbg_model_best'))
