import sys

sys.path.append('.')
import os

import tensorflow as tf
import tqdm

import data_loader
import utils
import model
from config_loader import dbg_config

""" Get checkpoint, result dir
"""
checkpoint_dir = dbg_config.checkpoint_dir
result_dir = dbg_config.result_dir
""" Get test scale, feature dimension
"""
tscale = dbg_config.tscale
feature_dim = dbg_config.feature_dim
""" Get video infor file 
"""
video_info_file = dbg_config.video_info_file
""" Get test batch size, test_mode
"""
batch_size = dbg_config.test_batch_size
test_mode = dbg_config.test_mode

""" Generate map mask """
mask = data_loader.gen_mask(tscale)
tf_mask = tf.convert_to_tensor(mask, tf.float32)
tf_mask = tf.reshape(tf_mask, [1, tscale, tscale, 1])

"""
This test script is used for evaluating our algorithm 
This script saves all proposals results (csv format)
Then, use post_processing.py to generate the final result
Finally, use eval.py to evaluate the final result
You can got about 68% AUC
"""

""" 
Testing procedure
1.Get Test data
2.Define DBG model
3.Load model weights 
4.Run DBG model
5.Save proposal results (csv format)
"""

if __name__ == "__main__":
    """ Define DBG model """
    X_feature = tf.placeholder(tf.float32, shape=(batch_size, tscale, feature_dim))
    scores, iou_mat, x1, x2, xc, prop_start, prop_end = model.model(X_feature, training=False)

    """ Boundary map fusion """
    prop_start = prop_start * tf_mask
    prop_end = prop_end * tf_mask
    pstart = tf.reduce_sum(prop_start, 2) / tf.maximum(tf.reduce_sum(tf_mask, 2), 1)
    pend = tf.reduce_sum(prop_end, 1) / tf.maximum(tf.reduce_sum(tf_mask, 1), 1)

    """ Load model weights """
    model_saver = tf.train.Saver()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()
    model_saver.restore(sess, os.path.join(checkpoint_dir, 'dbg_model_best'))

    """ Get test or validation video list 
    """
    train_dict, val_dict, test_dict = data_loader.getDatasetDict(video_info_file)
    if test_mode == 'validation':
        video_dict = val_dict
    else:
        video_dict = test_dict
    """ load test or validation data
    """
    batch_video_list = data_loader.getBatchListTest(video_dict, batch_size)
    """ init result list
    """
    batch_result_xmin = []
    batch_result_xmax = []
    batch_result_iou = []
    batch_result_pstart = []
    batch_result_pend = []

    """ Run DBG model 
    """
    print('Runing DBG model ...')
    for idx in tqdm.tqdm(range(len(batch_video_list))):
        """ Get batch data 
        """
        batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature = \
            data_loader.getProposalDataTest(
                batch_video_list[idx], dbg_config)
        """ Run batch data 
        """
        out_iou, out_start, out_end = sess.run([iou_mat, pstart, pend],
                                               feed_dict={X_feature: batch_anchor_feature})
        batch_result_xmin.append(batch_anchor_xmin)
        batch_result_xmax.append(batch_anchor_xmax)
        batch_result_iou.append(out_iou[:, :, :, 0])
        batch_result_pstart.append(out_start[:, :, 0])
        batch_result_pend.append(out_end[:, :, 0])

    utils.save_proposals_result(batch_video_list,
                                batch_result_xmin,
                                batch_result_xmax,
                                batch_result_iou,
                                batch_result_pstart,
                                batch_result_pend,
                                tscale, result_dir)
