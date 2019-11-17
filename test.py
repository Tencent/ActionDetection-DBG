import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

import data_loader
import model
from config import DBGConfig

dbg_config = DBGConfig()

checkpoint_dir = dbg_config.checkpoint_dir
result_dir = dbg_config.result_dir

tscale = dbg_config.tscale
feature_dim = dbg_config.feature_dim

# test batch size
batch_size = dbg_config.test_batch_size
test_mode = dbg_config.test_mode

mask = np.zeros([tscale, tscale], np.float32)
for i in range(tscale):
    for j in range(i, tscale):
        mask[i, j] = 1
tf_mask = tf.convert_to_tensor(mask, tf.float32)
tf_mask = tf.reshape(tf_mask, [1, tscale, tscale, 1])

if __name__ == "__main__":
    X_feature = tf.placeholder(tf.float32, shape=(batch_size, tscale, feature_dim))
    scores, iou_mat, x1, x2, xc, prop_start, prop_end = model.model(X_feature, training=False)

    prop_start = prop_start * tf_mask
    prop_end = prop_end * tf_mask

    # boundary map fusion
    pstart = tf.reduce_sum(prop_start, 2) / tf.maximum(tf.reduce_sum(tf_mask, 2), 1)
    pend = tf.reduce_sum(prop_end, 1) / tf.maximum(tf.reduce_sum(tf_mask, 1), 1)

<<<<<<< HEAD
    model_saver = tf.train.Saver()
=======
<<<<<<< HEAD
<<<<<<< HEAD
    model_saver = tf.train.Saver(max_to_keep=80)
=======
    model_saver = tf.train.Saver()
>>>>>>> update DBG
=======
    model_saver = tf.train.Saver(max_to_keep=80)
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
>>>>>>> 866c21db4d01ab64bb8ef60fb1f456c2b0c8f380
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()
    model_saver.restore(sess, os.path.join(checkpoint_dir, 'dbg_model_best'))

    train_dict, val_dict, test_dict = data_loader.getDatasetDict()
    if test_mode == 'validation':
        video_dict = val_dict
    else:
        video_dict = test_dict

    batch_video_list = data_loader.getBatchListTest(video_dict, batch_size, shuffle=False)

    batch_result_xmin = []
    batch_result_xmax = []
    batch_result_iou = []
    batch_result_pstart = []
    batch_result_pend = []

    print('Runing DBG model ...')
    for idx in tqdm.tqdm(range(len(batch_video_list))):
        batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature = data_loader.getProposalDataTest(
            batch_video_list[idx], video_dict)
        out_iou, out_start, out_end = sess.run([iou_mat, pstart, pend],
                                               feed_dict={X_feature: batch_anchor_feature})
        batch_result_xmin.append(batch_anchor_xmin)
        batch_result_xmax.append(batch_anchor_xmax)
        batch_result_iou.append(out_iou[:, :, :, 0])
        batch_result_pstart.append(out_start[:, :, 0])
        batch_result_pend.append(out_end[:, :, 0])

    print('Saving results ...')
    columns = ["iou", "start", "end", "xmin", "xmax"]

    for idx in tqdm.tqdm(range(len(batch_video_list))):
        b_video = batch_video_list[idx]
        b_xmin = batch_result_xmin[idx]
        b_xmax = batch_result_xmax[idx]
        b_iou = batch_result_iou[idx]
        b_pstart = batch_result_pstart[idx]
        b_pend = batch_result_pend[idx]

        for j in range(len(b_video)):
            tmp_video = b_video[j]
            tmp_xmin = b_xmin[j]
            tmp_xmax = b_xmax[j]
            tmp_iou = b_iou[j]
            tmp_pstart = b_pstart[j]
            tmp_pend = b_pend[j]
            res = []

            # save all proposals result
            for i in range(tscale):
                for j in range(i, tscale):
                    start = tmp_pstart[i]
                    end = tmp_pend[j]
                    iou = tmp_iou[i, j]
                    res.append([iou, start, end, tmp_xmin[i], tmp_xmax[j]])
            tmp_result = np.stack(res)
            tmp_df = pd.DataFrame(tmp_result, columns=columns)
            tmp_df.to_csv(os.path.join(result_dir, tmp_video + '.csv'), index=False)
