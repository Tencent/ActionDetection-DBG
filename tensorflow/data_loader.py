import random

from config_loader import dbg_config
from utils import *

""" Load config"""
""" get input feature temporal scale """
tscale = dbg_config.tscale
tgap = 1.0 / tscale
""" get video information json file path """
video_info_file = dbg_config.video_info_file
""" set filter videos or not """
video_filter = dbg_config.video_filter
""" get feature directory """
data_dir = dbg_config.feat_dir
""" set data augmentation or not """
data_aug = dbg_config.data_aug

def getBatchList(video_dict, batch_size,
                 training=False, train_video_mean_len=None):
    """Generate batch list
    """
    video_list = list(video_dict.keys())
    video_idxs = np.arange(len(video_list))
    if training and data_aug:
        assert len(video_idxs) == len(train_video_mean_len)
        # add train video with short proposals
        add_list = np.where(np.array(train_video_mean_len) < 0.2)
        add_list = np.reshape(add_list, [-1])
        np.random.shuffle(add_list)

        video_idxs = np.concatenate([video_idxs, add_list[:]], 0)

        random.shuffle(video_idxs)

    batch_start_list = [i * batch_size for i in range(len(video_idxs) // batch_size)]
    batch_start_list.append(len(video_idxs) - batch_size)
    batch_video_list = []
    for bstart in batch_start_list:
        batch_video_list.append(video_idxs[bstart: (bstart + batch_size)])
    return batch_video_list


def getBatchData(video_list, data_dict):
    """Given a video list (batch), get corresponding data
    """
    batch_label_action = []
    batch_label_start = []
    batch_label_end = []
    batch_anchor_feature = []
    batch_iou_label = []
    for idx in video_list:
        batch_label_action.append(data_dict["gt_action"][idx])
        batch_label_start.append(data_dict["gt_start"][idx])
        batch_label_end.append(data_dict["gt_end"][idx])
        batch_anchor_feature.append(data_dict["feature"][idx])
        batch_iou_label.append(data_dict['iou_label'][idx])

    batch_label_action = np.array(batch_label_action)
    batch_label_start = np.array(batch_label_start)
    batch_label_end = np.array(batch_label_end)
    batch_anchor_feature = np.array(batch_anchor_feature)
    batch_anchor_feature = np.reshape(
        batch_anchor_feature, [len(video_list), tscale, -1]
    )
    batch_iou_label = np.expand_dims(np.stack(batch_iou_label), -1)
    return batch_label_action, batch_label_start, batch_label_end, \
           batch_anchor_feature, batch_iou_label
