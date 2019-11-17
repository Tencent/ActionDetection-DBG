import json
import random

import numpy as np
import pandas as pd

from config import DBGConfig

dbg_config = DBGConfig()

tscale = dbg_config.tscale
tgap = 1.0 / tscale

video_info_file = dbg_config.video_info_file
data_dir = dbg_config.feat_dir


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def getDatasetDict():
    """Load dataset file
    """
    json_data = load_json(video_info_file)

    database = json_data
    train_dict = {}
    val_dict = {}
    test_dict = {}
    video_lists = list(json_data.keys())
    for video_name in video_lists[:]:
        video_info = database[video_name]
        video_new_info = {}
        video_new_info["duration_second"] = video_info["duration"]
        video_subset = video_info['subset']
        video_new_info["annotations"] = video_info["annotations"]
        if video_subset == "training":
            train_dict[video_name] = video_new_info
        elif video_subset == "validation":
            val_dict[video_name] = video_new_info
        elif video_subset == "testing":
            test_dict[video_name] = video_new_info
    return train_dict, val_dict, test_dict


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.0)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)

    inter_len = np.maximum(int_xmax - int_xmin, 0.)

    union_len = len_anchors - inter_len + box_max - box_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def getBatchListTest(video_dict, batch_size, shuffle=True):
    """Generate batch list during testing
    """
    video_list = list(video_dict.keys())
    batch_start_list = [i * batch_size for i in range(len(video_list) // batch_size)]
    batch_start_list.append(len(video_list) - batch_size)
    if shuffle == True:
        random.shuffle(video_list)
    batch_video_list = []
    for bstart in batch_start_list:
        batch_video_list.append(video_list[bstart: (bstart + batch_size)])
    return batch_video_list


def getProposalDataTest(video_list, video_dict):
    """Load data during testing
    """
    batch_anchor_xmin = []
    batch_anchor_xmax = []
    batch_anchor_feature = []
    for i in range(len(video_list)):
        video_name = video_list[i]
        tmp_anchor_xmin = [tgap * i for i in range(tscale)]
        tmp_anchor_xmax = [tgap * i for i in range(1, tscale + 1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))
        batch_anchor_xmax.append(list(tmp_anchor_xmax))
        tmp_df = pd.read_csv(data_dir + video_name + ".csv")
        video_feat = tmp_df.values[:, :]
        batch_anchor_feature.append(video_feat)
    batch_anchor_xmin = np.array(batch_anchor_xmin)
    batch_anchor_xmax = np.array(batch_anchor_xmax)
    batch_anchor_feature = np.array(batch_anchor_feature)
    batch_anchor_feature = np.reshape(
        batch_anchor_feature, [len(video_list), tscale, -1]
    )
    return batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature
