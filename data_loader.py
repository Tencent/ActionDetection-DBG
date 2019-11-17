import json
<<<<<<< HEAD
<<<<<<< HEAD
import os
=======
>>>>>>> update DBG
=======
import os
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
import random

import numpy as np
import pandas as pd

from config import DBGConfig

dbg_config = DBGConfig()

tscale = dbg_config.tscale
tgap = 1.0 / tscale

video_info_file = dbg_config.video_info_file
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
video_filter = dbg_config.video_filter
video_filter_file = dbg_config.video_filter_file
data_dir = dbg_config.feat_dir
iou_label_dir = dbg_config.iou_label_dir
data_aug = dbg_config.data_aug

train_video_mean_len = []

gt_len_mode = 1
gt_len_ratio = 2
<<<<<<< HEAD
=======
data_dir = dbg_config.feat_dir
>>>>>>> update DBG
=======
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def getDatasetDict():
    """Load dataset file
    """
    json_data = load_json(video_info_file)

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
    # load filter video name
    filter_df = pd.read_csv(video_filter_file)
    filter_video_names = filter_df.video_name.values[:]
    filter_video_names = set(filter_video_names)

<<<<<<< HEAD
=======
>>>>>>> update DBG
=======
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
    database = json_data
    train_dict = {}
    val_dict = {}
    test_dict = {}
    video_lists = list(json_data.keys())
    for video_name in video_lists[:]:
<<<<<<< HEAD
<<<<<<< HEAD
        if video_filter and video_name in filter_video_names:
            continue
=======
>>>>>>> update DBG
=======
        if video_filter and video_name in filter_video_names:
            continue
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
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


<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
def getBatchList(numVideo, batch_size, shuffle=True):
    """Generate batch list for each epoch randomly
    """
    video_list = np.arange(numVideo)
    batch_start_list = [i * batch_size for i in range(len(video_list) // batch_size)]
    batch_start_list.append(len(video_list) - batch_size)
    if shuffle == True:
        random.shuffle(video_list)
    batch_video_list = []
    for bstart in batch_start_list:
        batch_video_list.append(video_list[bstart: (bstart + batch_size)])
    return batch_video_list


def getBatchListTrain(numVideo, batch_size):
    """ Generate batch list during training
    """
    video_list = np.arange(numVideo)
    assert len(video_list) == len(train_video_mean_len)

    if data_aug:
        # add train video with short proposals
        add_list = np.where(np.array(train_video_mean_len) < 0.2)
        add_list = np.reshape(add_list, [-1])
        np.random.shuffle(add_list)

        video_list = np.concatenate([video_list, add_list[:]], 0)

    batch_start_list = [i * batch_size for i in range(len(video_list) // batch_size)]
    batch_start_list.append(len(video_list) - batch_size)
    random.shuffle(video_list)
    batch_video_list = []
    for bstart in batch_start_list:
        batch_video_list.append(video_list[bstart: (bstart + batch_size)])
    return batch_video_list


<<<<<<< HEAD
=======
>>>>>>> update DBG
=======
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
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


<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
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
    return batch_label_action, batch_label_start, batch_label_end, batch_anchor_feature, batch_iou_label


def getFullData(dataSet):
    """Load full data in dataset
    """
    train_dict, val_dict, test_dict = getDatasetDict()
    if dataSet == "train":
        video_dict = train_dict
    else:
        video_dict = val_dict
    video_list = list(video_dict.keys())
    batch_bbox = []
    batch_index = [0]
    batch_anchor_xmin = []
    batch_anchor_xmax = []
    batch_anchor_feature = []
    batch_anchor_iou = []
    for i in range(len(video_list)):
        if i % 100 == 0:
            print("%d / %d %s videos are loaded" % (i, len(video_list), dataSet))
        video_name = video_list[i]
        video_info = video_dict[video_name]
        video_second = video_info["duration_second"]
        corrected_second = video_second
        gt_lens = []
        video_labels = video_info["annotations"]
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = tmp_info["segment"][0]
            tmp_end = tmp_info["segment"][1]
            tmp_start = max(min(1, tmp_start / corrected_second), 0)
            tmp_end = max(min(1, tmp_end / corrected_second), 0)
            batch_bbox.append([tmp_start, tmp_end])
            gt_lens.append(tmp_end - tmp_start)

        # calculate gt average length
        mean_len = 2
        if len(gt_lens):
            mean_len = np.mean(gt_lens)
        if dataSet == "train":
            train_video_mean_len.append(mean_len)

        tmp_anchor_xmin = [tgap * i for i in range(tscale)]
        tmp_anchor_xmax = [tgap * i for i in range(1, tscale + 1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))
        batch_anchor_xmax.append(list(tmp_anchor_xmax))
        batch_index.append(batch_index[-1] + len(video_labels))

        # load feature
        tmp_df = pd.read_csv(os.path.join(data_dir, video_name + '.csv'))
        video_feat = tmp_df.values[:, :]
        batch_anchor_feature.append(video_feat)

        # load iou label
        tmp_df2 = pd.read_csv(os.path.join(iou_label_dir, video_name + '.csv'))
        iou_label = tmp_df2.values[:, :]
        batch_anchor_iou.append(iou_label)

    num_data = len(batch_anchor_feature)
    batch_label_action = []
    batch_label_start = []
    batch_label_end = []

    for idx in range(num_data):
        gt_bbox = np.array(batch_bbox[batch_index[idx]: batch_index[idx + 1]])
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        anchor_xmin = batch_anchor_xmin[idx]
        anchor_xmax = batch_anchor_xmax[idx]

        gt_lens = gt_xmaxs - gt_xmins
        if gt_len_mode == 0:
            gt_len_small = np.maximum(tgap, 0.1 * gt_lens)
        else:
            gt_len_small = tgap * gt_len_ratio

        gt_start_bboxs = np.stack(
            (gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1
        )
        gt_end_bboxs = np.stack(
            (gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1
        )
        match_score_action = []
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(
                np.max(
                    ioa_with_anchors(
                        anchor_xmin[jdx], anchor_xmax[jdx], gt_xmins, gt_xmaxs
                    )
                )
            )

        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(
                np.max(
                    ioa_with_anchors(
                        anchor_xmin[jdx],
                        anchor_xmax[jdx],
                        gt_start_bboxs[:, 0],
                        gt_start_bboxs[:, 1],
                    )
                )
            )
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(
                np.max(
                    ioa_with_anchors(
                        anchor_xmin[jdx],
                        anchor_xmax[jdx],
                        gt_end_bboxs[:, 0],
                        gt_end_bboxs[:, 1],
                    )
                )
            )
        batch_label_action.append(match_score_action)
        batch_label_start.append(match_score_start)
        batch_label_end.append(match_score_end)

    dataDict = {
        "gt_action": batch_label_action,
        "gt_start": batch_label_start,
        "gt_end": batch_label_end,
        "feature": batch_anchor_feature,
        "iou_label": batch_anchor_iou
    }
    return dataDict


<<<<<<< HEAD
=======
>>>>>>> update DBG
=======
>>>>>>> d0ce64a2678aa11de7d2698263df36b082bfae09
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
