import json
import os

import numpy as np
import pandas as pd

from config import DBGConfig

dbg_config = DBGConfig()

tscale = dbg_config.tscale
tgap = 1.0 / tscale

iou_label_dir = dbg_config.iou_label_dir
video_info_file = dbg_config.video_info_file

if not os.path.exists((iou_label_dir)):
    os.makedirs(iou_label_dir)


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


def getLabelData(dataSet):
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

        tmp_anchor_xmin = [tgap * i for i in range(tscale)]
        tmp_anchor_xmax = [tgap * i for i in range(1, tscale + 1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))
        batch_anchor_xmax.append(list(tmp_anchor_xmax))
        batch_index.append(batch_index[-1] + len(video_labels))

    num_data = len(video_list)

    for idx in range(num_data):
        gt_bbox = np.array(batch_bbox[batch_index[idx]: batch_index[idx + 1]])
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]

        # gen iou_labels
        iou_labels = np.zeros([tscale, tscale])
        for i in range(tscale):
            for j in range(i, tscale, 1):
                iou_labels[i, j] = np.max(iou_with_anchors(i * tgap, (j + 1) * tgap, gt_xmins, gt_xmaxs))

        df = pd.DataFrame(iou_labels)
        df.to_csv(os.path.join(iou_label_dir, video_list[idx] + '.csv'), index=False)


if __name__ == "__main__":
    getLabelData("train")
    getLabelData("validation")
