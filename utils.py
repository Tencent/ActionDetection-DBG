import json
import os

import numpy as np
import pandas as pd
import tqdm


def load_json(file):
    """
    :param file: json file path
    :return: data of json
    """
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_filter_video_names(video_info_file, gt_len_thres=0.98):
    """
    Select video according to length of ground truth
    :param video_info_file: json file path of video information
    :param gt_len_thres: max length of ground truth
    :return: list of video names
    """
    filter_video_names = []
    json_data = load_json(video_info_file)
    video_lists = list(json_data)
    for video_name in video_lists:
        video_info = json_data[video_name]
        if video_info['subset'] != "training":
            continue
        video_second = video_info["duration"]
        gt_lens = []
        video_labels = video_info["annotations"]
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = tmp_info["segment"][0]
            tmp_end = tmp_info["segment"][1]
            tmp_start = max(min(1, tmp_start / video_second), 0)
            tmp_end = max(min(1, tmp_end / video_second), 0)
            gt_lens.append(tmp_end - tmp_start)
        if len(gt_lens):
            mean_len = np.mean(gt_lens)
            if mean_len >= gt_len_thres:
                filter_video_names.append(video_name)
    return filter_video_names


def getDatasetDict(video_info_file, video_filter=False):
    """Load dataset file
    """
    json_data = load_json(video_info_file)

    # load filter video name
    filter_video_names = get_filter_video_names(video_info_file)

    database = json_data
    train_dict = {}
    val_dict = {}
    test_dict = {}
    video_lists = list(json_data.keys())
    for video_name in video_lists[:]:
        if video_filter and video_name in filter_video_names:
            continue
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


def gen_mask(tscale):
    """
    generator map mask
    :param tscale: temporal scale of feature
    :return: numpy array
    """
    mask = np.zeros([tscale, tscale], np.float32)
    for i in range(tscale):
        for j in range(i, tscale):
            mask[i, j] = 1
    return mask


def getProposalDataTest(video_list, dbg_config):
    """Load data during testing
    """
    tscale = dbg_config.tscale
    tgap = 1.0 / tscale
    data_dir = dbg_config.feat_dir

    batch_anchor_xmin = []
    batch_anchor_xmax = []
    batch_anchor_feature = []
    for i in range(len(video_list)):
        video_name = video_list[i]
        tmp_anchor_xmin = [tgap * i for i in range(tscale)]
        tmp_anchor_xmax = [tgap * i for i in range(1, tscale + 1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))
        batch_anchor_xmax.append(list(tmp_anchor_xmax))
        tmp_df = pd.read_csv(os.path.join(data_dir, video_name + ".csv"))
        video_feat = tmp_df.values[:, :]
        batch_anchor_feature.append(video_feat)
    batch_anchor_xmin = np.array(batch_anchor_xmin)
    batch_anchor_xmax = np.array(batch_anchor_xmax)
    batch_anchor_feature = np.array(batch_anchor_feature)
    batch_anchor_feature = np.reshape(
        batch_anchor_feature, [len(video_list), tscale, -1]
    )
    return batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature


def getFullData(video_dict, dbg_config, last_channel=True, training=True):
    tscale = dbg_config.tscale
    tgap = 1.0 / tscale
    data_dir = dbg_config.feat_dir

    gt_len_mode = 1
    gt_len_ratio = 2

    video_list = list(video_dict.keys())

    batch_anchor_feature = []
    batch_anchor_iou = []
    batch_label_action = []
    batch_label_start = []
    batch_label_end = []

    train_video_mean_len = []

    for i in range(len(video_list)):
        if i % 100 == 0:
            print("%d / %d videos are loaded" % (i, len(video_list)))
        video_name = video_list[i]
        video_info = video_dict[video_name]
        video_second = video_info["duration_second"]
        gt_lens = []
        bboxes = []
        video_labels = video_info["annotations"]
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = tmp_info["segment"][0]
            tmp_end = tmp_info["segment"][1]
            tmp_start = max(min(1, tmp_start / video_second), 0)
            tmp_end = max(min(1, tmp_end / video_second), 0)
            bboxes.append([tmp_start, tmp_end])
            gt_lens.append(tmp_end - tmp_start)

        # calculate gt average length
        mean_len = 2
        if len(gt_lens):
            mean_len = np.mean(gt_lens)
        if training:
            train_video_mean_len.append(mean_len)

        tmp_anchor_xmin = [tgap * i for i in range(tscale)]
        tmp_anchor_xmax = [tgap * i for i in range(1, tscale + 1)]

        # load feature
        tmp_df = pd.read_csv(os.path.join(data_dir, video_name + '.csv'))
        video_feat = tmp_df.values[:, :]
        if not last_channel:
            video_feat = np.transpose(video_feat, [1, 0])
        batch_anchor_feature.append(video_feat)

        # gen labels
        gt_bbox = np.array(bboxes)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        anchor_xmin = tmp_anchor_xmin
        anchor_xmax = tmp_anchor_xmax

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

        # gen iou_labels
        iou_labels = np.zeros([tscale, tscale])
        for i in range(tscale):
            for j in range(i, tscale):
                iou_labels[i, j] = np.max(
                    iou_with_anchors(i * tgap, (j + 1) * tgap, gt_xmins, gt_xmaxs))
        batch_anchor_iou.append(iou_labels)

    dataDict = {
        "gt_action": batch_label_action,
        "gt_start": batch_label_start,
        "gt_end": batch_label_end,
        "feature": batch_anchor_feature,
        "iou_label": batch_anchor_iou
    }
    if training:
        return dataDict, train_video_mean_len
    else:
        return dataDict


def getBatchListTest(video_dict, batch_size):
    """Generate batch list during testing
    """
    video_list = list(video_dict.keys())
    batch_start_list = [i * batch_size for i in range(len(video_list) // batch_size)]
    batch_start_list.append(len(video_list) - batch_size)
    batch_video_list = []
    for bstart in batch_start_list:
        batch_video_list.append(video_list[bstart: (bstart + batch_size)])
    return batch_video_list


def save_proposals_result(batch_video_list,
                          batch_result_xmin,
                          batch_result_xmax,
                          batch_result_iou,
                          batch_result_pstart,
                          batch_result_pend,
                          tscale, result_dir):
    """ Save proposal results to csv files
    """
    print('Saving results ...')
    columns = ["iou", "start", "end", "xmin", "xmax"]
    """for each batch video list
    """
    for idx in tqdm.tqdm(range(len(batch_video_list))):
        b_video = batch_video_list[idx]
        b_xmin = batch_result_xmin[idx]
        b_xmax = batch_result_xmax[idx]
        b_iou = batch_result_iou[idx]
        b_pstart = batch_result_pstart[idx]
        b_pend = batch_result_pend[idx]
        """for each video
        """
        for j in range(len(b_video)):
            tmp_video = b_video[j]
            tmp_xmin = b_xmin[j]
            tmp_xmax = b_xmax[j]
            tmp_iou = b_iou[j]
            tmp_pstart = b_pstart[j]
            tmp_pend = b_pend[j]
            res = []
            """ save all proposals result
            """
            for i in range(tscale):
                for j in range(i, tscale):
                    start = tmp_pstart[i]
                    end = tmp_pend[j]
                    iou = tmp_iou[i, j]
                    res.append([iou, start, end, tmp_xmin[i], tmp_xmax[j]])
            tmp_result = np.stack(res)
            tmp_df = pd.DataFrame(tmp_result, columns=columns)
            """ write csv file 
            """
            tmp_df.to_csv(os.path.join(result_dir, tmp_video + '.csv'), index=False)
