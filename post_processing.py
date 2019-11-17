import argparse
import json
import multiprocessing as mp
import os
import threading

import numpy as np
import pandas as pd
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', type=str)
parser.add_argument('output_file', type=str)
parser.add_argument('top_number', type=int, nargs='?', default=100)
parser.add_argument('-t', '--thread', type=int, nargs='?', default=8)
parser.add_argument('-m', '--mode', type=str, nargs='?', default='validation')
args = parser.parse_args()

video_info_file = 'data/video_info_19993.json'
top_number = args.top_number
thread_num = args.thread


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


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def NMS(df, nms_threshold):
    df = df.sort(columns="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    rstart = []
    rend = []
    rscore = []
    while len(tstart) > 1 and len(rscore) < top_number:
        idx = 1
        while idx < len(tstart):
            if IOU(tstart[0], tend[0], tstart[idx], tend[idx]) > nms_threshold:
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
            else:
                idx += 1
        rstart.append(tstart[0])
        rend.append(tend[0])
        rscore.append(tscore[0])
        tstart.pop(0)
        tend.pop(0)
        tscore.pop(0)
    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def softNMS(df):
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []
    while len(tscore) > 1 and len(rscore) < top_number:
        max_index = tscore.index(max(tscore))
        tmp_start = tstart[max_index]
        tmp_end = tend[max_index]
        tmp_score = tscore[max_index]
        rstart.append(tmp_start)
        rend.append(tmp_end)
        rscore.append(tmp_score)
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

        tstart = np.array(tstart)
        tend = np.array(tend)
        tscore = np.array(tscore)

        tt1 = np.maximum(tmp_start, tstart)
        tt2 = np.minimum(tmp_end, tend)
        intersection = tt2 - tt1
        duration = tend - tstart
        tmp_width = tmp_end - tmp_start
        iou = intersection / (tmp_width + duration - intersection).astype(np.float)

        idxs = np.where(iou > 0.65 + 0.25 * tmp_width)[0]
        tscore[idxs] = tscore[idxs] * np.exp(-np.square(iou[idxs]) / 0.75)

        tstart = list(tstart)
        tend = list(tend)
        tscore = list(tscore)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def min_max(x):
    x = (x - min(x)) / (max(x) - min(x))
    return x


def sub_processor(lock, pid, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm.tqdm(
            total=len(video_list),
            position=pid,
            desc=text
        )
    for i in range(len(video_list)):
        video_name = video_list[i]
        df = pd.read_csv(os.path.join(result_dir, video_name + ".csv"))
        df['score'] = df.iou.values[:] * df.start.values[:] * df.end.values[:]
        if len(df) > 1:
            df = softNMS(df)
        df = df.sort_values(by="score", ascending=False)
        video_info = video_dict[video_name]
        video_duration = video_info["duration_second"]
        proposal_list = []

        for j in range(min(top_number, len(df))):
            tmp_proposal = {}
            tmp_proposal["score"] = df.score.values[j]
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                       min(1, df.xmax.values[j]) * video_duration]
            proposal_list.append(tmp_proposal)
        result_dict[video_name[2:]] = proposal_list
        with lock:
            progress.update(1)

    with lock:
        progress.close()


train_dict, val_dict, test_dict = getDatasetDict()
mode = args.mode
if mode == 'validation':
    video_dict = val_dict
else:
    video_dict = test_dict

result_dir = args.input_dir
video_list = list(video_dict.keys())

""" Post processing using multiprocessing
"""
global result_dict
result_dict = mp.Manager().dict()

processes = []
lock = threading.Lock()

total_video_num = len(video_list)
per_thread_video_num = total_video_num // thread_num
for i in range(thread_num):
    if i == thread_num - 1:
        sub_video_list = video_list[i * per_thread_video_num:]
    else:
        sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
    p = mp.Process(target=sub_processor, args=(lock, i, sub_video_list))
    p.start()
    processes.append(p)
for p in processes:
    p.join()

result_dict = dict(result_dict)
output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}

with open(args.output_file, 'w') as outfile:
    json.dump(output_dict, outfile)
