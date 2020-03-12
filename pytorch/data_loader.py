import torch
from torch.utils.data import Dataset

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


class DBGDataSet(Dataset):
    """
    DBG dataset to load ActivityNet-1.3 data
    """
    def __init__(self, mode='training'):
        train_dict, val_dict, test_dict = getDatasetDict(video_info_file, video_filter)
        training = True
        if mode == 'training':
            video_dict = train_dict
        else:
            training = False
            video_dict = val_dict
        self.mode = mode
        self.video_dict = video_dict
        video_num = len(list(video_dict.keys()))
        video_list = np.arange(video_num)

        # load raw data
        if training:
            data_dict, train_video_mean_len = getFullData(video_dict, dbg_config,
                                                          last_channel=False,
                                                          training=True)
        else:
            data_dict = getFullData(video_dict, dbg_config,
                                    last_channel=False, training=False)

        # transform data to torch tensor
        for key in list(data_dict.keys()):
            data_dict[key] = torch.Tensor(data_dict[key]).float()
        self.data_dict = data_dict

        if data_aug and training:
            # add train video with short proposals
            add_list = np.where(np.array(train_video_mean_len) < 0.2)
            add_list = np.reshape(add_list, [-1])
            video_list = np.concatenate([video_list, add_list[:]], 0)

        self.video_list = video_list
        np.random.shuffle(self.video_list)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.video_list[idx]
        data_dict = self.data_dict
        gt_action = data_dict['gt_action'][idx].unsqueeze(0)
        gt_start = data_dict['gt_start'][idx].unsqueeze(0)
        gt_end = data_dict['gt_end'][idx].unsqueeze(0)
        feature = data_dict['feature'][idx]
        iou_label = data_dict['iou_label'][idx].unsqueeze(0)
        return gt_action, gt_start, gt_end, feature, iou_label
