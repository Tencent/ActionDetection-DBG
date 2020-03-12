import sys

sys.path.append('.')
import os

import numpy as np
import tqdm

import data_loader
import utils
from model import DBG
from config_loader import dbg_config

import torch
import torch.nn as nn

""" set checkpoint directory """
checkpoint_dir = dbg_config.checkpoint_dir
""" set result directory to save all csv files """
result_dir = dbg_config.result_dir

""" get input feature scale """
tscale = dbg_config.tscale
""" get input feature channel number """
feature_dim = dbg_config.feature_dim

""" get testing batch size """
batch_size = dbg_config.test_batch_size
""" get testing mode: validation or test """
test_mode = dbg_config.test_mode

""" get map mask """
mask = data_loader.gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), 1)
mask = torch.from_numpy(mask).float().requires_grad_(False).cuda()

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
    torch.backends.cudnn.enabled = False # set False to speed up Conv3D operation
    with torch.no_grad():
        """ setup DBG model and load weights """
        net = DBG(feature_dim)
        state_dict = torch.load(os.path.join(checkpoint_dir, 'DBG_checkpoint_best.ckpt'))
        net.load_state_dict(state_dict)
        net = nn.DataParallel(net, device_ids=[0]).cuda()
        net.eval()

        """ get testing dataset """
        train_dict, val_dict, test_dict = data_loader.getDatasetDict(dbg_config.video_info_file)
        if test_mode == 'validation':
            video_dict = val_dict
        else:
            video_dict = test_dict

        batch_video_list = data_loader.getBatchListTest(video_dict, batch_size)

        batch_result_xmin = []
        batch_result_xmax = []
        batch_result_iou = []
        batch_result_pstart = []
        batch_result_pend = []

        """ runing DBG model """
        print('Runing DBG model ...')
        for idx in tqdm.tqdm(range(len(batch_video_list))):
            batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature = \
                data_loader.getProposalDataTest(batch_video_list[idx], dbg_config)
            in_feature = torch.from_numpy(batch_anchor_feature).float().cuda().permute(0, 2, 1)
            output_dict = net(in_feature)
            out_iou = output_dict['iou']
            out_start = output_dict['prop_start']
            out_end = output_dict['prop_end']

            # fusion starting and ending map score
            out_start = out_start * mask
            out_end = out_end * mask
            out_start = torch.sum(out_start, 3) / torch.sum(mask, 3)
            out_end = torch.sum(out_end, 2) / torch.sum(mask, 2)

            batch_result_xmin.append(batch_anchor_xmin)
            batch_result_xmax.append(batch_anchor_xmax)
            batch_result_iou.append(out_iou[:, 0].cpu().detach().numpy())
            batch_result_pstart.append(out_start[:, 0].cpu().detach().numpy())
            batch_result_pend.append(out_end[:, 0].cpu().detach().numpy())

        utils.save_proposals_result(batch_video_list,
                                    batch_result_xmin,
                                    batch_result_xmax,
                                    batch_result_iou,
                                    batch_result_pstart,
                                    batch_result_pend,
                                    tscale, result_dir)
