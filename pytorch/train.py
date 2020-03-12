import sys
sys.path.append('.')

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from data_loader import DBGDataSet, gen_mask
from model import DBG
from torch.utils.data import DataLoader

""" Load config
"""
from config_loader import dbg_config

checkpoint_dir = dbg_config.checkpoint_dir
batch_size = dbg_config.batch_size
learning_rate = dbg_config.learning_rate
tscale = dbg_config.tscale
feature_dim = dbg_config.feature_dim
epoch_num = dbg_config.epoch_num

""" Initialize map mask
"""
mask = gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), 1)
mask = torch.from_numpy(mask).float().requires_grad_(False).cuda()
tmp_mask = mask.repeat(batch_size, 1, 1, 1).requires_grad_(False)
tmp_mask = tmp_mask > 0


def binary_logistic_loss(gt_scores, pred_anchors):
    """
    Calculate weighted binary logistic loss
    :param gt_scores: gt scores tensor
    :param pred_anchors: prediction score tensor
    :return: loss output tensor
    """
    gt_scores = gt_scores.view(-1)
    pred_anchors = pred_anchors.view(-1)

    pmask = (gt_scores > 0.5).float()
    num_positive = torch.sum(pmask)
    num_entries = pmask.size()[0]

    ratio = num_entries / max(num_positive, 1)
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 1e-6
    neg_pred_anchors = 1.0 - pred_anchors + epsilon
    pred_anchors = pred_anchors + epsilon

    loss = coef_1 * pmask * torch.log(pred_anchors) + coef_0 * (1.0 - pmask) * torch.log(
        neg_pred_anchors)
    loss = -1.0 * torch.mean(loss)
    return loss


def IoU_loss(gt_iou, pred_iou):
    """
    Calculate IoU loss
    :param gt_iou: gt IoU tensor
    :param pred_iou: prediction IoU tensor
    :return: loss output tensor
    """
    u_hmask = (gt_iou > 0.6).float()
    u_mmask = ((gt_iou <= 0.6) & (gt_iou > 0.2)).float()
    u_lmask = (gt_iou <= 0.2).float() * mask

    u_hmask = u_hmask.view(-1)
    u_mmask = u_mmask.view(-1)
    u_lmask = u_lmask.view(-1)

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = 1.0 * num_h / num_m
    r_m = torch.min(r_m, torch.Tensor([1.0]).cuda())
    u_smmask = torch.rand(u_hmask.size()[0], requires_grad=False).cuda() * u_mmask
    u_smmask = (u_smmask > (1.0 - r_m)).float()

    r_l = 2.0 * num_h / num_l
    r_l = torch.min(r_l, torch.Tensor([1.0]).cuda())

    u_slmask = torch.rand(u_hmask.size()[0], requires_grad=False).cuda() * u_lmask
    u_slmask = (u_slmask > (1.0 - r_l)).float()

    iou_weights = u_hmask + u_smmask + u_slmask

    gt_iou = gt_iou.view(-1)
    pred_iou = pred_iou.view(-1)

    iou_loss = F.smooth_l1_loss(pred_iou * iou_weights, gt_iou * iou_weights, reduction='none')
    iou_loss = torch.sum(iou_loss * iou_weights) / torch.max(torch.sum(iou_weights),
                                                             torch.Tensor([1.0]).cuda())
    return iou_loss


def DBG_train(net, dl_iter, optimizer, epoch, training):
    """
    One epoch of runing DBG model
    :param net: DBG network module
    :param dl_iter: data loader
    :param optimizer: optimizer module
    :param epoch: current epoch number
    :param training: bool, training or not
    :return: None
    """
    if training:
        net.train()
    else:
        net.eval()
    loss_action_val = 0
    loss_iou_val = 0
    loss_start_val = 0
    loss_end_val = 0
    cost_val = 0
    for n_iter, \
        (gt_action, gt_start, gt_end, feature, iou_label) in tqdm.tqdm(enumerate(dl_iter)):
        gt_action = gt_action.cuda()
        gt_start = gt_start.cuda()
        gt_end = gt_end.cuda()
        feature = feature.cuda()
        iou_label = iou_label.cuda()

        output_dict = net(feature)
        x1 = output_dict['x1']
        x2 = output_dict['x2']
        x3 = output_dict['x3']
        iou = output_dict['iou']
        prop_start = output_dict['prop_start']
        prop_end = output_dict['prop_end']

        # calculate action loss
        loss_action = binary_logistic_loss(gt_action, x1) + \
                      binary_logistic_loss(gt_action, x2) + \
                      binary_logistic_loss(gt_action, x3)
        loss_action /= 3.0

        # calculate IoU loss
        iou_losses = 0.0
        for i in range(batch_size):
            iou_loss = IoU_loss(iou_label[i:i + 1], iou[i:i + 1])
            iou_losses += iou_loss
        loss_iou = iou_losses / batch_size * 10.0

        # calculate starting and ending map loss
        gt_start = torch.unsqueeze(gt_start, 3).repeat(1, 1, 1, tscale)
        gt_end = torch.unsqueeze(gt_end, 2).repeat(1, 1, tscale, 1)
        loss_start = binary_logistic_loss(
            torch.masked_select(gt_start, tmp_mask),
            torch.masked_select(prop_start, tmp_mask)
        )
        loss_end = binary_logistic_loss(
            torch.masked_select(gt_end, tmp_mask),
            torch.masked_select(prop_end, tmp_mask)
        )

        # total loss
        cost = 2.0 * loss_action + loss_iou + loss_start + loss_end

        if training:
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        loss_action_val += loss_action.cpu().detach().numpy()
        loss_iou_val += loss_iou.cpu().detach().numpy()
        loss_start_val += loss_start.cpu().detach().numpy()
        loss_end_val += loss_end.cpu().detach().numpy()
        cost_val += cost.cpu().detach().numpy()

    loss_action_val /= (n_iter + 1)
    loss_iou_val /= (n_iter + 1)
    loss_start_val /= (n_iter + 1)
    loss_end_val /= (n_iter + 1)
    cost_val /= (n_iter + 1)

    if training:
        print(
            "Epoch-%d Train Loss: "
            "Total - %.05f, Action - %.05f, Start - %.05f, End - %.05f, IoU - %.05f"
            % (epoch, cost_val, loss_action_val, loss_start_val, loss_end_val, loss_iou_val))
    else:
        print(
            "Epoch-%d Validation Loss: "
            "Total - %.05f, Action - %.05f, Start - %.05f, End - %.05f, IoU - %.05f"
            % (epoch, cost_val, loss_action_val, loss_start_val, loss_end_val, loss_iou_val))

        torch.save(net.module.state_dict(),
                   os.path.join(checkpoint_dir, 'DBG_checkpoint-%d.ckpt' % epoch))
        if cost_val < net.module.best_loss:
            net.module.best_loss = cost_val
            torch.save(net.module.state_dict(),
                       os.path.join(checkpoint_dir, 'DBG_checkpoint_best.ckpt'))



def set_seed(seed):
    """
    Set randon seed for pytorch
    :param seed:
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('Only train on GPU.')
        exit()
    torch.backends.cudnn.enabled = False # set False to speed up Conv3D operation
    set_seed(2020)
    net = DBG(feature_dim)
    net = nn.DataParallel(net, device_ids=[0]).cuda()

    # set weight decay for different parameters
    Net_bias = []
    for name, p in net.module.named_parameters():
        if 'bias' in name:
            Net_bias.append(p)

    DSBNet_weight = []
    for name, p in net.module.DSBNet.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)

    PFG_weight = []
    for name, p in net.module.PropFeatGen.named_parameters():
        if 'bias' not in name:
            PFG_weight.append(p)

    ACR_TBC_weight = []
    for name, p in net.module.ACRNet.named_parameters():
        if 'bias' not in name:
            ACR_TBC_weight.append(p)
    for name, p in net.module.TBCNet.named_parameters():
        if 'bias' not in name:
            ACR_TBC_weight.append(p)

    # setup Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': Net_bias, 'weight_decay': 0},
        {'params': DSBNet_weight, 'weight_decay': 2e-3},
        {'params': PFG_weight, 'weight_decay': 2e-4},
        {'params': ACR_TBC_weight, 'weight_decay': 2e-5}
    ], lr=1.0)

    # setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda x: learning_rate[x])
    # setup training and validation data loader
    train_dl = DataLoader(DBGDataSet(mode='training'), batch_size=batch_size,
                          shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    val_dl = DataLoader(DBGDataSet(mode='validation'), batch_size=batch_size,
                        shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    # train DBG
    for i in range(epoch_num):
        scheduler.step(i)
        print('current learning rate:', scheduler.get_lr()[0])
        DBG_train(net, train_dl, optimizer, i, training=True)
        DBG_train(net, val_dl, optimizer, i, training=False)
