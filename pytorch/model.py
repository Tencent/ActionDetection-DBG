import numpy as np
import torch
import torch.nn as nn
from custom_op.prop_tcfg_op import PropTcfg


def conv1d(in_channels, out_channels, kernel_size=3, is_relu=True):
    """
    Construct Conv1D operation
    :param in_channels: channel number of input tensor
    :param out_channels: channel number of output tensor
    :param kernel_size: int
    :param is_relu: bool, use ReLU or not
    :return: Conv1D module
    """
    if is_relu:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2)
        )


def conv2d(in_channels, out_channels, kernel_size=3, is_relu=True):
    """
    Construct Conv2D operation
    :param in_channels: channel number of input tensor
    :param out_channels: channel number of output tensor
    :param kernel_size: int
    :param is_relu: bool, use ReLU or not
    :return: Conv2D module
    """
    if is_relu:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2)
        )


class DSBaseNet(nn.Module):
    """
    Setup dual stream base network (DSB)
    """
    def __init__(self, feature_dim):
        super(DSBaseNet, self).__init__()
        feature_dim = feature_dim // 2
        self.feature_dim = feature_dim
        self.conv1_1 = conv1d(feature_dim, 256, 3)
        self.conv1_2 = conv1d(256, 128, 3)
        self.conv1_3 = conv1d(128, 1, 1, is_relu=False)

        self.conv2_1 = conv1d(feature_dim, 256, 3)
        self.conv2_2 = conv1d(256, 128, 3)
        self.conv2_3 = conv1d(128, 1, 1, is_relu=False)

        self.conv3 = conv1d(128, 1, 1, is_relu=False)

    def forward(self, x):
        x1, x2 = torch.split(x, self.feature_dim, 1)
        x1 = self.conv1_1(x1)
        x1 = self.conv1_2(x1)
        x1_feat = x1
        x1 = torch.sigmoid(self.conv1_3(x1))

        x2 = self.conv2_1(x2)
        x2 = self.conv2_2(x2)
        x2_feat = x2
        x2 = torch.sigmoid(self.conv2_3(x2))

        xc = x1_feat + x2_feat
        xc_feat = xc
        x3 = torch.sigmoid(self.conv3(xc))

        score = (x1 + x2 + x3) / 3.0

        output_dict = {
            'score': score,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'xc_feat': xc_feat
        }
        return output_dict


class ProposalFeatureGeneration(nn.Module):
    """
    Setup proposal feature generation module
    """
    def __init__(self, in_channels=128):
        super(ProposalFeatureGeneration, self).__init__()
        self.prop_tcfg = PropTcfg()
        self.conv3d = nn.Conv3d(in_channels, 512, kernel_size=(32, 1, 1))

    def forward(self, action_score, xc_feat):
        action_feat = self.prop_tcfg(action_score)  # B x 1 x 32 x T x T
        action_feat = torch.squeeze(action_feat, 1)  # B x 32 x T x T
        net_feat = self.prop_tcfg(xc_feat)  # B x 128 x 32 x T x T
        net_feat = self.conv3d(net_feat)  # B x 512 x 1 x T x T
        net_feat = torch.squeeze(net_feat, 2)  # B x 512 x T x T

        return action_feat, net_feat


class ACRNet(nn.Module):
    """
    Setup action classification regression network (ACR)
    """
    def __init__(self, in_channels=32):
        super(ACRNet, self).__init__()
        self.conv2d = nn.Sequential(
            conv2d(in_channels, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 1, 1, is_relu=False)
        )

    def forward(self, action_feat):
        iou = self.conv2d(action_feat)
        iou = torch.sigmoid(iou)
        return iou


class TBCNet(nn.Module):
    """
    Setup temporal boundary classification network (TBC)
    """
    def __init__(self, in_channels=512):
        super(TBCNet, self).__init__()
        self.conv2d = nn.Sequential(
            conv2d(in_channels, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 2, 1, is_relu=False)
        )

    def forward(self, net_feat):
        x = self.conv2d(net_feat)
        x = torch.sigmoid(x)

        prop_start = x[:, :1].contiguous()
        prop_end = x[:, 1:].contiguous()
        return prop_start, prop_end


class DBG(nn.Module):
    """
    Setup dense boundary generator framework (DBG)
    """
    def __init__(self, feature_dim):
        super(DBG, self).__init__()

        self.DSBNet = DSBaseNet(feature_dim)
        self.PropFeatGen = ProposalFeatureGeneration()
        self.ACRNet = ACRNet()
        self.TBCNet = TBCNet()

        self.best_loss = 999999
        self.reset_params() # reset all params by glorot uniform

    @staticmethod
    def glorot_uniform_(tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        scale = 1.0
        scale /= max(1., (fan_in + fan_out) / 2.)
        limit = np.sqrt(3.0 * scale)
        return nn.init._no_grad_uniform_(tensor, -limit, limit)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
            DBG.glorot_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        DSB_output = self.DSBNet(x)
        action_feat, net_feat = self.PropFeatGen(DSB_output['score'], DSB_output['xc_feat'])
        iou = self.ACRNet(action_feat)
        prop_start, prop_end = self.TBCNet(net_feat)

        output_dict = {
            'x1': DSB_output['x1'],
            'x2': DSB_output['x2'],
            'x3': DSB_output['x3'],
            'iou': iou,
            'prop_start': prop_start,
            'prop_end': prop_end
        }
        return output_dict
