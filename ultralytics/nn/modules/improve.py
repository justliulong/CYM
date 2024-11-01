# this is where you add your custom improvement modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


class TFE(nn.Module):
    '''
    #!论文中这个模块叫TFE
    '''

    def __init__(self, in_dim):
        super().__init__()
        self.conv_l_post_down = Conv(in_dim, in_dim, 3, 1, 1)
        self.conv_m = Conv(in_dim, in_dim, 3, 1, 1)
        # self.conv_s_pre_up = Conv(in_dim, 2*in_dim, 3, 1, 1)
        # self.conv_s_post_up = Conv(2*in_dim, in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        # print(len(x))
        # l, m, s = x[0], x[1], x[2]
        # tgt_size = m.shape[2:]
        # print("tgt=>", tgt_size)
        # print("0=>", x[0].shape, "1=>", x[1].shape, "2=>", x[2].shape, "3=>", x[3].shape, "4=>", x[4].shape, "5=>",
        #       x[5].shape, "6=>", x[6].shape, "7=>", x[7].shape, "8=>", x[8].shape, "9=>", x[9].shape, "10=>",
        #       x[10].shape)
        x[0] = F.adaptive_max_pool2d(x[0], x[1].shape[2:]) + F.adaptive_avg_pool2d(x[0], x[1].shape[2:])
        # x[0] = self.conv_l_post_down(x[0])
        # x[1] = self.conv_m(x[1])
        # s = self.conv_s_pre_up(s)
        x[2] = F.interpolate(x[2], x[1].shape[2:], mode='nearest')
        # s = self.conv_s_post_up(s)
        # lms = torch.cat([l, m, s], dim=1)
        return torch.cat([x[0], x[1], x[2]], dim=1)
