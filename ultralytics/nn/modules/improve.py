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

class channel_att(nn.Module):

    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):
    '''
    这个在论文中是Position attention network模块，是所谓的位置注意力机制，这是用来提取局部特征的
    '''

    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel,
                                  out_channels=channel // reduction,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction,
                             out_channels=channel,
                             kernel_size=1,
                             stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction,
                             out_channels=channel,
                             kernel_size=1,
                             stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class CPAM(nn.Module):
    # Concatenate a list of tensors along dimension
    '''
    argument:
        ch : 通道数
    '''

    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        # input1, input2 = x[0], x[1]
        # input1 = self.channel_att(input1)
        # print("10=>", x[0].shape, "-1=>", x[1].shape)
        #* 由于这里的x[0]将会通过通道注意力，所以尽量希望x[0]是全局特征，而x[1]是会进入位置注意力，所以尽量希望x[1]是局部特征
        x = self.channel_att(x[0]) + x[1]
        x = self.local_att(x)
        return x