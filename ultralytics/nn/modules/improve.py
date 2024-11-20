# this is where you add your custom improvement modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv
import math
from einops import rearrange


class TFE(nn.Module):
    '''
    #!论文中这个模块叫TFE
    这个模块从参数量上看，会将模型显存提升2～3倍
    '''

    def __init__(self, in_dim):
        super().__init__()
        self.conv_l = Conv(in_dim, in_dim, 3, 1, 1)
        self.conv_m = Conv(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre_up = Conv(in_dim, 2 * in_dim, 3, 1, 1)
        self.conv_s_post_up = Conv(2 * in_dim, in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        # print(len(x))
        # l, m, s = x[0], x[1], x[2]
        # tgt_size = m.shape[2:]
        # print("tgt=>", tgt_size)
        # print(x[0].shape, x[1].shape, x[2].shape)

        # x[0] = self.conv_l(x[0])
        x[0] = F.adaptive_max_pool2d(x[0], x[1].shape[2:]) + F.adaptive_avg_pool2d(x[0],
                                                                                   x[1].shape[2:])  # 这是最大的特征图像上做的特征提取
        # x[0] = self.conv_l(x[0])
        # x[1] = self.conv_m(x[1])
        # x[2] = self.conv_s_pre_up(x[2])
        x[2] = F.interpolate(x[2], x[1].shape[2:], mode='nearest')  # 这个应该是最小的特征图做特征提取
        # x[2] = self.conv_s_post_up(x[2])
        # lms = torch.cat([l, m, s], dim=1)
        # print(x[0].shape, x[1].shape, x[2].shape)
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


class ScalSeq(nn.Module):
    '''
    #!在论文中，这部分模块名为 SSFF,和这个地方有待机会嵌入mamba模块来提取特征
    '''

    def __init__(self, channel):
        super(ScalSeq, self).__init__()
        self.conv1 = Conv(512, channel, 1)
        self.conv2 = Conv(512, channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        # self.act = nn.LeakyReLU(0.1)
        self.act = nn.SiLU(inplace=True)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2] # 大特征图、中特征图、小特征图
        # p4_2 = self.conv1(p4) # 貌似此处通道数必须要为512
        p4 = F.interpolate(self.conv1(p4), p3.size()[2:], mode='nearest') # channel and size adjustment to p3
        # p5_2 = self.conv2(p5) 
        p5 = F.interpolate(self.conv2(p5), p3.size()[2:], mode='nearest') # channel and size adjustment to p3
        p3 = torch.unsqueeze(p3, -3)  # (c,h,w) -> (1,c,h,w), p3_3d
        p4 = torch.unsqueeze(p4, -3)  # (c,h,w) -> (1,c,h,w), p4_3d
        p5 = torch.unsqueeze(p5, -3)  # (c,h,w) -> (1,c,h,w), p5_3d
        # combine = torch.cat([p3,p4,p5],dim = 2)
        # conv_3d = self.conv3d(torch.cat([p3,p4,p5],dim = 2))
        # bn = self.bn(self.conv3d(torch.cat([p3,p4,p5],dim = 2)))
        # act = self.act(self.bn(self.conv3d(torch.cat([p3,p4,p5],dim = 2))))
        x = self.pool_3d(self.act(self.bn(self.conv3d(torch.cat([p3, p4, p5], dim=2)))))
        del p3, p4, p5
        x = torch.squeeze(x, 2)
        return x


class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),nn.BatchNorm2d(outc),nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x,p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0,base_int), indexing='ij')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number >  0:
            mod_p_n_x,mod_p_n_y = torch.meshgrid(
                torch.arange(row_number,row_number+1),
                torch.arange(0,mod_number), indexing='ij')

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x,p_n_y  = torch.cat((p_n_x,mod_p_n_x)),torch.cat((p_n_y,mod_p_n_y))
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride), indexing='ij')

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset