from .vmamba import SS2D
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import traceback
import os
import sys

# class SelfSS2D(nn.Module):

#     def __init__(self, d_model, dropout=0, d_state=16) -> None:
#         super().__init__()
#         self.ss2d = SS2D(d_model, dropout, d_state)

#     def forward(self, x):
#         # 出现0向量的veiw
#         print(x.shape)
#         x = x.permute(0, 2, 3, 1)
#         x = self.ss2d(x)
#         x = x.permute(0, 3, 1, 2)
#         return x


class DepthWiseConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in,
                               dim_in,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


# 如果系统中存在DWCONV_IMPL环境变量，则尝试导入深度可分离卷积，否则用普通的卷积（目前使用普通卷积，作者或许测试过了，最终并没有采取深度卷积）
if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM

        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)

        # print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())

        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

        # print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:

    def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

    # print('Using PyTorch large kernel dw conv impl')


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape)).to('cuda')
        self.bias = nn.Parameter(torch.zeros(normalized_shape)).to('cuda')
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        x.to('cuda')
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Local_SS2D(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2).to('cuda')
        # self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        # trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first').to('cuda')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first').to('cuda')

        self.SS2D = SS2D(d_model=dim // 2, dropout=0, d_state=16).to('cuda')

    def forward(self, x):
        x.to('cuda')
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        B, C, a, b = x2.shape

        x2 = x2.permute(0, 2, 3, 1)

        x2 = self.SS2D(x2)

        x2 = x2.permute(0, 3, 1, 2)

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class H_SS2D(nn.Module):

    def __init__(self, dim, order=5, gflayer=None, s=1.0, d_state=16):
        super().__init__()
        if not torch.cuda.is_available():
            raise NotImplementedError('H_SS2D is only supported on CUDA devices')
        self.order = order
        self.dims = [dim // 2**i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            gflayer = eval(gflayer)
            self.dwconv = gflayer(sum(self.dims)).to('cuda')

        self.proj_out = nn.Conv2d(dim, dim, 1).to('cuda')

        self.pws = nn.ModuleList([nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]).to('cuda')

        num = len(self.dims)
        if num == 2:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16)
        elif num == 3:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16)
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=16)
        elif num == 4:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16).to('cuda')
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=16).to('cuda')
            self.ss2d_3 = SS2D(d_model=self.dims[3], dropout=0, d_state=16).to('cuda')
        elif num == 5:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16).to('cuda')
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=16).to('cuda')
            self.ss2d_3 = SS2D(d_model=self.dims[3], dropout=0, d_state=16).to('cuda')
            self.ss2d_4 = SS2D(d_model=self.dims[4], dropout=0, d_state=16).to('cuda')

        self.ss2d_in = SS2D(d_model=self.dims[0], dropout=0, d_state=16).to('cuda')

        self.scale = eval(s)

        print('[H_SS2D]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        x.to('cuda')
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        # 构造一个和self.scale相同的张量到cuda上
        # scale = torch.ones_like(pwa) * self.scale
        # scale = scale.to('cuda')
        # dw_abc = self.dwconv(abc) * scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        x = x.permute(0, 2, 3, 1)
        x = self.ss2d_in(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
            if i == 0:
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_1(x)
                x = x.permute(0, 3, 1, 2)
            elif i == 1:
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_2(x)
                x = x.permute(0, 3, 1, 2)
            elif i == 2:
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_3(x)
                x = x.permute(0, 3, 1, 2)
            elif i == 3:
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_4(x)
                x = x.permute(0, 3, 1, 2)

        x = self.proj_out(x)

        return x


if __name__ == '__main__':
    # 尝试不同阶数的H-SS2d，这个模块必须使用cuda
    # 在论文中图像大小越小，使用阶数越高
    for i in range(2, 6):
        # print('order =', i)
        hss2d = H_SS2D(dim=64, order=i, s=1 / 3, gflayer=Local_SS2D).to('cuda')
        # print(hss2d)
        x = torch.randn(1, 64, 128, 128).to('cuda')
        y = hss2d(x)
        print("y:", y.shape)
        # 清空cuda缓存
        torch.cuda.empty_cache()
