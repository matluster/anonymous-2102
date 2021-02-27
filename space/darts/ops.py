import torch
import torch.nn as nn
import torch.nn.functional as F


OPS = {
    'avg_pool_3x3': lambda C: Pool("avg", C, 3, 1),
    'max_pool_3x3': lambda C: Pool("max", C, 3, 1),
    'skip_connect': lambda C: SkipConnection(C, C),
    'sep_conv_3x3': lambda C: SepConv(C, C, 3, 1),
    'sep_conv_5x5': lambda C: SepConv(C, C, 5, 2),
    'sep_conv_7x7': lambda C: SepConv(C, C, 7, 3),
    'dil_conv_3x3': lambda C: DilConv(C, C, 3, 2, 2),
    'dil_conv_5x5': lambda C: DilConv(C, C, 5, 4, 2),
}


class DynamicConv2d(nn.Conv2d):
    def forward(self, input, stride):
        return F.conv2d(input, self.weight, self.bias, stride,
                        self.padding, self.dilation, self.groups)


class DynamicStrideModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_path = DropPath_()
        self.ops = []

    def forward(self, x, stride):
        for i in range(len(self.op)):
            if i == 1:
                x = self.op[i](x, stride)
            else:
                x = self.op[i](x)
        return self.drop_path(x)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Pool(nn.Module):
    def __init__(self, pool_type, channels, kernel_size, padding, affine=True):
        super().__init__()
        self.pool_type = pool_type.lower()
        self.kernel_size = kernel_size
        self.padding = padding
        # self.bn = nn.BatchNorm2d(channels, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, stride):
        if self.pool_type == "max":
            out = F.max_pool2d(x, self.kernel_size, stride, self.padding)
        elif self.pool_type == "avg":
            out = F.avg_pool2d(x, self.kernel_size, stride, self.padding, count_include_pad=False)
        else:
            raise ValueError()
        # out = self.bn(out)
        return self.drop_path(out)


class StdConv(nn.Module):
    def __init__(self, C_in, C_out, stride, padding, affine=True):
        super().__init__()
        self.op = nn.ModuleList([
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, 1, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        ])

    def forward(self, x):
        return self.op(x)


class DilConv(DynamicStrideModule):
    def __init__(self, C_in, C_out, kernel_size, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.ModuleList([
            nn.ReLU(),
            DynamicConv2d(C_in, C_in, kernel_size, 1, padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        ])


class SepConv(DynamicStrideModule):
    def __init__(self, C_in, C_out, kernel_size, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            DynamicConv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )


class SkipConnection(FactorizedReduce):
    def __init__(self, C_in, C_out):
        super().__init__(C_in, C_out)
        self.drop_path = DropPath_()

    def forward(self, x, stride):
        if stride > 1:
            out = super().forward(x)
            return self.drop_path(out)
        return x


#### utility layers ####

class DropPath_(nn.Module):
    # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
    def __init__(self, drop_prob=0.):
        super(DropPath_, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            return x.div(keep_prob).mul(mask)
        return x


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, padding):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)
