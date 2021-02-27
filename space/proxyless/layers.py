from collections import OrderedDict

import torch
import torch.nn as nn

from .utils import shuffle_layer, get_same_padding


class BasicUnit(nn.Module):
    def __str__(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BasicLayer(BasicUnit):
    def __init__(self, in_channels, out_channels, use_bn=True, act_func="relu", dropout_rate=0, ops_order="weight_bn_act"):
        super(BasicLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm2d(in_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        # activation
        if act_func == "relu":
            self.activation = nn.ReLU(inplace=self.ops_list[0] != "act")
        elif act_func == "relu6":
            self.activation = nn.ReLU6(inplace=self.ops_list[0] != "act")
        else:
            self.activation = None
        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            self.dropout = None

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == "weight":
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == "bn":
                if self.bn is not None:
                    x = self.bn(x)
            elif op == "act":
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError("Unrecognized op: %s" % op)
        return x

    @property
    def config(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }


class ConvLayer(BasicLayer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=False,
                 has_shuffle=False, use_bn=True, act_func="relu", dropout_rate=0, ops_order="weight_bn_act"):
        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation
        # `kernel_size`, `stride`, `padding`, `dilation` can either be `int` or `tuple` of int
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                              padding=padding, dilation=self.dilation, groups=self.groups, bias=self.bias)

    def weight_call(self, x):
        x = self.conv(x)
        if self.has_shuffle and self.groups > 1:
            x = shuffle_layer(x, self.groups)
        return x

    def __str__(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return "%dx%d_DilatedConv" % (kernel_size[0], kernel_size[1])
            else:
                return "%dx%d_Conv" % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return "%dx%d_DilatedGroupConv" % (kernel_size[0], kernel_size[1])
            else:
                return "%dx%d_GroupConv" % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        config = {
            "name": ConvLayer.__name__,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "has_shuffle": self.has_shuffle,
        }
        config.update(super(ConvLayer, self).config)
        return config


class DepthConvLayer(BasicLayer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=False,
                 has_shuffle=False, use_bn=True, act_func="relu", dropout_rate=0, ops_order="weight_bn_act"):
        super(DepthConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding = (padding[0] * self.dilation, padding[1] * self.dilation)
        # `kernel_size`, `stride`, `padding`, `dilation` can either be `int` or `tuple` of int
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, stride=self.stride,
                                    padding=padding, dilation=self.dilation, groups=in_channels, bias=False)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=self.groups, bias=self.bias)

    def weight_call(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        if self.has_shuffle and self.groups > 1:
            x = shuffle_layer(x, self.groups)
        return x

    def __str__(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            return "%dx%d_DilatedDepthConv" % (kernel_size[0], kernel_size[1])
        else:
            return "%dx%d_DepthConv" % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        config = {
            "name": DepthConvLayer.__name__,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "has_shuffle": self.has_shuffle,
        }
        config.update(super(DepthConvLayer, self).config)
        return config


class PoolingLayer(BasicLayer):
    def __init__(self, in_channels, out_channels, pool_type, kernel_size=2, stride=2, use_bn=False,
                 act_func=None, dropout_rate=0, ops_order="weight_bn_act"):
        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        padding = get_same_padding(self.kernel_size) if self.stride == 1 else 0
        if self.pool_type == "avg":
            self.pool = nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False)
        elif self.pool_type == "max":
            self.pool = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError

    def weight_call(self, x):
        return self.pool(x)

    def __str__(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return "%dx%d_%sPool" % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        config = {
            "name": PoolingLayer.__name__,
            "pool_type": self.pool_type,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
        }
        config.update(super(PoolingLayer, self).config)
        return config


class IdentityLayer(BasicLayer):

    def __init__(self, in_channels, out_channels, use_bn=False, act_func=None, dropout_rate=0, ops_order="weight_bn_act"):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_call(self, x):
        return x

    def __str__(self):
        return "Identity"

    @property
    def config(self):
        config = {
            "name": IdentityLayer.__name__,
        }
        config.update(super(IdentityLayer, self).config)
        return config


class LinearLayer(BasicUnit):

    def __init__(self, in_features, out_features, bias=True, use_bn=False,
                 act_func=None, dropout_rate=0, ops_order="weight_bn_act"):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm1d(in_features)
            else:
                self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = None
        # activation
        if act_func == "relu":
            self.activation = nn.ReLU(inplace=self.ops_list[0] != "act")
        elif act_func == "relu6":
            self.activation = nn.ReLU6(inplace=self.ops_list[0] != "act")
        elif act_func == "tanh":
            self.activation = nn.Tanh()
        elif act_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        # linear
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def forward(self, x):
        for op in self.ops_list:
            if op == "weight":
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.linear(x)
            elif op == "bn":
                if self.bn is not None:
                    x = self.bn(x)
            elif op == "act":
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError("Unrecognized op: %s" % op)
        return x

    @property
    def __str__(self):
        return "%dx%d_Linear" % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            "name": LinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }


class MBInvertedConvLayer(BasicUnit):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio

        if self.expand_ratio > 1:
            feature_dim = round(in_channels * self.expand_ratio)
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels, feature_dim, 1, 1, 0, bias=False)),
                ("bn", nn.BatchNorm2d(feature_dim)),
                ("relu", nn.ReLU6(inplace=True)),
            ]))
        else:
            feature_dim = in_channels
            self.inverted_bottleneck = None

        # depthwise convolution
        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ("bn", nn.BatchNorm2d(feature_dim)),
            ("relu", nn.ReLU6(inplace=True)),
        ]))

        # pointwise linear
        self.point_linear = OrderedDict([
            ("conv", nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ("bn", nn.BatchNorm2d(out_channels)),
        ])

        self.point_linear = nn.Sequential(self.point_linear)

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    def __str__(self):
        return "%dx%d_MBConv%d" % (self.kernel_size, self.kernel_size, self.expand_ratio)

    @property
    def config(self):
        return {
            "name": MBInvertedConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
        }


class ZeroLayer(BasicUnit):
    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        return torch.zeros((n, c, h, w), device=x.device)

    def __str__(self):
        return "Zero"

    @property
    def config(self):
        return {
            "name": ZeroLayer.__name__,
            "stride": self.stride,
        }


class MobileInvertedResidualBlock(BasicUnit):
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()
        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        conv_x = self.mobile_inverted_conv(x)
        if self.shortcut is not None:
            return conv_x + self.shortcut(x)
        return conv_x

    def __str__(self):
        return "(%s, %s)" % (str(self.mobile_inverted_conv),
                             str(self.shortcut) if self.shortcut is not None else None)

    @property
    def config(self):
        return {
            "name": MobileInvertedResidualBlock.__name__,
            "mobile_inverted_conv": self.mobile_inverted_conv.config,
            "shortcut": self.shortcut.config if self.shortcut is not None else None,
        }

    @classmethod
    def from_config(cls, config):
        mobile_inverted_conv = layer_from_config(config["mobile_inverted_conv"])
        shortcut = layer_from_config(config["shortcut"])
        return cls(mobile_inverted_conv, shortcut)


def layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        ZeroLayer.__name__: ZeroLayer,
    }

    layer_name = layer_config.pop("name")
    return name2layer[layer_name].from_config(layer_config)


def layer_from_name(op_name, in_channels, out_channels, stride, ops_order):
    name2ops = {
        'Identity': lambda in_C, out_C, S: IdentityLayer(in_C, out_C, ops_order=ops_order),
        'Zero': lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    # add MBConv layers
    name2ops.update({
        '3x3_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1),
        '3x3_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 2),
        '3x3_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3),
        '3x3_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 4),
        '3x3_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 5),
        '3x3_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6),
        #######################################################################################
        '5x5_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1),
        '5x5_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 2),
        '5x5_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3),
        '5x5_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 4),
        '5x5_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 5),
        '5x5_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6),
        #######################################################################################
        '7x7_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1),
        '7x7_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 2),
        '7x7_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3),
        '7x7_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 4),
        '7x7_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 5),
        '7x7_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6),
    })
    return name2ops[op_name](in_channels, out_channels, stride)
