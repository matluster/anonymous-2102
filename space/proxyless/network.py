import math

import torch
import torch.nn as nn
from nni.nas.pytorch.mutables import LayerChoice

from .layers import (
    BasicUnit, layer_from_config, layer_from_name, ConvLayer, 
    MobileInvertedResidualBlock, IdentityLayer, LinearLayer
)
from .utils import make_divisible


PRIMITIVES = [
    "3x3_MBConv3",
    "3x3_MBConv6",
    "5x5_MBConv3",
    "5x5_MBConv6",
    "7x7_MBConv3",
    "7x7_MBConv6"
]


class ProxylessNASNet(BasicUnit):
    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNet, self).__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer:
            x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    def __str__(self):
        return "\n".join(map(str, self.blocks))

    @property
    def config(self):
        return {
            "name": ProxylessNASNet.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "feature_mix_layer": self.feature_mix_layer.config if self.feature_mix_layer is not None else None,
            "classifier": self.classifier.config,
            "blocks": [block.config for block in self.blocks],
        }

    @classmethod
    def from_config(cls, config):
        first_conv = layer_from_config(config["first_conv"])
        feature_mix_layer = layer_from_config(config["feature_mix_layer"])
        classifier = layer_from_config(config["classifier"])
        blocks = [MobileInvertedResidualBlock.from_config(cfg) for cfg in config["blocks"]]
        return cls(first_conv, blocks, feature_mix_layer, classifier)

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = momentum
                m.eps = eps
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                return {
                    "momentum": m.momentum,
                    "eps": m.eps,
                }
        return None

    def reset_parameters(self, model_init="he_fout", init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == "he_fout":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == "he_fin":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ProxylessNASSuperNet(ProxylessNASNet):
    def __init__(self, args):
        input_width = make_divisible(32 * args.width_mult, 8)
        first_cell_width = make_divisible(16 * args.width_mult, 8)

        # first conv layer
        first_conv = ConvLayer(
            3, input_width, kernel_size=3, stride=2, use_bn=True, act_func="relu6", ops_order=args.ops_order
        )

        # first block
        first_block = MobileInvertedResidualBlock(
            layer_from_name("3x3_MBConv1", input_width, first_cell_width, 1, args.ops_order), None
        )
        input_width = first_cell_width

        # blocks
        blocks = [first_block]
        for k, (width, n_cell, s) in enumerate(zip(args.width_stages, args.n_cell_stages, args.stride_stages)):
            width = make_divisible(width * args.width_mult, 8)
            for i in range(n_cell):
                stride = s if i == 0 else 1
                # conv and shortcut:
                # if tensor size is changed, shortcut is none, conv cannot be zero
                # in other cells, zero can be selected for conv, shortcut is identity to make sure connectivity
                if stride == 1 and input_width == width:
                    conv_candidates = PRIMITIVES + ["Zero"]
                    shortcut = IdentityLayer(input_width, input_width)
                else:
                    conv_candidates = PRIMITIVES
                    shortcut = None

                conv_op = LayerChoice([
                    layer_from_name(candidate, input_width, width, stride, args.ops_order)
                    for candidate in conv_candidates
                ], key="stage%d_cell%d" % (k, i))
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_width = width

        # feature mix layer
        last_width = make_divisible(1280 * args.width_mult, 8) if args.width_mult > 1 else 1280
        feature_mix_layer = ConvLayer(input_width, last_width, kernel_size=1,
                                      use_bn=True, act_func="relu6", ops_order=args.ops_order)

        classifier = LinearLayer(last_width, args.num_labels, dropout_rate=args.dropout_rate)
        super(ProxylessNASSuperNet, self).__init__(first_conv, blocks, feature_mix_layer, classifier)

        self.set_bn_param(momentum=args.bn_momentum, eps=args.bn_epsilon)
        self.reset_parameters(model_init=args.model_init, init_div_groups=args.init_div_groups)
