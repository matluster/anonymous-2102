import logging
import math

import torch
import torch.nn as nn
from nni.nas.pytorch.mutables import LayerChoice, InputChoice, MutableScope

from .base_ops import Conv3x3BnRelu, Conv1x1BnRelu, MaxPool3x3, ConvBnRelu

logger = logging.getLogger(__name__)


class DropPath_(nn.Module):
    # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            # per data point mask; assuming x in cuda.
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            return x.div(keep_prob).mul(mask)
        return x


class AuxiliaryHead(nn.Module):
    # Auxiliary head in 2/3 place of network to let the gradient flow well

    def __init__(self, args, C):
        super().__init__()
        self.input_size = 8
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=self.input_size - 5, padding=0, count_include_pad=False),  # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),  # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, args.num_labels)

    def forward(self, x):
        assert x.size(2) == self.input_size and x.size(3) == self.input_size
        out = self.net(x)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits


class OneShotNode(MutableScope):
    def __init__(self, configs, key, channels, in_channels, num_parent_nodes, prev_keys, last_node):
        super(OneShotNode, self).__init__(f"node{key}")
        self.last_node = last_node
        # select input node for current node
        self.switch = InputChoice(choose_from=prev_keys, n_chosen=num_parent_nodes, key=f"input{key}",
                                  reduction="concat" if last_node else "sum", return_mask=True)
        # having an input_op for each node to transform the input to align with current node
        self.input_op = ConvBnRelu(configs, in_channels, channels, kernel_size=1)

        if not last_node:
            # operation for each node except last (concat)
            self.op = LayerChoice([
                Conv3x3BnRelu(configs, channels, channels),
                Conv1x1BnRelu(configs, channels, channels),
                MaxPool3x3(configs, channels, channels)
            ], key=f"op{key}")
            self.drop_path = None
            if configs.drop_path_prob > 0:
                self.drop_path = DropPath_(configs.drop_path_prob)

    def forward(self, tensors):
        if self.last_node:
            # assuming tensors[0] will never be chosen.
            # otherwise this doesn't align with nasbench.
            out, mask = self.switch(tensors)
            assert not mask[0]
        else:
            processed_input_node = self.input_op(tensors[0])
            input_sum, mask = self.switch([processed_input_node] + tensors[1:])
            if hasattr(self.input_op, "redundant_op") and not mask[0]:
                # input should be avoided from calculation
                self.input_op.redundant_op = True
            out = self.op(input_sum)
            if self.drop_path:
                out = self.drop_path(out)
        return out


class OneShotCell(nn.Module):

    def __init__(self, configs, in_channels, out_channels):
        super(OneShotCell, self).__init__()
        self.num_vertices = len(configs.num_parents)
        assert out_channels % configs.num_parents[-1] == 0, "One shot cell must have all equal channels."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vertex_channels = out_channels // configs.num_parents[-1]

        self.nodes = nn.ModuleList()
        prev_keys = [InputChoice.NO_KEY]  # for first node
        for t in range(1, self.num_vertices):
            self.nodes.append(OneShotNode(configs, str(t), self.vertex_channels, in_channels,
                                          configs.num_parents[t], prev_keys, t + 1 == self.num_vertices))
            prev_keys.append(self.nodes[-1].key)

    def forward(self, x):
        tensors = [x]
        for node in self.nodes:
            tensors.append(node(tensors))
        out = tensors[-1]
        return out


class Nb101Network(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # initial stem convolution
        self.stem_conv = Conv3x3BnRelu(configs, 3, configs.stem_out_channels)
        self.aux_pos = -1

        layers = []
        in_channels = out_channels = configs.stem_out_channels
        for stack_num in range(configs.num_stacks):
            if stack_num == 2 and configs.aux_weight:
                self.aux_head = AuxiliaryHead(configs, out_channels)
                self.aux_pos = len(layers)
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                layers.append(downsample)
                out_channels *= 2
            for _ in range(configs.num_modules_per_stack):
                cell = OneShotCell(configs, in_channels, out_channels)
                layers.append(cell)
                in_channels = out_channels

        self.features = nn.ModuleList(layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(configs.dropout_rate)
        self.classifier = nn.Linear(out_channels, configs.num_labels)

        self.reset_parameters()

    def forward(self, x):
        bs = x.size(0)
        aux_logits = None
        out = self.stem_conv(x)
        feature_maps = [out]
        for i, layer in enumerate(self.features):
            out = layer(out)
            feature_maps.append(out)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(out)
        out = self.gap(out).view(bs, -1)
        out = self.dropout(out)
        out = self.classifier(out)

        if aux_logits is not None:
            return out, aux_logits
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()

    def drop_path_prob(self, drop_prob):
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.drop_prob = drop_prob
