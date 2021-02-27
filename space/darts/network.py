import copy
import logging

import torch
import torch.nn as nn
from nni.nas.pytorch.mutables import Mutable

from .ops import OPS, FactorizedReduce, ReLUConvBN, DropPath_


PRIMITIVES = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5"
]

logger = logging.getLogger(__name__)


class AuxiliaryHead(nn.Module):
    def __init__(self, C, num_classes):
        super(AuxiliaryHead, self).__init__()
        if num_classes == 1000:
            # assuming input size 14x14
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
                nn.Conv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        else:
            # assuming input size 8x8
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
                nn.Conv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class LayerWithInputChoice(Mutable):
    def __init__(self, key, channels, n_input_candidates, reduction_indices):
        super(LayerWithInputChoice, self).__init__(key)
        self.choices = nn.ModuleDict()
        self.n_inputs = n_input_candidates
        self.ops = copy.deepcopy(PRIMITIVES)
        self.strides = [2 if i in reduction_indices else 1 for i in range(n_input_candidates)]
        for name in PRIMITIVES:
            self.choices[name] = OPS[name](channels)

    def forward(self, *inputs):
        return self.mutator.on_forward_layer_with_input(self, *inputs)


class Node(nn.Module):
    def __init__(self, key, C, n_prev_nodes, reduction_indices):
        super(Node, self).__init__()
        self.key = key
        self.reduction_indices = reduction_indices
        self.op0 = LayerWithInputChoice(f"{key}/0", C, n_prev_nodes, reduction_indices)
        self.op1 = LayerWithInputChoice(f"{key}/1", C, n_prev_nodes, reduction_indices)

    def forward(self, inputs):
        return self.op0(inputs) + self.op1(inputs)


class Cell(nn.Module):
    def __init__(self, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.n_nodes = 4
        self.cell_type = "reduce" if reduction else "normal"
        logger.info("Cell %s created: channels %d -> %d -> %d, prev reduction %s, %d nodes",
                    self.cell_type, C_prev_prev, C_prev, C, reduction_prev, self.n_nodes)

        assert not (reduction_prev and reduction), "Reduction of prev node and current node cannot be both true."
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 0)

        self.nodes = nn.ModuleList()
        for i in range(2, self.n_nodes + 2):
            self.nodes.append(Node(f"{self.cell_type}/{i}", C, i, [0, 1] if reduction else []))

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for node in self.nodes:
            states.append(node(states))
        return torch.cat(states[2:], dim=1)


class DartsNetwork(nn.Module):

    def __init__(self, args):
        super(DartsNetwork, self).__init__()
        self._layers = args.layers
        self.model_type = "imagenet" if args.num_labels == 1000 else "cifar"
        C = args.init_channels

        if self.model_type == "imagenet":
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            C_prev_prev, C_prev, C_curr = C, C, C
            reduction_prev = True
        elif self.model_type == "cifar":
            self.stem = nn.Sequential(
                nn.Conv2d(3, 3 * C, 3, padding=1, bias=False),
                nn.BatchNorm2d(3 * C)
            )
            C_prev_prev, C_prev, C_curr = 3 * C, 3 * C, C
            reduction_prev = False

        self.cells = nn.ModuleList()
        for i in range(args.layers):
            reduction = False
            if i in [args.layers // 3, 2 * args.layers // 3]:
                C_curr *= 2
                reduction = True
            cell = Cell(C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.n_nodes * C_curr
            if i == 2 * args.layers // 3:
                C_to_auxiliary = C_prev

        self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, args.num_labels)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, args.num_labels)

    def forward(self, inputs):
        if self.model_type == "imagenet":
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == 2 * self._layers // 3:
                if self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self.training:
            return logits, logits_aux
        else:
            return logits

    def drop_path_prob(self, drop_prob):
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.drop_prob = drop_prob
