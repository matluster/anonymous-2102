from copy import deepcopy

import torch.nn as nn
from nni.nas.pytorch.mutables import LayerChoice

from .ops import PRIMITIVES, OPS, ResNetBasicblock


class Cell(nn.Module):
    NUM_NODES = 4
    ENABLE_VIS = False

    def __init__(self, configs, cell_id, C_in, C_out, stride):
        super(Cell, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(self.NUM_NODES):
            node_ops = nn.ModuleList()
            for j in range(0, i):
                op_choices = [OPS[op](configs, C_in, C_out, stride if j == 0 else 1) for op in PRIMITIVES]
                node_ops.append(LayerChoice(op_choices, key=f"{j}_{i}", reduction="mean"))
            self.layers.append(node_ops)
        self.in_dim = C_in
        self.out_dim = C_out
        self.cell_id = cell_id
        self.stride = stride

    def forward(self, inputs):
        nodes = [inputs]
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]


class Nb201Network(nn.Module):
    def __init__(self, configs):
        super(Nb201Network, self).__init__()
        self.channels = C = configs.stem_out_channels
        self.num_modules = N = configs.num_modules_per_stack
        self.num_labels = 10
        if configs.dataset.startswith("cifar100"):
            self.num_labels = 100
        if configs.dataset == "imagenet-16-120":
            self.num_labels = 120

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=configs.bn_momentum)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(configs, C_prev, C_curr, 2)
            else:
                cell = Cell(configs, i, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, momentum=configs.bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_labels)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits
