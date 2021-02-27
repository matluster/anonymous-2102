import copy
import json
import logging
import math
import os
import pickle
import random

import numpy as np
import nni
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from nni.nas.pytorch.mutables import LayerChoice
from nni.nas.pytorch.utils import AverageMeterGroup
from torch.utils.tensorboard import SummaryWriter

from configs import Nb201Parser
from datasets.nb201 import nb201_dataloader
from space.nb201 import Nb201Mutator, Nb201Network
from space.nb201.network import PRIMITIVES, Cell, OPS

from trainers.standard import train_loop
from trainers.nb201 import train, validate
from trainers.utils import (
    AuxiliaryCrossEntropyLoss, CyclicIterator, Timer, accuracy, load_checkpoint,
    save_checkpoint, set_running_statistics, write_tensorboard
)

logger = logging.getLogger(__name__)


class Nb201AugmentedSpaceParser(Nb201Parser):
    def default_params(self):
        return {
            **super().default_params(),
            "setup": "444444",
            "replicate": False,
            "num_valid_arch": 1000,
            "num_arch_intermediate": 50,
            "epochs": 2000,
            "eval_every": 200,
            "eval": {"default": False, "action": "store_true"},
            "initial_lr": 5e-2,
            "weight_decay": 5e-4,
            "nesterov": True,
        }

    def validate_args(self, args):
        assert len(args.setup) == 6 and all(x in "01234" for x in args.setup)
        args.setup = [int(x) for x in args.setup]
        assert args.pruned == {}
        return super().validate_args(args)


class Nb201ReplicateMutator(Nb201Mutator):
    def on_forward_layer_choice(self, mutable, *inputs):
        return mutable.choices[self.random_state.randint(5)](*inputs), None


def main():
    args = Nb201AugmentedSpaceParser.parse_configs()

    train_loader, valid_loader, test_loader = nb201_dataloader(args)
    sanitize_loader, _, _ = nb201_dataloader(args)
    sanitize_loader = CyclicIterator(sanitize_loader)
    writer = SummaryWriter(args.tb_dir)

    model = Nb201Network(args)
    pruned = {}
    counter = 0
    for i in range(4):
        for j in range(i):
            pruned[f"{j}_{i}"] = [args.setup[counter]]
            counter += 1
    logger.info("Parsed pruned: %s", pruned)
    if args.replicate:
        for name, module in model.named_modules():
            if isinstance(module, Cell):
                for i in range(4):
                    for j in range(i):
                        assert isinstance(module.layers[i][j], LayerChoice)
                        chosen = PRIMITIVES[pruned[f"{j}_{i}"][0]]
                        logger.info("Module %s layer %d.%d choosing %s", name, i, j, chosen)
                        new_choice_list = [OPS[chosen](args, module.in_dim, module.out_dim, module.stride if j == 0 else 1)
                                           for _ in range(5)]
                        module.layers[i][j].choices = nn.ModuleList(new_choice_list)
        mutator = Nb201ReplicateMutator(model, args.dataset, validation_size=args.num_valid_arch, seed=args.seed, pruned=pruned)
    else:
        mutator = Nb201Mutator(model, args.dataset, validation_size=args.num_valid_arch, seed=args.seed, pruned=pruned)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.ending_lr)

    train_loop(model, mutator, criterion, optimizer, scheduler,
               train_loader, sanitize_loader, valid_loader,
               train, validate, writer, args)


if __name__ == "__main__":
    main()
