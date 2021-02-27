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
from nni.nas.pytorch.utils import AverageMeterGroup
from torch.utils.tensorboard import SummaryWriter

from configs import Nb101Parser
from datasets.cifar10 import dataloader_cifar
from space.nb101 import Nb101Mutator, Nb101Network
from space.nb101.base_ops import Conv3x3BnRelu, Conv1x1BnRelu, MaxPool3x3
from space.nb101.network import OneShotCell

from trainers.standard import train_loop
from trainers.nb101 import train, validate
from trainers.utils import (
    AuxiliaryCrossEntropyLoss, CyclicIterator, Timer,
    accuracy, load_checkpoint, save_checkpoint, set_running_statistics, write_tensorboard
)

logger = logging.getLogger(__name__)


class Nb101AugmentedSpaceParser(Nb101Parser):
    def default_params(self):
        return {
            **super().default_params(),
            "setup": "00000",
            "replicate": False,
            "num_valid_arch": 1000,
            "num_arch_intermediate": 50,
            "epochs": 1000,
            "eval_every": 100,
            "eval": {"default": False, "action": "store_true"},
        }

    def validate_args(self, args):
        assert len(args.setup) == 5 and all(x in ["0", "1", "2"] for x in args.setup)
        args.setup = [int(x) for x in args.setup]
        assert args.pruned == {}
        return super().validate_args(args)


class Nb101ReplicateMutator(Nb101Mutator):
    def on_forward_layer_choice(self, mutable, *inputs):
        return mutable.choices[self.random_state.randint(3)](*inputs), None


def main():
    args = Nb101AugmentedSpaceParser.parse_configs()

    if args.cifar_split == "40k":
        train_split, valid_split = "train", "val"
    else:
        train_split, valid_split = "augment", "test"

    train_loader = dataloader_cifar("data/cifar10", train_split, args)
    valid_loader = dataloader_cifar("data/cifar10", valid_split, args)
    sanitize_loader = CyclicIterator(dataloader_cifar("data/cifar10", train_split, args, batch_size=args.bn_sanitize_batch_size))
    writer = SummaryWriter(args.tb_dir)

    model = Nb101Network(args)
    pruned = {f"op{i + 1}": [[args.setup[i] == k for k in range(3)]] for i in range(5)}
    if args.replicate:
        for name, module in model.named_modules():
            if isinstance(module, OneShotCell):
                for i in range(5):
                    channels = module.nodes[i].op.choices[0].out_channels
                    logger.info("Module %s, node %d, %d channels.", name, i, channels)
                    if args.setup[i] == 0:
                        new_choice_list = [Conv3x3BnRelu(args, channels, channels) for _ in range(3)]
                    elif args.setup[i] == 1:
                        new_choice_list = [Conv1x1BnRelu(args, channels, channels) for _ in range(3)]
                    elif args.setup[i] == 2:
                        new_choice_list = [MaxPool3x3(args, channels, channels) for _ in range(3)]
                    else:
                        raise ValueError
                    module.nodes[i].op.choices = nn.ModuleList(new_choice_list)
        mutator = Nb101ReplicateMutator(model, validation_size=args.num_valid_arch, seed=args.seed, pruned=pruned)
    else:
        mutator = Nb101Mutator(model, validation_size=args.num_valid_arch, seed=args.seed, pruned=pruned)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    model.cuda()

    if args.aux_weight > 0:
        criterion = AuxiliaryCrossEntropyLoss(args.aux_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.ending_lr)

    train_loop(model, mutator, criterion, optimizer, scheduler,
               train_loader, sanitize_loader, valid_loader,
               train, validate, writer, args)


if __name__ == "__main__":
    main()
