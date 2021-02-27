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

from trainers.standard import train_loop
from trainers.nb101 import train, validate
from trainers.utils import (
    AuxiliaryCrossEntropyLoss, CyclicIterator, Timer, accuracy, load_checkpoint,
    save_checkpoint, set_running_statistics, write_tensorboard
)

logger = logging.getLogger(__name__)


class Nb101OmniscienceParser(Nb101Parser):
    def default_params(self):
        return {
            **super().default_params(),
            "ratio": 0.33,
            "num_valid_arch": 1000,
            "num_arch_intermediate": 50,
            "epochs": 1000,
            "eval_every": 100,
            "eval": {"default": False, "action": "store_true"},
            "random": False,
        }


def obtain_split(args):
    with open("data/nb101/nb1shot1c.pkl", "rb") as f:
        data = pickle.load(f)
    length = len(data)
    if args.random:
        random.shuffle(data)
    else:
        data.sort(key=lambda d: np.mean(d["val_acc"]), reverse=True)
    return data[:int(length * args.ratio)]


def main():
    args = Nb101OmniscienceParser.parse_configs()

    if args.cifar_split == "40k":
        train_split, valid_split = "train", "val"
    else:
        train_split, valid_split = "augment", "test"

    train_loader = dataloader_cifar("data/cifar10", train_split, args)
    valid_loader = dataloader_cifar("data/cifar10", valid_split, args)
    sanitize_loader = CyclicIterator(dataloader_cifar("data/cifar10", train_split, args, batch_size=args.bn_sanitize_batch_size))
    writer = SummaryWriter(args.tb_dir)

    model = Nb101Network(args)
    split = obtain_split(args)
    logger.info("Generated split. %d candidate architectures.", len(split))
    mutator = Nb101Mutator(model, archset=split, validation_size=args.num_valid_arch, seed=args.seed, pruned=args.pruned)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    model.cuda()

    if args.aux_weight > 0:
        criterion = AuxiliaryCrossEntropyLoss(args.aux_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.ending_lr)

    if args.eval:
        validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, 0)
        return

    train_loop(model, mutator, criterion, optimizer, scheduler,
               train_loader, sanitize_loader, valid_loader,
               train, validate, writer, args)


if __name__ == "__main__":
    main()
