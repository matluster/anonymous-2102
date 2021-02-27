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

from trainers.nb101 import train, validate
from trainers.utils import (
    AuxiliaryCrossEntropyLoss, CyclicIterator, Timer, accuracy, load_checkpoint,
    save_checkpoint, set_running_statistics, write_tensorboard
)

logger = logging.getLogger(__name__)


class Nb101SplitParser(Nb101Parser):
    def default_params(self):
        return {
            **super().default_params(),
            "split_type": ["shuffle", "structured", "val_acc"],
            "split_seed": 0,
            "split_op_name": ["op1", "op2", "op3", "op4", "op5"],
            "split_index": 0,
            "split_folds": 3,
        }


def obtain_split(args):
    with open("data/nb101/nb1shot1c.pkl", "rb") as f:
        data = pickle.load(f)
    length = len(data)
    folds = args.split_folds
    if args.split_type == "structured":
        folds = [[d for d in data if d["arch"][args.split_op_name][k]] for k in range(3)]
    if args.split_type == "shuffle":
        random_state = random.Random(args.split_seed)
        random_state.shuffle(data)
        folds = [data[length // folds * k:length // folds * (k + 1)] for k in range(folds)]
    if args.split_type == "val_acc":
        data.sort(key=lambda d: np.mean(d["val_acc"]))
        folds = [data[length // folds * k:length // folds * (k + 1)] for k in range(folds)]
    assert sum(map(len, folds)) == len(data)
    return folds[args.split_index]


def main():
    args = Nb101SplitParser.parse_configs()

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

    validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, 0)
    for epoch in range(1, args.epochs + 1):
        train(model, mutator, train_loader, criterion, optimizer, scheduler, writer, args, epoch)
        if (args.eval_every and epoch % args.eval_every == 0) or epoch == args.epochs:
            validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, epoch)
    save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", "final.pth.tar"))


if __name__ == "__main__":
    main()
