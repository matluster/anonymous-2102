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

from configs import Nb201Parser
from datasets.nb201 import nb201_dataloader
from space.nb201 import Nb201Mutator, Nb201Network

from trainers.nb201 import train, validate
from trainers.utils import (
    AuxiliaryCrossEntropyLoss, CyclicIterator, Timer, accuracy, load_checkpoint,
    save_checkpoint, set_running_statistics, write_tensorboard
)

logger = logging.getLogger(__name__)


class Nb201SplitParser(Nb201Parser):
    def default_params(self):
        return {
            **super().default_params(),
            "split_type": ["shuffle", "structured", "val_acc"],
            "split_seed": 0,
            "split_op_name": ["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"],
            "split_index": 0,
            "split_folds": 5,
        }


def obtain_split(args):
    with open("data/nb201/nb201.pkl", "rb") as f:
        data = pickle.load(f)
    length = len(data)
    folds = args.split_folds
    if args.split_type == "structured":
        folds = [[d for d in data if d["arch"][args.split_op_name] == k] for k in range(folds)]
    if args.split_type == "shuffle":
        random_state = random.Random(args.split_seed)
        random_state.shuffle(data)
        folds = [data[length // folds * k:length // folds * (k + 1)] for k in range(folds)]
    if args.split_type == "val_acc":
        data.sort(key=lambda d: np.mean(d[args.dataset][:, 3]))
        folds = [data[length // folds * k:length // folds * (k + 1)] for k in range(folds)]
    return folds[args.split_index]


def main():
    args = Nb201SplitParser.parse_configs()

    train_loader, valid_loader, test_loader = nb201_dataloader(args)
    sanitize_loader, _, _ = nb201_dataloader(args)
    sanitize_loader = CyclicIterator(sanitize_loader)
    writer = SummaryWriter(args.tb_dir)

    model = Nb201Network(args)
    split = obtain_split(args)
    logger.info("Generated split. %d candidate architectures.", len(split))
    mutator = Nb201Mutator(model, args.dataset, archset=split, validation_size=args.num_valid_arch, seed=args.seed, pruned=args.pruned)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.ending_lr)

    validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, 0)
    for epoch in range(1, args.epochs + 1):
        train(model, mutator, train_loader, criterion, optimizer, scheduler, writer, args, epoch)
        if (args.eval_every and epoch % args.eval_every == 0) or epoch == args.epochs:
            validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, epoch)
    save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", "final.pth.tar"))


if __name__ == "__main__":
    main()
