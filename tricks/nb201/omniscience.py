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


class Nb201OmniscienceParser(Nb201Parser):
    def default_params(self):
        return {
            **super().default_params(),
            "ratio": 0.33,
            "num_valid_arch": 1000,
            "num_arch_intermediate": 50,
            "epochs": 2000,
            "eval_every": 100,
            "eval": {"default": False, "action": "store_true"},
            "random": False,
            "initial_lr": 5e-2,
            "weight_decay": 5e-4,
            "nesterov": True,
        }


def obtain_split(args):
    with open("data/nb201/nb201.pkl", "rb") as f:
        data = pickle.load(f)
    length = len(data)
    if args.random:
        random.shuffle(data)
    else:
        data.sort(key=lambda d: np.mean(d[args.dataset][:, 3]), reverse=True)
    return data[:int(length * args.ratio)]


def main():
    args = Nb201OmniscienceParser.parse_configs()

    train_loader, valid_loader, test_loader = nb201_dataloader(args)
    sanitize_loader, _, _ = nb201_dataloader(args)
    sanitize_loader = CyclicIterator(sanitize_loader)
    writer = SummaryWriter(args.tb_dir)

    model = Nb201Network(args)
    split = obtain_split(args)
    mutator = Nb201Mutator(model, args.dataset, archset=split, validation_size=args.num_valid_arch, seed=args.seed, pruned=args.pruned)
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
