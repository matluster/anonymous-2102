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

from trainers.nb101 import validate
from trainers.utils import (
    AuxiliaryCrossEntropyLoss, CyclicIterator, Timer, accuracy, load_checkpoint,
    save_checkpoint, set_running_statistics, write_tensorboard
)

logger = logging.getLogger(__name__)


class Nb101MultiPathParser(Nb101Parser):
    def default_params(self):
        return {
            **super().default_params(),
            "steps": 3,
            "fairnas": False,
            "num_valid_arch": 50,
            "num_arch_intermediate": 50,
            "epochs": 333,
            "eval_every": 100,
        }


def train(model, mutator, loader, criterion, optimizer, scheduler, writer, args, epoch):
    model.train()
    mutator.train()
    meters = AverageMeterGroup()

    logger.info("Current learning rate: %.6f", optimizer.param_groups[0]["lr"])
    global_step = len(loader) * (epoch - 1)
    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        for step_count in range(args.steps):
            sample = mutator.reset()
            if args.fairnas:
                assert args.steps == 3
                if step_count == 0:
                    permutations = [np.random.permutation(3) for k in range(5)]
                for i in range(5):
                    sample["arch"][f"op{i + 1}"] = permutations[i] == step_count
                sample = mutator.reset(mutator._archset_inverse[mutator._build_arch_key(sample)])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if args.aux_weight > 0:
                logits, _ = outputs
            else:
                logits = outputs
            loss.backward()
        metrics = {"acc": accuracy(logits, targets), "loss": loss.item()}
        meters.update(metrics)
        write_tensorboard(writer, "train", metrics, step=global_step)
        global_step += 1
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_frequency == 0 or step + 1 == len(loader):
            logger.info("Epoch [%d/%d] Step [%d/%d]  %s", epoch, args.epochs, step + 1, len(loader), meters)


def main():
    args = Nb101MultiPathParser.parse_configs()

    if args.cifar_split == "40k":
        train_split, valid_split = "train", "val"
    else:
        train_split, valid_split = "augment", "test"

    train_loader = dataloader_cifar("data/cifar10", train_split, args)
    valid_loader = dataloader_cifar("data/cifar10", valid_split, args)
    sanitize_loader = CyclicIterator(dataloader_cifar("data/cifar10", train_split, args, batch_size=args.bn_sanitize_batch_size))
    writer = SummaryWriter(args.tb_dir)

    model = Nb101Network(args)
    mutator = Nb101Mutator(model, validation_size=args.num_valid_arch, seed=args.seed, pruned=args.pruned)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    model.cuda()

    if args.aux_weight > 0:
        criterion = AuxiliaryCrossEntropyLoss(args.aux_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.ending_lr)

    for epoch in range(1, args.epochs + 1):
        train(model, mutator, train_loader, criterion, optimizer, scheduler, writer, args, epoch)
        if (args.eval_every and epoch % args.eval_every == 0) or epoch == args.epochs:
            validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, epoch)
    save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", "final.pth.tar"))


if __name__ == "__main__":
    main()
