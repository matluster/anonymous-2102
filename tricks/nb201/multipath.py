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

from trainers.nb201 import validate
from trainers.utils import (
    AuxiliaryCrossEntropyLoss, CyclicIterator, Timer, accuracy, load_checkpoint,
    save_checkpoint, set_running_statistics, write_tensorboard
)

logger = logging.getLogger(__name__)


class Nb201MultiPathParser(Nb201Parser):
    def default_params(self):
        return {
            **super().default_params(),
            "steps": 5,
            "fairnas": False,
            "num_valid_arch": 50,
            "num_arch_intermediate": 50,
            "epochs": 400,
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
                assert args.steps == 5
                if step_count == 0:
                    permutations = [np.random.permutation(5).tolist() for k in range(6)]
                i_count = 0
                for i in range(4):
                    for j in range(i):
                        sample["arch"][f"{j}_{i}"] = permutations[i_count][step_count]
                        i_count += 1
                mutator._cache = sample["arch"]  # could be not found
            logits = model(inputs)
            loss = criterion(logits, targets)
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
    args = Nb201MultiPathParser.parse_configs()

    train_loader, valid_loader, test_loader = nb201_dataloader(args)
    sanitize_loader, _, _ = nb201_dataloader(args)
    sanitize_loader = CyclicIterator(sanitize_loader)
    writer = SummaryWriter(args.tb_dir)

    model = Nb201Network(args)
    mutator = Nb201Mutator(model, args.dataset, validation_size=args.num_valid_arch, seed=args.seed, pruned=args.pruned)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.ending_lr)

    for epoch in range(1, args.epochs + 1):
        train(model, mutator, train_loader, criterion, optimizer, scheduler, writer, args, epoch)
        if (args.eval_every and epoch % args.eval_every == 0) or epoch == args.epochs:
            validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, epoch)
    save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", "final.pth.tar"))


if __name__ == "__main__":
    main()
