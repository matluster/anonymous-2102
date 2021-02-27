import copy
import json
import logging
import math
import os
import pickle

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
from .utils import (
    AuxiliaryCrossEntropyLoss, CyclicIterator, Timer, UniversalEncoder,
    accuracy, load_checkpoint, save_checkpoint, set_running_statistics, write_tensorboard
)


def train(model, mutator, loader, criterion, optimizer, scheduler, writer, args, epoch):
    logger = logging.getLogger(__name__)
    model.train()
    mutator.train()
    meters = AverageMeterGroup()

    logger.info("Current learning rate: %.6f", optimizer.param_groups[0]["lr"])
    global_step = len(loader) * (epoch - 1)
    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        mutator.reset()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
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


def validate(model, mutator, train_loader, valid_loader, criterion, writer, args, epoch):
    model_checkpoint = copy.deepcopy(model.state_dict())
    logger = logging.getLogger(__name__)
    gt_top1, eval_top1 = [], []
    mutator.eval()
    if epoch == args.epochs:
        num_arch = args.num_valid_arch
    else:
        num_arch = args.num_arch_intermediate
    num_arch = min(num_arch, len(mutator))
    for i, arch in zip(range(num_arch), mutator.iterative_reset()):
        gt_top1.append(arch["val_acc"])
        with Timer() as timer_finetune:
            if args.bn_sanitize_steps > 0:
                set_running_statistics(args, model, train_loader)
        with Timer() as timer_eval:
            loss, top1 = eval_one_arch(model, valid_loader, criterion)
        eval_top1.append(top1)
        with Timer() as timer_restore:
            model.load_state_dict(copy.deepcopy(model_checkpoint))
        if epoch == args.epochs:
            logger.info("Eval Epoch [%d/%d] Arch [%d/%d] Sample: %s", epoch, args.epochs, i + 1, num_arch,
                        json.dumps(arch, cls=UniversalEncoder))
        logger.info("Eval Epoch [%d/%d] Arch [%d/%d] Time (restore %s sanitize %s eval %s) "
                    "Loss = %.6f Acc = %.6f (%.6f)",
                    epoch, args.epochs, i + 1, num_arch, timer_restore,
                    timer_finetune, timer_eval, loss, top1, arch["val_acc"])
    logger.info("Eval Epoch [%d/%d] Evaluation Result: %s", epoch, args.epochs, eval_top1)
    corr = stats.spearmanr(eval_top1, gt_top1)
    logger.info("Eval Epoch [%d/%d] Corr = %.6f (p = %.6f) Average acc = %.6f±%.6f (vs. %.6f±%.6f)",
                epoch, args.epochs, corr[0], corr[1],
                np.mean(eval_top1), np.std(eval_top1), np.mean(gt_top1), np.std(gt_top1))

    if writer is not None:
        writer.add_scalar("eval/corr", corr[0], global_step=epoch)
        writer.add_scalar("eval/avg_top1", np.mean(eval_top1), global_step=epoch)
        writer.add_histogram("eval/supernet_top1", np.array(eval_top1), global_step=epoch)
        writer.add_histogram("eval/gt_top1", np.array(gt_top1), global_step=epoch)

    avg_result = np.mean(eval_top1).item()
    nni.report_intermediate_result(avg_result)
    return avg_result


def eval_one_arch(model, loader, criterion):
    model.eval()
    correct = loss = total = 0.
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            bs = targets.size(0)
            logits = model(inputs)
            loss += criterion(logits, targets).item() * bs
            correct += accuracy(logits, targets) * bs
            total += bs
    return loss / total, correct / total


def main():
    args = Nb201Parser.parse_configs()

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

    validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, 0)
    for epoch in range(1, args.epochs + 1):
        train(model, mutator, train_loader, criterion, optimizer, scheduler, writer, args, epoch)
        if (args.eval_every and epoch % args.eval_every == 0) or epoch == args.epochs:
            validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, epoch)
    # validate(model, mutator, sanitize_loader, test_loader, criterion, writer, args, epoch)

    save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", "final.pth.tar"))


if __name__ == "__main__":
    main()
