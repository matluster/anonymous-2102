import copy
import json
import logging
import math
import os
import pickle
import random
from collections import defaultdict

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

from utils import flops_counter
from trainers.nb101 import train, eval_one_arch
from trainers.utils import (
    AuxiliaryCrossEntropyLoss, CyclicIterator, Timer, UniversalEncoder,
    accuracy, load_checkpoint, save_checkpoint, set_running_statistics, write_tensorboard
)

logger = logging.getLogger(__name__)


def _verbose_arch(arch):
    return tuple(arch.astype(np.int).tolist())


def ranking_score(evaluations, mask, args):
    # the higher, the better
    if args.metrics == "accuracy":
        return np.mean(evaluations[mask, 1]).item()
    elif args.metrics == "nondominated":
        success = 0
        pos_evaluations = evaluations[mask]
        neg_evaluations = evaluations[~np.array(mask)]
        total = len(pos_evaluations) * len(neg_evaluations)
        for e1 in pos_evaluations:
            for e2 in neg_evaluations:
                if e1[0] < e2[0] and e1[1] > e2[1]:
                    success += 1
        return success / total
    elif args.metrics == "lstsq":
        x = np.log(evaluations[:, 0])
        A = np.vstack([x, np.ones(len(evaluations))]).T
        y = evaluations[:, 1]
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        logger.info("Least-squares solution: %.6f, %.6f", m, c)
        calibrated = y - m * x - c
        return np.mean(calibrated[mask]).item()
    raise ValueError


def _ranking_score_tester():
    args = Nb101FocusParser.parse_configs()
    for _ in range(100):
        print(ranking_score(np.random.uniform(0, 1, size=(100, 2)),
                            np.random.randint(0, 2, size=100).astype(np.bool), args))


def validate(model, mutator, train_loader, valid_loader, criterion, writer, args, epoch):
    model_checkpoint = copy.deepcopy(model.state_dict())
    mutator.eval()
    num_arch = min(args.num_arch_intermediate, len(mutator))
    evaluations = np.zeros((num_arch, 2))
    classifiers = defaultdict(list)
    for i, arch in zip(range(num_arch), mutator.iterative_reset()):
        with Timer() as timer_finetune:
            set_running_statistics(args, model, train_loader)
        with Timer() as timer_eval:
            loss, top1 = eval_one_arch(model, valid_loader, criterion)
        with Timer() as timer_restore:
            model.load_state_dict(copy.deepcopy(model_checkpoint))
        logger.info("Eval Epoch [%d/%d] Arch [%d/%d] Sample: %s", epoch, args.epochs, i + 1, num_arch,
                    json.dumps(arch, cls=UniversalEncoder))
        for k, v in arch["arch"].items():
            classifiers[k].append(_verbose_arch(v))
        evaluations[i, 0] = arch["parameters"]
        evaluations[i, 1] = top1
        logger.info("Eval Epoch [%d/%d] Arch [%d/%d] Time (restore %s sanitize %s eval %s) "
                    "Params = %.3fM Loss = %.6f Acc = %.6f (%.6f)",
                    epoch, args.epochs, i + 1, num_arch, timer_restore,
                    timer_finetune, timer_eval, loss, arch["parameters"], top1, arch["val_acc"])
    scores = []
    for choice_name, choice_list in classifiers.items():
        choice_set = sorted(set(choice_list))
        for choice in choice_set:
            if len([c for c in choice_list if c == choice]) < args.min_arch:
                logger.info("Too few architectures choose %s to be %s. Skipping.", choice_name, choice)
                continue
            if len([c for c in choice_list if c == choice]) == len(choice_list):
                logger.info("Choosing %s to be %s is choosing all architectures. Skipping.", choice_name, choice)
                continue
            branch_score = ranking_score(evaluations, [c == choice for c in choice_list], args)
            scores.append((branch_score, choice_name, choice))
            logger.info("Score [%s = %s]: %.6f", choice_name, choice, branch_score)
    scores.sort(reverse=True)
    avg_result = np.mean(evaluations[:, 1]).item()
    nni.report_intermediate_result(avg_result)
    return avg_result, scores[0][0], scores[0][1], scores[0][2]


class Nb101FocusParser(Nb101Parser):
    def default_params(self):
        return {
            **super().default_params(),
            "metrics": ["nondominated", "lstsq", "accuracy"],
            "eval_every": 500,
            "epochs": 2000,
            "min_arch": 10,
            "num_valid_arch": 150,
            "restart_every": -1,
            "scheduler": ["restart", "annealing"]
        }

    def validate_args(self, args):
        assert args.epochs % args.eval_every == 0
        if args.restart_every < 0:
            args.restart_every = args.epochs
        assert args.epochs % args.restart_every == 0
        return super().validate_args(args)


def pack_creator(args, pruned, model=None, mutator=None, optimizer=None, scheduler=None):
    new_model = Nb101Network(args)
    new_mutator = Nb101Mutator(new_model, validation_size=args.num_valid_arch, seed=args.seed, pruned=pruned)
    new_optimizer = optim.SGD(new_model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.scheduler == "annealing":
        new_scheduler = optim.lr_scheduler.CosineAnnealingLR(new_optimizer, args.restart_every * args.batch_per_epoch, eta_min=args.ending_lr)
    else:
        new_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(new_optimizer, args.restart_every * args.batch_per_epoch, eta_min=args.ending_lr)
    new_model.cuda()
    if model is not None:
        new_model.load_state_dict(model.state_dict())
        new_mutator.load_state_dict(mutator.state_dict())
        new_optimizer.load_state_dict(optimizer.state_dict())
        new_scheduler.load_state_dict(scheduler.state_dict())
    return new_model, new_mutator, new_optimizer, new_scheduler


def main():
    args = Nb101FocusParser.parse_configs()

    if args.cifar_split == "40k":
        train_split, valid_split = "train", "val"
    else:
        train_split, valid_split = "augment", "test"

    train_loader = dataloader_cifar("data/cifar10", train_split, args)
    valid_loader = dataloader_cifar("data/cifar10", valid_split, args)
    sanitize_loader = CyclicIterator(dataloader_cifar("data/cifar10", train_split, args, batch_size=args.bn_sanitize_batch_size))
    writer = SummaryWriter(args.tb_dir)

    args.batch_per_epoch = len(train_loader)
    model, mutator, optimizer, scheduler = pack_creator(args, args.pruned)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    if args.aux_weight > 0:
        criterion = AuxiliaryCrossEntropyLoss(args.aux_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    pruned = copy.deepcopy(mutator.pruned)
    for epoch in range(args.epochs):
        train(model, mutator, train_loader, criterion, optimizer, scheduler, writer, args, epoch + 1)
        if (epoch + 1) % args.eval_every == 0:
            _, branch_score, branch_base, branch_selection = validate(model, mutator, sanitize_loader, valid_loader,
                                                                      criterion, writer, args, epoch + 1)
            save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", f"{epoch + 1:06d}.pth.tar"))
            logger.info("Selection score: %.6f (%s = %s)", branch_score, branch_base, branch_selection)
            pruned[branch_base] = [branch_selection]
            model, mutator, optimizer, scheduler = pack_creator(args, pruned, model, mutator, optimizer, scheduler)


if __name__ == "__main__":
    main()
