import copy
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from configs import DartsCifarParser
from datasets.cifar10 import dataloader_cifar
from nni.nas.pytorch.utils import AverageMeterGroup
from utils import flops_counter
from space.darts import DartsMutator, DartsNetwork

from .utils import (AuxiliaryCrossEntropyLoss, CyclicIterator, LabelSmoothingLoss, Timer,
                    UniversalEncoder, accuracy, load_checkpoint, reduce_metrics, reduce_tensor, save_checkpoint,
                    set_running_statistics, write_tensorboard)

logger = logging.getLogger(__name__)


def train(model, mutator, loader, criterion, optimizer, scheduler, writer, args, epoch):
    model.train()
    mutator.train()
    meters = AverageMeterGroup()

    if args.drop_path_prob > 0:
        drop_prob = args.drop_path_prob * epoch / args.epochs
        if hasattr(model, "drop_path_prob"):
            model.drop_path_prob(drop_prob)
        else:
            model.module.drop_path_prob(drop_prob)
        logger.info("Current drop path prob: %.6f", drop_prob)

    logger.info("Current learning rate: %.6f", optimizer.param_groups[0]["lr"])
    global_step = len(loader) * (epoch - 1)
    model_params = tuple(model.parameters())
    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        if args.parallel_sampling and args.distributed:
            for _ in range(args.rank + 1):
                mutator.reset()
            logits, aux_logits = model(inputs)
            for _ in range(args.rank + 1, args.world_size):
                mutator.reset()
        else:
            mutator.reset()
            logits, aux_logits = model(inputs)
        loss = criterion((logits, aux_logits), targets)
        loss.backward()
        top1 = accuracy(logits, targets)
        metrics = reduce_metrics({"loss": loss.item(), "acc1": top1}, args.distributed)
        meters.update(metrics)
        write_tensorboard(writer, "train", metrics, step=global_step)
        global_step += 1
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_frequency == 0 or step + 1 == len(loader):
            logger.info("Epoch [%d/%d] Step [%d/%d]  %s", epoch, args.epochs, step + 1, len(loader), meters)

    if args.parallel_sampling:
        torch.distributed.barrier()


def validate(model, mutator, train_loader, valid_loader, criterion, writer, args, epoch):
    model_checkpoint = copy.deepcopy(model.state_dict())
    mutator.eval()
    num_arch = min(args.num_arch_intermediate, len(mutator))
    top1_, flops_, params_ = np.zeros(num_arch), np.zeros(num_arch), np.zeros(num_arch)
    architectures = []
    for i, arch in zip(range(args.num_arch_intermediate), mutator.iterative_reset()):
        architectures.append(json.dumps(arch, cls=UniversalEncoder))
        if args.distributed and i % args.world_size != args.rank:
            continue
        with Timer() as timer_finetune:
            flops, params = flops_counter(model, (1, 3, 32, 32))
            if args.bn_sanitize_steps > 0:
                set_running_statistics(args, model, train_loader)
        with Timer() as timer_eval:
            loss, top1 = eval_one_arch(model, valid_loader, criterion)
        top1_[i] = top1
        flops_[i] = flops
        params_[i] = params
        with Timer() as timer_restore:
            model.load_state_dict(copy.deepcopy(model_checkpoint))
        logger.info("Eval Progress Epoch [%d/%d] Arch [%d/%d] Time (restore %.3fs sanitize %.3fs eval %.3fs) "
                    "Loss = %.6f Acc = %.6f FLOPS = %.3fM Params = %.3fM",
                    epoch, args.epochs, i + 1, num_arch, timer_restore.elapsed_secs,
                    timer_finetune.elapsed_secs, timer_eval.elapsed_secs, loss, top1, flops / 1e6, params / 1e6)
    if args.distributed:
        top1_ = reduce_tensor(top1_, reduction="sum").cpu().numpy()
        flops_ = reduce_tensor(flops_, reduction="sum").cpu().numpy()
        params_ = reduce_tensor(params_, reduction="sum").cpu().numpy()
    if writer is not None:
        writer.add_histogram("eval/top1_supernet", top1_, global_step=epoch)
        writer.add_scalar("eval/top1_avg", np.mean(top1_), global_step=epoch)
    avg_result = np.mean(top1_).item()
    for i, arch, top1, flops, params in zip(range(num_arch), architectures, top1_, flops_, params_):
        logger.info("Eval Rank Epoch [%d/%d] Arch [%d/%d] Acc = %.6f FLOPS = %.0f Params = %.0f Sample: %s",
                    epoch, args.epochs, i + 1, num_arch, top1, flops, params, arch)
    logger.info("Eval Rank Epoch [%d/%d] Average = %.6f Evaluation Result: [%s]",
                epoch, args.epochs, avg_result, ", ".join("%.6f" % a for a in top1_))
    del model_checkpoint
    torch.cuda.empty_cache()
    return avg_result


def eval_one_arch(model, loader, criterion):
    model.eval()
    loss_tot = correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)
            loss = criterion(logits, targets)
            top1 = accuracy(logits, targets)
            correct += top1 * inputs.size(0)
            loss_tot += loss.item() * inputs.size(0)
            total += inputs.size(0)
            logger.debug("Eval One Arch: correct %.6f total %.0f loss %.6f", correct, total, loss_tot)
    return loss_tot / total, correct / total


def main():
    args = DartsCifarParser.parse_configs()

    if args.cifar_split == "40k":
        train_split, valid_split = "train", "val"
    else:
        train_split, valid_split = "augment", "test"

    train_loader = dataloader_cifar("data/cifar10", train_split, args, distributed=args.distributed)
    valid_loader = dataloader_cifar("data/cifar10", valid_split, args)
    sanitize_loader = CyclicIterator(dataloader_cifar("data/cifar10", train_split, args, batch_size=args.bn_sanitize_batch_size))
    writer = SummaryWriter(args.tb_dir)

    model = DartsNetwork(args)
    mutator = DartsMutator(model, validation_size=args.num_valid_arch, seed=args.seed, pruned=args.pruned)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    model.cuda()

    learning_rate = args.initial_lr
    if args.distributed:
        from apex.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, delay_allreduce=True)
        logger.info("Converted to DDP.")
        torch.distributed.barrier()
        learning_rate *= args.world_size

    criterion = nn.CrossEntropyLoss()
    criterion = AuxiliaryCrossEntropyLoss(args.aux_weight, criterion)
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=0.)
    last_epoch = 0

    if args.distributed:
        torch.distributed.barrier()
    auto_resume = os.path.join(args.output_dir, "checkpoints", "latest.pth.tar")
    if os.path.exists(auto_resume):
        auto_resume_ckpt = load_checkpoint(model, auto_resume, args=args, optimizer=optimizer, scheduler=scheduler)
        if auto_resume_ckpt is not None:
            last_epoch = auto_resume_ckpt["epoch"]
            logger.info("Resume from checkpoint. Proceeding from epoch %d...", last_epoch)

    if args.distributed and args.parallel_sampling:
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

    validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, last_epoch)
    for epoch in range(last_epoch + 1, args.epochs + 1):
        train(model, mutator, train_loader, criterion, optimizer, scheduler, writer, args, epoch)
        if (args.eval_every and epoch % args.eval_every == 0) or epoch == args.epochs:
            validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, epoch)
            if args.is_worker_main:
                save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch:06d}.pth.tar"))
        if args.is_worker_main and epoch % 5 == 0:
            save_checkpoint(args, model, auto_resume, optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), epoch=epoch)


if __name__ == "__main__":
    main()
