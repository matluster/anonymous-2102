import copy
import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import nni
from apex.parallel import DistributedDataParallel
from configs import ProxylessParser
from datasets.imagenet import imagenet_loader_gpu
from nni.nas.pytorch.utils import AverageMeterGroup
from utils import flops_counter
from space.proxyless import ProxylessMutator, ProxylessNASSuperNet

from .utils import (LabelSmoothingLoss, Timer, UniversalEncoder,
                    accuracy, adjust_learning_rate, load_checkpoint,
                    reduce_metrics, reduce_tensor, save_checkpoint,
                    set_running_statistics, write_tensorboard)

logger = logging.getLogger(__name__)


def train(model, mutator, loader, criterion, optimizer, writer, args, epoch):
    model.train()
    mutator.train()
    meters = AverageMeterGroup()

    logger.info("Current learning rate: %.6f", optimizer.param_groups[0]["lr"])
    global_step = len(loader) * (epoch - 1)
    for step, (inputs, targets) in enumerate(loader):
        learning_rate = adjust_learning_rate(args, optimizer, epoch - 1 + step / len(loader))
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        if args.parallel_sampling and args.distributed:
            for _ in range(args.rank + 1):
                mutator.reset()
            logits = model(inputs)
            for _ in range(args.rank + 1, args.world_size):
                mutator.reset()
        else:
            mutator.reset()
            logits = model(inputs)

        loss = criterion(logits, targets)
        loss.backward()
        top1, top5 = accuracy(logits, targets, topk=(1, 5))
        metrics = reduce_metrics({"loss": loss.item(), "acc1": top1, "acc5": top5}, args.distributed)
        metrics["lr"] = learning_rate
        meters.update(metrics)
        write_tensorboard(writer, "train", metrics, step=global_step)
        global_step += 1
        optimizer.step()

        if step % args.log_frequency == 0 or step + 1 == len(loader):
            logger.info("Epoch [%d/%d] Step [%d/%d]  %s", epoch, args.epochs, step + 1, len(loader), meters)

    if args.parallel_sampling and args.distributed:
        torch.distributed.barrier()


def validate(model, mutator, train_loader, valid_loader, criterion, writer, args, epoch, prefix="Valid"):
    model_checkpoint = copy.deepcopy(model.state_dict())
    mutator.eval()
    if epoch == args.epochs:
        num_arch = args.num_valid_arch
    else:
        num_arch = args.num_arch_intermediate
    num_arch = min(num_arch, len(mutator))
    flops_, params_ = np.zeros(num_arch), np.zeros(num_arch)
    top1_, top5_ = np.zeros(num_arch), np.zeros(num_arch)
    architectures = []
    for i, arch in zip(range(num_arch), mutator.iterative_reset()):
        architectures.append(json.dumps(arch, cls=UniversalEncoder))
        if args.distributed and i % args.world_size != args.rank:
            continue
        with Timer() as timer_finetune:
            flops, params = flops_counter(model, (1, 3, 224, 224))
            if args.bn_sanitize_steps > 0:
                set_running_statistics(args, model, train_loader)
        with Timer() as timer_eval:
            loss, top1, top5 = eval_one_arch(model, valid_loader, criterion)
        top1_[i] = top1
        top5_[i] = top5
        flops_[i] = flops
        params_[i] = params
        with Timer() as timer_restore:
            model.load_state_dict(copy.deepcopy(model_checkpoint))
        logger.info("[%s] Eval Progress Epoch [%d/%d] Arch [%d/%d] Time (restore %.3fs sanitize %.3fs eval %.3fs) "
                    "Loss = %.6f Acc = %.6f",
                    prefix, epoch, args.epochs, i + 1, num_arch, timer_restore.elapsed_secs,
                    timer_finetune.elapsed_secs, timer_eval.elapsed_secs, loss, top1)
    if args.distributed:
        top1_ = reduce_tensor(top1_, reduction="sum").cpu().numpy()
        top5_ = reduce_tensor(top5_, reduction="sum").cpu().numpy()
        flops_ = reduce_tensor(flops_, reduction="sum").cpu().numpy()
        params_ = reduce_tensor(params_, reduction="sum").cpu().numpy()
    if writer is not None:
        writer.add_histogram("eval/top1_supernet", top1_, global_step=epoch)
        writer.add_histogram("eval/top5_supernet", top5_, global_step=epoch)
        writer.add_scalar("eval/top1_avg", np.mean(top1_), global_step=epoch)
        writer.add_scalar("eval/top5_avg", np.mean(top5_), global_step=epoch)
    avg_result = np.mean(top1_).item()
    for i, arch, top1, top5, flops, params in zip(range(num_arch), architectures, top1_, top5_, flops_, params_):
        logger.info("[%s] Eval Rank Epoch [%d/%d] Arch [%d/%d] Top1 = %.6f Top5 = %.6f FLOPS = %.0f Params = %.0f Sample: %s",
                    prefix, epoch, args.epochs, i + 1, num_arch, top1, top5, flops, params, arch)
    logger.info("[%s] Eval Rank Epoch [%d/%d] Average = %.6f Evaluation Result: [%s]",
                prefix, epoch, args.epochs, avg_result, ", ".join("%.6f" % a for a in top1_))
    del model_checkpoint
    torch.cuda.empty_cache()
    return avg_result


def eval_one_arch(model, loader, criterion):
    model.eval()
    loss_tot = correct = correct5 = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)
            loss = criterion(logits, targets)
            top1, top5 = accuracy(logits, targets, topk=(1, 5))
            correct += top1 * inputs.size(0)
            correct5 += top5 * inputs.size(0)
            loss_tot += loss.item() * inputs.size(0)
            total += inputs.size(0)
            logger.debug("Eval One Arch: correct %.0f total %.0f loss %.6f", correct, total, loss_tot)
    return loss_tot / total, correct / total, correct5 / total


def main():
    args = ProxylessParser.parse_configs()

    train_split, valid_split = "augment", "test"
    file_list_dir = "data/proxyless/imagenet"

    train_loader = imagenet_loader_gpu(args, file_list_dir, train_split)
    valid_loader = imagenet_loader_gpu(args, file_list_dir, valid_split, distributed=False)
    sanitize_loader = imagenet_loader_gpu(args, file_list_dir, train_split,
                                          batch_size=args.bn_sanitize_batch_size, infinite=True, distributed=False)
    writer = SummaryWriter(args.tb_dir)

    model = ProxylessNASSuperNet(args)
    mutator = ProxylessMutator(model, validation_size=args.num_valid_arch, seed=args.seed, pruned=args.pruned)
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    model.cuda()

    if args.distributed:
        if args.parallel_sampling:
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
        model = DistributedDataParallel(model, delay_allreduce=True)
        logger.info("Converted to DDP.")
        torch.distributed.barrier()

    if args.label_smoothing > 0:
        criterion = LabelSmoothingLoss(args.num_labels, smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), args.initial_lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)

    last_epoch = 0

    if args.distributed:
        torch.distributed.barrier()
    auto_resume = os.path.join(args.output_dir, "checkpoints", "latest.pth.tar")
    if os.path.exists(auto_resume):
        auto_resume_ckpt = load_checkpoint(model, auto_resume, args=args, optimizer=optimizer)
        if auto_resume_ckpt is not None:
            last_epoch = auto_resume_ckpt["epoch"]
            logger.info("Resume from checkpoint. Proceeding from epoch %d...", last_epoch)

    if args.distributed and args.parallel_sampling:
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

    for epoch in range(last_epoch + 1, args.epochs + 1):
        train(model, mutator, train_loader, criterion, optimizer, writer, args, epoch)
        if (args.eval_every and epoch % args.eval_every == 0) or epoch == args.epochs:
            latest = validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, epoch)
            if args.is_worker_main:
                save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", f"{epoch:06d}.pth.tar"))
        if args.is_worker_main and epoch % 5 == 0:
            save_checkpoint(args, model, auto_resume, optimizer=optimizer.state_dict(), epoch=epoch)
    if args.is_worker_main:
        save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", "final.pth.tar"))


if __name__ == "__main__":
    main()
