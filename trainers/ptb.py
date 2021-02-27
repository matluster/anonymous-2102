import copy
import json
import logging
import math
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import nni
from configs import PtbParser
from datasets.ptb import Corpus
from nni.nas.pytorch.utils import AverageMeterGroup
from space.ptb import PtbMutator, RNNModel

from .utils import (CyclicIterator, Timer, UniversalEncoder, accuracy, load_checkpoint,
                    save_checkpoint, set_running_statistics, write_tensorboard)

logger = logging.getLogger(__name__)


class _NanError(Exception):
    pass


class PtbCriterion(nn.CrossEntropyLoss):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.metrics = dict()

    def _cross_entropy(self, input, target):
        input = input.view(-1, input.size(-1))
        target = target.view(-1)
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

    def forward(self, input, target):
        if isinstance(input, tuple) and len(input) == 4:
            logits, _, rnn_h, dropped_rnn_h = input
            loss = self._cross_entropy(logits, target)
            raw_loss = loss.item()
            if self.alpha > 0:
                loss = loss + self.alpha * dropped_rnn_h.pow(2).mean()  # Activiation Regularization
            if self.beta > 0:
                loss = loss + self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()  # Temporal Activation Regularization (slowness)
            regularization_loss = loss.item() - raw_loss
            self.metrics = {
                "loss": raw_loss,
                "reg": regularization_loss,
                "ppl": math.exp(raw_loss),
            }
            return loss
        elif isinstance(input, tuple) and len(input) == 2:
            logits = input[0]
            return self._cross_entropy(logits, target)
        else:
            return self._cross_entropy(input, target)


def _adjust_learning_rate(args, seq_len, optimizer):
    lr = args.initial_lr * seq_len / args.bptt
    optimizer.param_groups[0]["lr"] = lr
    return lr


@contextmanager
def _use_ax_for_validation(model, optimizer, restore=True):
    if isinstance(optimizer, optim.ASGD):
        logger.info("Loading weights from optimizer states...")
        model_before_val = copy.deepcopy(model.state_dict())
        for p in model.parameters():
            if "ax" in optimizer.state[p]:
                p.data.copy_(optimizer.state[p]["ax"])
        yield
        if restore:
            logger.info("Restore previous weights...")
            model.load_state_dict(model_before_val)
            del model_before_val
        else:
            logger.info("Skipping restore...")
    else:
        # do nothing
        yield


def _dump_state_dict(model, optimizer, epoch, configs):
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "asgd": isinstance(optimizer, optim.ASGD),
        "epoch": epoch,
        "configs": configs,
    }


def _average(arr, bound=1e6):
    ss = count = 0
    for x in arr:
        if x < bound:
            ss += x
            count += 1
    if count == 0:
        return float("nan")
    return ss / count


def train(model, mutator, loader, criterion, optimizer, writer, args, epoch):
    model.train()
    mutator.train()
    meters = AverageMeterGroup()
    model_params = tuple(model.parameters())

    global_step = len(loader) * (epoch - 1)
    hidden = model.generate_hidden(args.batch_size)
    for step, (inputs, targets) in enumerate(loader):
        seq_len = inputs.size(0)
        learning_rate = _adjust_learning_rate(args, seq_len, optimizer)
        optimizer.zero_grad()
        mutator.reset()
        logits = model(inputs, hidden)
        hidden = logits[1].detach()  # replace original logits
        loss = criterion(logits, targets)
        loss.backward()
        loss = loss.item()
        metrics = {"lr": learning_rate, "seqlen": seq_len, **criterion.metrics}
        meters.update(metrics)
        write_tensorboard(writer, "train", metrics, step=global_step)
        global_step += 1
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model_params, args.grad_clip)
        optimizer.step()
        if np.isnan(loss):
            raise _NanError

        if step % args.log_frequency == 0 or step + 1 == len(loader):
            logger.info("Epoch [%d/%d] Step [%d/%d]  %s", epoch, args.epochs, step + 1, len(loader), meters)


def validate(model, mutator, train_loader, valid_loader, criterion, writer, args, epoch):
    model_checkpoint = copy.deepcopy(model.state_dict())
    mutator.eval()
    num_arch = min(args.num_arch_intermediate, len(mutator))
    loss_, ppl_ = np.zeros(num_arch), np.zeros(num_arch)
    for i, arch in zip(range(num_arch), mutator.iterative_reset()):
        with Timer() as timer_finetune:
            if args.state_bn:
                set_running_statistics(args, model, train_loader, seq2seq=True)
        with Timer() as timer_eval:
            loss, ppl = eval_one_arch(model, valid_loader, criterion, args)
        ppl_[i] = ppl
        loss_[i] = loss
        with Timer() as timer_restore:
            model.load_state_dict(copy.deepcopy(model_checkpoint))
        if epoch == args.epochs:
            logger.info("Eval Rank Epoch [%d/%d] Arch [%d/%d] Sample: %s", epoch, args.epochs, i + 1, num_arch,
                        json.dumps(arch, cls=UniversalEncoder))
        logger.info("Eval Rank Epoch [%d/%d] Arch [%d/%d] Time (restore %.3fs sanitize %.3fs eval %.3fs) ppl = %.6f",
                    epoch, args.epochs, i + 1, num_arch, timer_restore.elapsed_secs,
                    timer_finetune.elapsed_secs, timer_eval.elapsed_secs, ppl)
    if writer is not None:
        writer.add_histogram("eval/loss_supernet", loss_, global_step=epoch)
        writer.add_histogram("eval/ppl_supernet", ppl_, global_step=epoch)
    average_loss = _average(loss_)
    average_ppl = _average(ppl_)
    logger.info("Eval Rank Epoch [%d/%d] Average ppl = %.6f loss = %.6f Evaluation Result: [%s]",
                epoch, args.epochs, average_ppl, average_loss, ", ".join("%.6f" % a for a in ppl_))
    writer.add_scalar("eval/average_ppl", average_ppl, global_step=epoch)
    del model_checkpoint
    return average_ppl


def eval_one_arch(model, loader, criterion, args):
    model.eval()
    total_loss = total_count = 0
    hidden = model.generate_hidden(loader.batch_size)
    with torch.no_grad():
        for i, (data, targets) in enumerate(loader):
            logits, hidden = model(data, hidden)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
            total_loss += loss * data.size(0)
            total_count += data.size(0)
            if i % args.log_frequency == 0:
                logger.debug("Eval Single [%d/%d]  total loss = %.6f, total samples = %d",
                             i + 1, len(loader), total_loss, total_count)
    if np.isnan(total_loss):
        logger.warning("NaN encountered when evaluating...")
        return 1e18, 1e18
    else:
        return total_loss / total_count, math.exp(total_loss / total_count)


def main():
    args = PtbParser.parse_configs()

    corpus = Corpus("data/ptb")
    train_loader = corpus.data_loader("train", args.batch_size, (args.bptt, 5, 5, args.bptt + args.max_seq_len_delta))
    sanitize_loader = CyclicIterator(
        corpus.data_loader("train", args.batch_size, (args.bptt, 5, 5, args.bptt + args.max_seq_len_delta))
    )
    valid_loader = corpus.data_loader("valid", args.eval_batch_size, args.bptt)
    sample_loader = corpus.data_loader("valid", args.batch_size, args.bptt)
    test_loader = corpus.data_loader("test", 1, args.bptt)
    writer = SummaryWriter(args.tb_dir)

    model = RNNModel(args, corpus.n_tokens)
    mutator = PtbMutator(model, validation_size=args.num_valid_arch, seed=args.seed, pruned=args.pruned)
    criterion = nn.CrossEntropyLoss()
    if args.resume_checkpoint:
        load_checkpoint(model, args.resume_checkpoint)
    model.cuda()

    criterion = PtbCriterion(args.alpha, args.beta)
    optimizer = optim.SGD(model.parameters(), args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    asgd_patience = int(1E9) # forever
    if args.optimizer == "asgd":
        asgd_patience = args.asgd_patience
        if asgd_patience < 0:
            optimizer = optim.ASGD(model.parameters(), lr=args.initial_lr, t0=0., lambd=0., weight_decay=args.weight_decay)

    patience_counter = 0
    val_ppl = best_ppl = 1e9
    last_epoch = 0
    best_state_dict = _dump_state_dict(model, optimizer, 0, vars(args))

    auto_resume = os.path.join(args.output_dir, "checkpoints", "latest.pth.tar")
    if os.path.exists(auto_resume):
        auto_resume_ckpt = torch.load(auto_resume, map_location="cuda")
        if "configs" not in auto_resume_ckpt:
            auto_resume_ckpt["configs"] = dict()
        logger.info("Configs of saved checkpoint (%s): %s", auto_resume, json.dumps(auto_resume_ckpt["configs"]))
        if args is not None and vars(args) != auto_resume_ckpt["configs"]:
            logger.warning("Checkpoint not loaded. Not a match.")
        else:
            last_epoch = auto_resume_ckpt["epoch"]
            best_state_dict = auto_resume_ckpt["best"]
            if auto_resume_ckpt["asgd"] and isinstance(optimizer, optim.SGD):
                logger.info("Switching back to ASGD!")
                optimizer = optim.ASGD(model.parameters(), lr=args.initial_lr, t0=0., lambd=0., weight_decay=args.weight_decay)
            if not auto_resume_ckpt["asgd"] and isinstance(optimizer, optim.ASGD):
                logger.info("Switching back to SGD!")
                optimizer = optim.SGD(model.parameters(), args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
            model.load_state_dict(auto_resume_ckpt["model"])
            optimizer.load_state_dict(auto_resume_ckpt["optimizer"])
            logger.info("Resume from checkpoint. Proceeding from epoch %d...", last_epoch)

    for epoch in range(last_epoch + 1, args.epochs + 1):
        try:
            train(model, mutator, train_loader, criterion, optimizer, writer, args, epoch)
        except (_NanError, OverflowError):
            if not best_state_dict:
                raise
            logger.warning("NaN encountered. Rolling back to previous best model...")
            if not best_state_dict["asgd"] and isinstance(optimizer, optim.ASGD):
                logger.info("Switching back to SGD!")
                optimizer = optim.SGD(model.parameters(), args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
            model.load_state_dict(best_state_dict["model"])
            optimizer.load_state_dict(best_state_dict["optimizer"])
        if (args.eval_every and epoch % args.eval_every == 0) or epoch == args.epochs:
            with _use_ax_for_validation(model, optimizer):
                val_ppl = validate(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, epoch)
                nni.report_intermediate_result(val_ppl)
                if val_ppl < best_ppl:
                    patience_counter = 0
                    # for rolling back
                    best_state_dict = _dump_state_dict(model, optimizer, epoch, vars(args))
                    best_ppl = val_ppl
                    logger.info("Hit best ppl: %.6f", best_ppl)
                else:
                    patience_counter += 1
                    if patience_counter >= asgd_patience and not isinstance(optimizer, optim.ASGD):
                        logger.info("Switching to ASGD!")
                        optimizer = optim.ASGD(model.parameters(), lr=args.initial_lr, t0=0., lambd=0., weight_decay=args.weight_decay)
                    logger.info("PPL is not better. Patience is %d.", patience_counter)
            if epoch % args.auto_save_every == 0:
                # this is for preemption
                current_state_dict = _dump_state_dict(model, optimizer, epoch, vars(args))
                current_state_dict["best"] = best_state_dict
                torch.save(current_state_dict, auto_resume)

    # restore best
    if best_state_dict:
        model.load_state_dict(best_state_dict["model"])

    if len(mutator) == 1:
        # validate on test loader
        validate(model, mutator, sanitize_loader, test_loader, criterion, writer, args, args.epochs)

    save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", "final.pth.tar"))


if __name__ == "__main__":
    main()
