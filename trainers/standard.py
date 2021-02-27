import logging
import os

from .utils import load_checkpoint, save_checkpoint


logger = logging.getLogger(__name__)


def train_loop(model, mutator, criterion, optimizer, scheduler,
               train_loader, sanitize_loader, valid_loader,
               train_fn, valid_fn, writer, args):
    last_epoch = 0
    auto_resume = os.path.join(args.output_dir, "checkpoints", "latest.pth.tar")
    if os.path.exists(auto_resume):
        auto_resume_ckpt = load_checkpoint(model, auto_resume, args=args, optimizer=optimizer, scheduler=scheduler)
        if auto_resume_ckpt is not None:
            last_epoch = auto_resume_ckpt["epoch"]
            logger.info("Resume from checkpoint. Proceeding from epoch %d...", last_epoch)

    for epoch in range(last_epoch + 1, args.epochs + 1):
        train_fn(model, mutator, train_loader, criterion, optimizer, scheduler, writer, args, epoch)
        if (args.eval_every and epoch % args.eval_every == 0) or epoch == args.epochs:
            valid_fn(model, mutator, sanitize_loader, valid_loader, criterion, writer, args, epoch)
        if epoch % 20 == 0:
            save_checkpoint(args, model, auto_resume, optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), epoch=epoch)

    save_checkpoint(args, model, os.path.join(args.output_dir, "checkpoints", "final.pth.tar"))
