from .base import str2list, str2dict, BasicParser


class Nb201Parser(BasicParser):
    def default_params(self):
        return {
            **super().default_params(),
            "num_modules_per_stack": 5,
            "stem_out_channels": 16,
            "batch_size": 256,
            "epochs": 200,
            "initial_lr": 0.1,
            "ending_lr": 0.,
            "nesterov": True,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "bn_momentum": 0.1,
            "bn_affine": True,
            "bn_track_running_stats": True,
            "grad_clip": 0.,
            "dataset": ["cifar100", "imagenet-16-120", "cifar10-valid", "cifar10"],
            "resume_checkpoint": None,

            # evaluation
            "bn_sanitize_steps": 25,
            "bn_sanitize_batch_size": 400,
            "eval_every": 100,
            "num_valid_arch": 100,
            "num_arch_intermediate": {"type": int, "default": None},
            "pruned": {"type": str2dict, "default": {}},
        }

    def validate_args(self, args):
        if args.num_arch_intermediate is None:
            args.num_arch_intermediate = args.num_valid_arch
        args.num_arch_intermediate = min(args.num_arch_intermediate, args.num_valid_arch)
