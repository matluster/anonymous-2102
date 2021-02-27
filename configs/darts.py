from .base import BasicParser, str2list, str2dict


class DartsParser(BasicParser):
    def default_params(self):
        return {
            **super().default_params(),

            # model
            "num_labels": 10,
            "init_channels": 36,
            "aux_weight": 0.4,
            "layers": 20,

            # training
            "resume_checkpoint": None,
            "num_threads": 5,
            "batch_size": 128,
            "eval_batch_size": 256,
            "nesterov": False,
            "momentum": 0.9,
            "initial_lr": 0.025,
            "weight_decay": 3e-4,
            "epochs": 600,
            "grad_clip": 5.,
            "parallel_sampling": False,

            # evaluation
            "bn_sanitize_steps": 100,
            "bn_sanitize_batch_size": 200,
            "eval_every": 50,

            # split related
            "exp_id": {"default": None, "type": str},
            "current_trial_id": {"default": None, "type": int},
            "parent_trial_id": {"default": None, "type": int},
            "pruned": {"type": str2dict, "default": {}},
            "epochs_mult": 1.,
            "initial_lr_mult": 1.,
            "num_valid_arch": 256,
            "num_arch_intermediate": 32,
            "num_batch_partition": 10,
            "num_partition_max": 7,
            "min_sample_per_partition": 5,

            # distributed
            "distributed": {"default": False, "action": "store_true"},
            "local_rank": 0,
        }

    def validate_args(self, args):
        if args.num_arch_intermediate is None:
            args.num_arch_intermediate = args.num_valid_arch
        args.num_arch_intermediate = min(args.num_arch_intermediate, args.num_valid_arch)



class DartsCifarParser(DartsParser):
    def default_params(self):
        return {
            **super().default_params(),
            "cifar_split": ["40k", "50k"],
            "cutout": 16,
            "drop_path_prob": 0.2,
        }


class DartsImageNetParser(DartsParser):
    def default_params(self):
        return {
            **super().default_params(),
            "layers": 14,
            "init_channels": 48,
            "num_labels": 1000,

            # training
            "imagenet_dir": "data/imagenet",
            "imagenet_split": ["train", "augment"],
            "batch_size": 128,
            "label_smoothing": 0.1,
            "weight_decay": 3e-5,
            "warmup_epochs": 5,
            "epochs": 250,
            "color_jitter": ["normal", "tf"],
            "enable_gpu_dataloader": False,
        }
