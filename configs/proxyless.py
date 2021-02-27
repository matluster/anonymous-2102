from .base import BasicParser, str2list, str2dict


class ProxylessParser(BasicParser):
    def default_params(self):
        return {
            **super().default_params(),

            # model
            "width_mult": 1.35,
            "width_stages": {"default": [24, 40, 80, 96, 192, 320], "type": str2list},
            "n_cell_stages": {"default": [4, 4, 4, 4, 4, 1], "type": str2list},
            "stride_stages": {"default": [2, 2, 2, 1, 2, 1], "type": str2list},
            "bn_momentum": 0.1,
            "bn_epsilon": 1e-3,
            "model_init": ["he_fout", "he_fin"],
            "init_div_groups": False,
            "num_labels": 1000,
            "ops_order": ["weight_bn_act"],
            "dropout_rate": 0.,
            "image_size": 224,

            # training
            "resume_checkpoint": None,
            "num_threads": 5,
            "imagenet_dir": "data/imagenet",
            "imagenet_split": ["train", "augment"],
            "batch_size": 128,
            "eval_batch_size": 256,
            "nesterov": False,
            "momentum": 0.9,
            "label_smoothing": 0.1,
            "initial_lr": 0.025,
            "weight_decay": 5e-5,
            "warmup_epochs": 5,
            "epochs": 300,
            "color_jitter": ["normal", "tf"],
            "keep_checkpoint": ["none", "epoch", "end"],
            "enable_gpu_dataloader": False,
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
            "num_valid_arch": 300,
            "num_arch_intermediate": {"type": int, "default": None},
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
