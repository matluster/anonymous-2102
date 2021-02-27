from .base import str2list, str2dict, BasicParser


class Nb101Parser(BasicParser):
    def default_params(self):
        return {
            **super().default_params(),

            # training
            "resume_checkpoint": None,
            "weight_decay": 1e-4,
            "initial_lr": 0.1,
            "ending_lr": 0.,
            "grad_clip": 5.,
            "batch_size": 256,
            "momentum": 0.9,
            "epochs": 108,
            "cifar_split": ["40k", "50k"],

            # nb101
            "archset": "nb1shot1c",  # not used yet
            "num_stacks": 3,
            "num_modules_per_stack": 3,
            "num_parents": {"default": [0, 1, 1, 1, 2, 2, 2], "type": str2list},
            "stem_out_channels": 128,
            "bn_affine": True,
            "bn_momentum": 3e-3,
            "bn_epsilon": 1e-5,
            "num_labels": 10,

            # tricks
            "dropout_rate": 0.,
            "drop_path_prob": 0.,
            "aux_weight": 0.,
            "cutout": 0,

            # evaluation
            "bn_sanitize_steps": 25,
            "bn_sanitize_batch_size": 400,
            "eval_every": 12,

            # split related
            "exp_id": {"default": None, "type": str},
            "current_trial_id": {"default": None, "type": int},
            "parent_trial_id": {"default": None, "type": int},
            "pruned": {"type": str2dict, "default": {}},
            "epochs_mult": 1.,
            "initial_lr_mult": 1.,
            "drop_path_prob_delta": 0.,
            "num_valid_arch": 300,
            "num_arch_intermediate": {"type": int, "default": None},
            "num_batch_partition": 10,
            "num_partition_max": 4,
            "min_sample_per_partition": 5,

            # multi-objective optimization
            "pareto_front": {"default": False, "action": "store_true"},
            "num_generations": 15,
            "population_size": 48,
        }

    def validate_args(self, args):
        if args.num_arch_intermediate is None:
            args.num_arch_intermediate = args.num_valid_arch
        args.num_arch_intermediate = min(args.num_arch_intermediate, args.num_valid_arch)
