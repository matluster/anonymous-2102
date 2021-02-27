from .base import BasicParser, str2dict


class PtbParser(BasicParser):
    def default_params(self):
        return {
            **super().default_params(),
            "log_frequency": 50,

            # model
            "n_hidden": 850,
            "dropout": 0.75,
            "dropout_rnn_h": 0.25,
            "dropout_rnn_x": 0.75,
            "dropout_inp": 0.2,
            "dropout_emb": 0.1,
            "init_range": 0.04,
            "steps": 8,

            # training
            "resume_checkpoint": None,
            "bptt": 35,
            "batch_size": 256,
            "eval_batch_size": 10,
            "max_seq_len_delta": 20,
            "momentum": 0.,
            "initial_lr": 40.,
            "weight_decay": 8e-7,
            "alpha": 0.,
            "beta": 1e-3,
            "epochs": 3000,
            "grad_clip": 0.1,
            "state_bn": False,
            "optimizer": ["asgd", "sgd"],
            "asgd_patience": 5,

            # evaluation
            "bn_sanitize_steps": 50,
            "bn_sanitize_batch_size": 64,
            "eval_every": 5,
            "auto_save_every": 20,

            # split related
            "exp_id": {"default": None, "type": str},
            "current_trial_id": {"default": None, "type": int},
            "parent_trial_id": {"default": None, "type": int},
            "pruned": {"type": str2dict, "default": {}},
            "epochs_mult": 1.,
            "initial_lr_mult": 1.,
            "num_valid_arch": 10,
            "num_arch_intermediate": 10,
            "num_batch_partition": 10,
            "num_partition_max": 4,
            "min_sample_per_partition": 5,
        }

    def validate_args(self, args):
        if args.num_arch_intermediate is None:
            args.num_arch_intermediate = args.num_valid_arch
        assert args.auto_save_every % args.eval_every == 0
        args.num_arch_intermediate = min(args.num_arch_intermediate, args.num_valid_arch)
