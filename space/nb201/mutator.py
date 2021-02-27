import copy
import itertools
import json
import logging
import pickle

import numpy as np
import torch
from nni.nas.pytorch.mutables import LayerChoice, InputChoice, MutableScope
from nni.nas.pytorch.mutator import Mutator


logger = logging.getLogger(__name__)


class Nb201Mutator(Mutator):
    def __init__(self, model, dataset, validation_size, archset="data/nb201/nb201.pkl", pruned=None, seed=0):
        super().__init__(model)
        if isinstance(archset, str):
            with open(archset, "rb") as f:
                self.archset = pickle.load(f)
        else:
            self.archset = archset
        if pruned is None:
            pruned = dict()
        self.seed = seed
        self.dataset = dataset
        self.random_state = np.random.RandomState(seed)
        self.validation_size = validation_size
        self.prune_archset(pruned)
        self._archset_inverse = {self._build_arch_key(sample): i for i, sample in enumerate(self.trn_archset)}
        self.val_archset = self.select_validation_archset()
        self.print_archset_summary("Train", self.trn_archset)
        self.print_archset_summary("Validation", self.val_archset)

    def select_validation_archset(self):
        if len(self.trn_archset) < self.validation_size:
            logger.warning("Attempting to choose %d architectures from length %d.", self.validation_size, len(self.trn_archset))
            index_list = list(range(len(self.trn_archset)))
        else:
            random_state = np.random.RandomState(self.seed)
            index_list = sorted(random_state.permutation(len(self.trn_archset))[:self.validation_size].tolist())
        return [self.trn_archset[i] for i in index_list]

    def prune_archset(self, pruned):
        original_length = len(self.archset)
        if isinstance(pruned, int):
            self.trn_archset = [self.archset[pruned]]
        else:
            self.trn_archset = [a for a in self.archset if self._validate_arch(a, pruned)]
        self.pruned = dict()
        for sample in self.trn_archset:
            for k, v in sample["arch"].items():
                if k not in self.pruned:
                    self.pruned[k] = set()
                self.pruned[k].add(v)
        self.pruned = {k: sorted(v) for k, v in self.pruned.items()}
        logger.info("Pruned architecture space: %s", json.dumps(self.pruned))
        logger.info("Pruned from %d architectures to %d.", original_length, len(self.trn_archset))

    def print_archset_summary(self, name, archset):
        logger.info("%s archset summary: %d architectures, mean val acc = %.6f, "
                    "mean test acc = %.6f, mean parameters = %.6f",
                    name, len(archset), np.mean([d[self.dataset][:, 3] for d in archset]),
                    np.mean([d[self.dataset][:, 3] for d in archset]), np.mean([d[self.dataset][:, 8] for d in archset]))

    def reset(self, arch_index=None):
        archset = self._get_current_archset()
        if arch_index is None:
            arch_index = self.random_state.randint(len(archset))
        selected = copy.deepcopy(archset[arch_index])
        self._cache = dict()
        for mutable in self.mutables:
            if isinstance(mutable, (LayerChoice, InputChoice)):
                k = mutable.key
                self._cache[k] = selected["arch"][k]
        return {
            "arch": selected["arch"],
            "val_acc": np.mean(selected[self.dataset][:, 3]),
            "test_acc": np.mean(selected[self.dataset][:, 5]),
            "flops": np.mean(selected[self.dataset][:, 6]),
            "params": np.mean(selected[self.dataset][:, 7]),
            "latency": np.mean(selected[self.dataset][:, 8]),
        }

    def on_forward_layer_choice(self, mutable, *inputs):
        return mutable.choices[self._cache[mutable.key]](*inputs), None  # fake mask

    def iterative_reset(self):
        archset = self._get_current_archset()
        for i in range(len(archset)):
            yield self.reset(i)

    def _get_current_archset(self):
        return self.trn_archset if self.training else self.val_archset

    def search_space_size(self, pruned):
        return len([True for a in self.archset if self._validate_arch(a, pruned)])

    def _validate_arch(self, sample, pruned):
        for k, allowed_values in pruned.items():
            if sample["arch"][k] not in allowed_values:
                return False
        return True

    @staticmethod
    def _build_arch_key(sample):
        return json.dumps(sample["arch"], sort_keys=True)

    def __len__(self):
        return len(self._get_current_archset())
