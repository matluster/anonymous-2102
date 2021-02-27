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


class Nb101Mutator(Mutator):
    def __init__(self, model, validation_size, archset="data/nb101/nb1shot1c.pkl", pruned=None, seed=-1):
        super().__init__(model)
        if isinstance(archset, str):
            with open(archset, "rb") as f:
                self.archset = pickle.load(f)
        else:
            self.archset = archset
        if pruned is None:
            pruned = dict()
        self.seed = seed
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
        # recover self.pruned
        for sample in self.trn_archset:
            for k, v in sample["arch"].items():
                if k not in self.pruned:
                    self.pruned[k] = set()
                self.pruned[k].add(tuple(v.astype(np.long).tolist()))
        self.pruned = {k: sorted(v) for k, v in self.pruned.items()}
        logger.info("Pruned architecture space: %s", json.dumps(self.pruned))
        logger.info("Pruned from %d architectures to %d.", original_length, len(self.trn_archset))

    def print_archset_summary(self, name, archset):
        logger.info("%s archset summary: %d architectures, mean val acc = %.6f, "
                    "mean test acc = %.6f, mean parameters = %.6f",
                    name, len(archset), np.mean([d["val_acc"] for d in archset]),
                    np.mean([d["test_acc"] for d in archset]), np.mean([d["parameters"] for d in archset]))

    def reset(self, arch_index=None):
        archset = self._get_current_archset()
        if arch_index is None:
            arch_index = self.random_state.randint(len(archset))
        selected = copy.deepcopy(archset[arch_index])
        self._cache = dict()
        for mutable in self.mutables:
            if isinstance(mutable, (LayerChoice, InputChoice)):
                k = mutable.key
                self._cache[k] = torch.tensor(selected["arch"][k], dtype=torch.bool)
        selected["val_acc"] = np.mean(selected["val_acc"])
        selected["test_acc"] = np.mean(selected["test_acc"])
        return selected

    def iterative_reset(self):
        archset = self._get_current_archset()
        for i in range(len(archset)):
            yield self.reset(i)

    def _get_current_archset(self):
        return self.trn_archset if self.training else self.val_archset

    def reset_val_archset(self, population=None):
        """
        Reset validation archset with an existing archset
        """
        if isinstance(population, list):
            self.val_archset = []
            for indv in population:
                arch_idx = self._archset_inverse.get(self._build_arch_key({"arch": indv}), None)
                if arch_idx is None:
                    # arch is not legal, compute some fake data
                    self.val_archset.append({"arch": indv, "parameters": float("nan"), "val_acc": float("nan"), "test_acc": float("nan")})
                else:
                    self.val_archset.append(self.trn_archset[arch_idx])
        elif isinstance(population, int):
            random_state = np.random.RandomState(self.seed)
            self.val_archset = [self.trn_archset[random_state.randint(len(self.trn_archset))] for _ in range(population)]
        else:
            raise ValueError("Population not recognized, architecture set not updated.")

    def search_space_size(self, pruned):
        return len([True for a in self.archset if self._validate_arch(a, pruned)])

    def _validate_arch(self, sample, pruned):
        for k, allowed_values in pruned.items():
            if not any(np.equal(sample["arch"][k], val).all() for val in allowed_values):
                return False
        return True

    @staticmethod
    def _build_arch_key(sample):
        return json.dumps({k: v.astype(np.long).tolist() if isinstance(v, np.ndarray) else v for k, v in sample["arch"].items()}, sort_keys=True)

    def __len__(self):
        return len(self._get_current_archset())
