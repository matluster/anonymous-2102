import copy
import functools
import itertools
import logging
from collections import defaultdict

import numpy as np
import torch
from nni.nas.pytorch.mutator import Mutator

from .network import LayerWithInputChoice, PRIMITIVES


logger = logging.getLogger(__name__)


class DartsMutator(Mutator):
    def __init__(self, model, validation_size, pruned=None, seed=-1):
        super().__init__(model)
        self.pruned = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerWithInputChoice):
                self.pruned[f"{mutable.key}/op"] = list(PRIMITIVES)
                self.pruned[f"{mutable.key}/input"] = list(range(mutable.n_inputs))
        self._generate_combinations()
        _original_search_space_size = self.search_space_size(self._combinations_pruned)
        if pruned is not None:
            for k, v in pruned.items():
                if isinstance(v, (str, int)):
                    v = [v]
                self.pruned[k] = v
        self._generate_combinations()
        self.total_size = self.search_space_size(self._combinations_pruned)
        
        logger.info("Pruned from %d (%.2e) architectures to %d (%.2e).",
                    _original_search_space_size, float(_original_search_space_size),
                    self.total_size, float(self.total_size))
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.validation_size = validation_size
        self.archset = None
        self.val_archset = self.select_validation_archset()

    def select_validation_archset(self):
        ss_size = self.search_space_size(self._combinations_pruned)
        if ss_size > self.validation_size * 10:
            return [self._random_sample() for _ in range(self.validation_size)]
        logger.warning("Attempting to choose %d architectures from %d.", self.validation_size, ss_size)
        all_candidates = list(self._all_permutations())
        assert len(all_candidates) == ss_size
        return [all_candidates[i] for i in self.random_state.permutation(ss_size)[:self.validation_size]]

    def reset(self, arch_index=None):
        if arch_index is not None:
            assert not self.training, "Training mode doesn't support index reset."
            arch = self.val_archset[arch_index]
        else:
            if self.training:
                arch = self._random_sample()
            else:
                arch = self.val_archset[self.random_state.randint(len(self.val_archset))]
        self._cache = copy.deepcopy(arch)
        return {"arch": arch}

    def iterative_reset(self):
        assert not self.training, "Training mode doesn't support iterative reset."
        for i in range(len(self.val_archset)):
            yield self.reset(i)

    def on_forward_layer_with_input(self, mutable, inputs):
        assert isinstance(inputs, (list, tuple))
        op_idx = self._cache[f"{mutable.key}/op"]
        input_idx = self._cache[f"{mutable.key}/input"]
        return mutable.choices[op_idx](inputs[input_idx], mutable.strides[input_idx])

    def _random_sample(self):
        result = dict()
        for k, lst in self._combinations_pruned.items():
            op1, inp1, op2, inp2 = lst[self.random_state.randint(len(lst))]
            result.update(self._verbose_arch_node(k, op1, inp1, op2, inp2))
        return result

    def _verbose_arch_node(self, k, op1, inp1, op2, inp2):
        return {
            f"{k}/0/op": op1,
            f"{k}/0/input": inp1,
            f"{k}/1/op": op2,
            f"{k}/1/input": inp2
        }

    def _all_permutations(self):
        keys, vals = zip(*self._combinations_pruned.items())
        for val in itertools.product(*vals):
            result = dict()
            for k, (op1, inp1, op2, inp2) in zip(keys, val):
                result.update(self._verbose_arch_node(k, op1, inp1, op2, inp2))
            yield result

    def _generate_combinations(self):
        nodes = sorted(set([k.rsplit("/", 2)[0] for k in self.pruned]))
        self._combinations_pruned = dict()
        for node in nodes:
            self._combinations_pruned[node] = []
            for op1, inp1, op2, inp2 in itertools.product(self.pruned[f"{node}/0/op"],
                                                          self.pruned[f"{node}/0/input"],
                                                          self.pruned[f"{node}/1/op"],
                                                          self.pruned[f"{node}/1/input"]):
                if inp1 != inp2:
                    self._combinations_pruned[node].append((op1, inp1, op2, inp2))
            logger.info("Node %s has %d combinations.", node, len(self._combinations_pruned[node]))

    def search_space_size(self, space_dict: dict):
        return functools.reduce(lambda a, b: a * len(b), space_dict.values(), 1)

    def __len__(self):
        return self.total_size if self.training else len(self.val_archset)
