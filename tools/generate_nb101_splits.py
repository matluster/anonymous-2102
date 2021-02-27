import argparse
import copy
import itertools
import json
import os
import pickle
import random

import h5py
import numpy as np
import tqdm

from space.nb101 import ModelSpec, INPUT, OUTPUT, PRIMITIVES, ID2LABEL, LABEL2ID


def adj_matrix_from_list(adj_list, num_vertices):
    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=np.int8)
    adj_list = np.array(adj_list)
    adj_matrix[adj_list[:, 0], adj_list[:, 1]] = 1
    return adj_matrix


def generate_adj_matrix_with_num_parents(num_parents, ban_i2o=False, ban_loose_ends=False):
    num_vertices = len(num_parents)
    comb_per_node = [itertools.combinations(list(range(i)), p) for i, p in enumerate(num_parents)]
    for adj_list_per_node in itertools.product(*comb_per_node):
        adj_list = [(i, j) for j in range(num_vertices) for i in adj_list_per_node[j]]
        adj_matrix = adj_matrix_from_list(adj_list, num_vertices)
        if ban_i2o and adj_matrix[0, num_vertices - 1]:
            continue
        if ban_loose_ends and not check_adj_matrix_validity(adj_matrix):
            continue
        yield adj_matrix


def generate_1shot1_candidates(num_parents, fix_op=None, fix_connection=None):
    for adj_matrix in generate_adj_matrix_with_num_parents(num_parents, ban_i2o=True, ban_loose_ends=True):
        for op_list in itertools.product(PRIMITIVES, repeat=len(num_parents) - 2):
            op_with_io = [INPUT] + list(op_list) + [OUTPUT]
            yield adj_matrix, op_with_io


def check_adj_matrix_validity(adj_matrix):
    num_vertices = adj_matrix.shape[0]
    model_spec = ModelSpec(adj_matrix, [INPUT] + [PRIMITIVES[0]] * (num_vertices - 2) + [OUTPUT])
    return model_spec.num_vertices == num_vertices


def onehot_repr(index, length):
    assert 0 <= index < length
    return np.array([i == index for i in range(length)], dtype=np.bool)


def main(data_dir):
    """
    Split format:
    [
        {
            "arch": mutable key to multi-hot,
            "hash": string,
            "num_vertices": int,
            "parameters": float (by million),
            "val_acc": 3 float,
            "test_acc": 3 float
        }
    ]
    """
    with h5py.File(os.path.join(data_dir, "nasbench.hdf5"), mode="r") as f:
        hash_vals = f["hash"][()]
        hash2id = {h.decode(): i for i, h in enumerate(hash_vals)}
        num_vertices = f["num_vertices"][()]
        adjacency = f["adjacency"][()]
        operations = f["operations"][()]
        metrics = f["metrics"][()]
        params = f["trainable_parameters"][()]

    # nasbench 1shot1
    nb1shot1c = []
    used_hash = set()
    num_parents = [0, 1, 1, 1, 2, 2, 2]
    for adj, op in tqdm.tqdm(generate_1shot1_candidates(num_parents), desc="Generating NasBench-1shot1-C"):
        arch = dict()
        for i in range(1, len(num_parents)):
            if op[i] != "output":
                arch["op%d" % i] = onehot_repr(LABEL2ID[op[i]], len(PRIMITIVES))
            arch["input%d" % i] = adj[:i, i].astype(np.bool)
        hash_val = ModelSpec(adj, op).hash_spec()
        h5id = hash2id[hash_val]
        used_hash.add(hash_val)
        nb1shot1c.append({
            "arch": arch,
            "hash": hash_val,
            "num_vertices": len(num_parents),
            "parameters": params[h5id].item() / 1e6,
            "val_acc": metrics[h5id, -1, :, -1, 2],
            "test_acc": metrics[h5id, -1, :, -1, 3],
        })
    print("Found %d architectures, %d distinct." % (len(nb1shot1c), len(used_hash)))

    with open(os.path.join(data_dir, "nb1shot1c.pkl"), "wb") as f:
        pickle.dump(nb1shot1c, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/nb101", type=str)
    args = parser.parse_args()
    main(args.data_dir)
