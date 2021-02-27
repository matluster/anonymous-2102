import collections
import pickle
import random

import h5py
import numpy as np
import tqdm
from nas_201_api import NASBench201API


def is_valid_arch(matrix):
    n = matrix.shape[0]
    visited = {0}
    q = collections.deque([0])
    while q:
        u = q.popleft()
        for v in range(u + 1, n):
            if v not in visited and matrix[u][v] != 0:
                # select a non-zero op
                visited.add(v)
                q.append(v)
    return (n - 1) in visited


random.seed(0)
api = NASBench201API("/tmp/NAS-Bench-201-v1_1-096897.pth")
results = []
for arch_index in tqdm.tqdm(range(len(api))):
    op_matrix = NASBench201API.str2matrix(api.arch(arch_index)).astype(np.uint8).T
    arch = {f"{i}_{j}": op_matrix[i, j].item() for i in range(op_matrix.shape[0]) for j in range(i + 1, op_matrix.shape[0])}
    result = {"arch": arch}
    if not is_valid_arch(op_matrix):
        continue
    for dataset in ["cifar10-valid", "cifar10", "cifar100", "ImageNet16-120"]:
        compute_data = api.query_by_index(arch_index, dataset)
        arch_index_data = []
        available_seeds = api.arch2infos_full[arch_index].get_dataset_seeds(dataset)
        for k in range(3):
            seed = available_seeds[k] if k < len(available_seeds) else random.choice(available_seeds)
            if dataset == "cifar10-valid":
                metrics_name = ["train-loss", "train-accuracy", "valid-loss", "valid-accuracy", "test-loss", "test-accuracy"]
            elif dataset == "cifar10":
                metrics_name = ["train-loss", "train-accuracy", "test-loss", "test-accuracy", "test-loss", "test-accuracy"]
            else:
                metrics_name = ["train-loss", "train-accuracy", "valid-loss", "valid-accuracy", "test-loss", "test-accuracy"]
            metrics = api.get_more_info(arch_index, dataset, is_random=seed)
            data = [metrics[k] / 100 if "accuracy" in k else metrics[k] for k in metrics_name]
            data = [d[0] if isinstance(d, tuple) else d for d in data]
            data += [compute_data[seed].flop, compute_data[seed].params, compute_data[seed].get_latency()]
            if arch_index == 0 and k == 0:
                print(arch, dataset, metrics, data)
            arch_index_data.append(data)
        register_dataset_name = dataset
        if dataset == "ImageNet16-120":
            register_dataset_name = "imagenet-16-120"
        result[register_dataset_name] = np.array(arch_index_data)
    results.append(result)

print("Found %d valid architectures." % len(results))
with open("data/nb201/nb201.pkl", "wb") as fp:
    pickle.dump(results, fp)
