import functools
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import VisionDataset


class CIFAR10(VisionDataset):
    base_folder = "cifar-10-batches-py"
    train_val_test_list = {
        "train": [
            ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
            ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
            ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
            ["data_batch_4", "634d18415352ddfa80567beed471001a"]
        ],
        "val": [
            ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
        ],
        "augment": [
            ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
            ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
            ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
            ["data_batch_4", "634d18415352ddfa80567beed471001a"],
            ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
        ],
        "test": [
            ["test_batch", "40351d587109b95175f43aff81a1287e"],
        ]
    }

    def __init__(self, root, split="train", transform=None, target_transform=None):
        super(CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        downloaded_list = self.train_val_test_list[split]
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.data = np.ascontiguousarray(self.data)  # increase locality

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format(self.split.capitalize())


def cutout_fn(img, length):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask

    return img


class DistributableDataLoader(DataLoader):
    def __init__(self, dataset, distributed=False, **kwargs):
        self.distributed_sampler = None
        if distributed:
            self.distributed_sampler = kwargs["sampler"] = DistributedSampler(dataset)
            kwargs.pop("shuffle", None)
        super().__init__(dataset, **kwargs)
        self._epoch = 0

    def __iter__(self):
        self._epoch += 1
        if self.distributed_sampler is not None:
            self.distributed_sampler.set_epoch(self._epoch)
        return super().__iter__()


def dataloader_cifar(image_dir, split, args, batch_size=None, shuffle=None, distributed=False):  # pylint: disable=inconsistent-return-statements
    if batch_size is None:
        batch_size = args.batch_size
    if shuffle is None:
        shuffle = split in ("train", "augment")
    assert torch.cuda.is_available()
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    if split in ("train", "augment"):
        cutout = []
        if args.cutout > 0:
            cutout.append(functools.partial(cutout_fn, length=args.cutout))
        dataset = CIFAR10(image_dir, split, transform=transforms.Compose(transf + normalize + cutout))
        return DistributableDataLoader(dataset, distributed=distributed, batch_size=batch_size, drop_last=True, shuffle=shuffle, num_workers=args.num_threads)
    if split in ("val", "test"):
        dataset = CIFAR10(image_dir, split, transform=transforms.Compose(normalize))
        return DistributableDataLoader(dataset, distributed=distributed, batch_size=batch_size, drop_last=False, shuffle=shuffle, num_workers=args.num_threads)
