import random
import os
import sys

random.seed(0)
imagenet_dir = sys.argv[1]
train_dir = os.path.join(imagenet_dir, "train")
val_dir = os.path.join(imagenet_dir, "val")
label_index = {}
train_files, valid_files, augment_files, test_files = [], [], [], []
for i, d in enumerate(sorted(os.listdir(train_dir))):
    label_index[d] = i
    file_num = len(list(os.listdir(os.path.join(train_dir, d))))
    print(i, d, file_num)
    bits = [True] * 50 + [False] * (file_num - 50)
    random.shuffle(bits)
    for b, f in zip(bits, sorted(os.listdir(os.path.join(train_dir, d)))):
        pr = (os.path.join("train", d, f), i)
        augment_files.append(pr)
        if b:
            valid_files.append(pr)
        else:
            train_files.append(pr)
for d in sorted(os.listdir(val_dir)):
    for f in sorted(os.listdir(os.path.join(val_dir, d))):
        test_files.append((os.path.join("val", d, f), label_index[d]))

def export_list(lst, target):
    with open(target, "w") as f:
        for a, b in lst:
            print(a, b, file=f)
export_list(train_files, "data/proxyless/imagenet/train_files.txt")
export_list(valid_files, "data/proxyless/imagenet/val_files.txt")
export_list(augment_files, "data/proxyless/imagenet/augment_files.txt")
export_list(test_files, "data/proxyless/imagenet/test_files.txt")
