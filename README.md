# Anonymous Paper Code

## Prepare data

First of all, you need to prepare the data directory in the following format. Directories and files marked with `D` can be easily downloaded because they are public datasets. `P` means that you need to download and prepare them from NAS-Bench. We provided a copy of those files in [Google Drive](https://drive.google.com/file/d/1llOGtlL8cCG4BUG-xVzdUXDAGEvgXnmb/view?usp=sharing).

```
    data
D   ├── cifar10
    │   └── cifar-10-batches-py
    │       ├── batches.meta
    │       ├── data_batch_1
    │       ├── data_batch_2
    │       ├── data_batch_3
    │       ├── data_batch_4
    │       ├── data_batch_5
    │       ├── readme.html
    │       └── test_batch
D   ├── cifar100
    │   ├── cifar-100-python
    │   │   ├── meta
    │   │   ├── test
    │   │   └── train
    │   └── cifar-100-python.tar.gz
D   ├── imagenet16
    │   ├── train_data_batch_1
    │   ├── train_data_batch_10
    │   ├── train_data_batch_2
    │   ├── train_data_batch_3
    │   ├── train_data_batch_4
    │   ├── train_data_batch_5
    │   ├── train_data_batch_6
    │   ├── train_data_batch_7
    │   ├── train_data_batch_8
    │   ├── train_data_batch_9
    │   └── val_data
P   ├── nb101
    │   └── nb1shot1c.pkl
P   ├── nb201
    │   ├── nb201.pkl
    │   ├── split-cifar100.txt
    │   ├── split-cifar10-valid.txt
    │   └── split-imagenet-16-120.txt
    ├── proxyless
P   │   └── imagenet
    │       ├── augment_files.txt
    │       ├── test_files.txt
    │       ├── train_files.txt
    │       └── val_files.txt
D   └── ptb
        ├── test.txt
        ├── train.txt
        └── valid.txt
```

## To train a new supernet

```bash
# nas-bench-101
python -m trainers.nb101 --eval_every 100 --num_valid_arch 300 --num_arch_intermediate 50 --epochs 1000

# nas-bench-201
python -m trainers.nb201 --dataset cifar100 --grad_clip 5 --epochs 1000

# darts-ptb
python -m trainers.ptb --num_arch_intermediate 10 --eval_every 5 --epochs 1000

# darts-cifar
python -m torch.distributed.launch --nproc_per_node=4 --module trainers.darts_cifar --cifar_split 50k --batch_size 112 --epochs 1000 --parallel_sampling true --distributed

# proxyless-nas
python -m torch.distributed.launch --nproc_per_node=8 --module trainers.proxyless --imagenet_split augment --imagenet_dir /path/to/imagenet --num_arch_intermediate 32 --num_valid_arch 192 --epochs 1000 --distributed
```

To train supernet with tricks, use the trainers from `tricks` folder.

## To evaluate on our pretrained checkpoints

We are still seeking for a proper storage service to host them...

## Performance benchmark

Benchmarks are provided in `benchmarks` folder. Please view the code in `space` for interpretation of `architecture`.
