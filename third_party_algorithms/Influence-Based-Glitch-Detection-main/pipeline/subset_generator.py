import argparse
import json
import os.path
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .data_ops import dispatcher as data_dispatcher
from .data_ops import subset_selection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name", required=True, type=str, help="Name of dataset for dispatcher"
    )
    parser.add_argument(
        "--data_folder", required=True, type=str, help="Folder to dataset"
    )
    parser.add_argument(
        "--ratio", required=True, type=float, help="Fraction of samples to keep"
    )
    parser.add_argument(
        "--subset_savedir", required=True, type=str, help="Saving folder"
    )
    parser.add_argument("--seed", required=True, type=int, default=42)
    args = parser.parse_args()
    return args


def convert_labels(labels):
    if type(labels) == torch.Tensor:
        all_train_labels = labels.numpy()
    else:
        all_train_labels = np.array(labels)
    return all_train_labels


def gen_save_subset(dataset, labels, ratio, savedir, fname, seed=42):
    labels = convert_labels(labels)
    subset, labels, _ = subset_selection(dataset, labels, ratio, seed)
    Path(savedir).mkdir(parents=True, exist_ok=True)
    x_tensor = torch.stack([x for x, _ in subset])
    y_tensor = torch.tensor(labels)
    fpath = os.path.join(savedir, fname)
    subset_dataset = TensorDataset(x_tensor, y_tensor)
    torch.save(subset_dataset, fpath)
    with open(os.path.join(savedir, "subset_info.json"), "w") as f:
        json.dump({"subset_ratio": ratio, "seed": seed}, f)
    return subset_dataset


if __name__ == "__main__":
    args = parse_args()
    trainset, testset = data_dispatcher[args.data_name]().load_data(args.data_folder)

    gen_save_subset(
        trainset,
        trainset.targets,
        args.ratio,
        args.subset_savedir,
        "clean_train.pt",
        args.seed,
    )
    gen_save_subset(
        testset,
        testset.targets,
        args.ratio,
        args.subset_savedir,
        "clean_test.pt",
        args.seed,
    )
