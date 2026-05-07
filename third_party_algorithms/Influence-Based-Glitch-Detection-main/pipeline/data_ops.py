import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import transforms


def subset_selection(dataset, labels, ratio, random_seed):
    ratio = int(len(labels) * ratio)
    (
        sel_ids,
        _,
    ) = train_test_split(
        np.arange(len(labels)),
        train_size=ratio,
        random_state=random_seed,
        stratify=labels,
    )
    return Subset(dataset, sel_ids), labels[sel_ids], sel_ids


class MnistLoader:
    def load_data(self, data_folder_path):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = MNIST(
            root=data_folder_path, download=True, train=True, transform=transform
        )
        testset = MNIST(
            root=data_folder_path, download=True, train=False, transform=transform
        )
        return trainset, testset, trainset.targets, testset.targets


class FmnistLoader:
    def load_data(self, data_folder_path):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = FashionMNIST(
            root=data_folder_path, download=True, train=True, transform=transform
        )
        testset = FashionMNIST(
            root=data_folder_path, download=True, train=False, transform=transform
        )
        return trainset, testset, trainset.targets, testset.targets


class Cifar10Loader:
    def load_data(self, data_folder_path):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = CIFAR10(
            root=data_folder_path, download=True, train=True, transform=transform
        )
        trainset.targets = torch.tensor(trainset.targets)
        testset = CIFAR10(
            root=data_folder_path, download=True, train=False, transform=transform
        )
        testset.targets = torch.tensor(testset.targets)
        return trainset, testset, trainset.targets, testset.targets


class CustomLoader:
    def load_data(self, data_folder_path):
        train_data_fp = os.path.join(data_folder_path, "train.pt")
        test_data_fp = os.path.join(data_folder_path, "test.pt")
        train_data = torch.load(train_data_fp)
        test_data = torch.load(test_data_fp)
        train_labels = train_data.tensors[1]
        test_labels = test_data.tensors[1]
        return train_data, test_data, train_labels, test_labels


dispatcher = {
    "mnist": MnistLoader,
    "fmnist": FmnistLoader,
    "cifar10": Cifar10Loader,
    "custom": CustomLoader,
}
