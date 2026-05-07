import argparse
import json
import os.path
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config_manager import ConfigManager
from .model_train import train
from .utils import load_model


def str_or_none(value):
    if value.lower() in ["none", "null", "na", "nan"]:
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        required=True,
        type=str,
        help="Path to train TensorDataset (.pt format)",
    )
    parser.add_argument(
        "--test_data_path",
        required=True,
        type=str,
        help="Path to test TensorDataset (.pt format)",
    )
    parser.add_argument(
        "--model_conf", required=True, type=str, help="Model configuration (.json)"
    )
    parser.add_argument(
        "--training_conf",
        required=True,
        type=str,
        help="Configuration with training information (.json)",
    )
    parser.add_argument(
        "--model_savedir",
        required=False,
        type=str_or_none,
        default=None,
        help="Directory to save the model checkpoints, info.json, if none, the "
        "results will be saved in the train data path",
    )
    parser.add_argument(
        "--device",
        required=False,
        type=str_or_none,
        default="cpu",
        help="Device (e.g. cuda or cpu)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cm = ConfigManager(model_conf=args.model_conf, training_conf=args.training_conf)

    savedir = args.model_savedir
    if savedir is None:
        savedir = os.path.join(
            Path(args.train_data_path).parent, cm.model_conf.get_name()
        )
        Path(savedir).mkdir(parents=True, exist_ok=True)

    ckpt_fp = os.path.join(savedir, "ckpts")
    info_file_fp = savedir

    Path(ckpt_fp).mkdir(parents=True, exist_ok=True)
    Path(info_file_fp).mkdir(parents=True, exist_ok=True)

    train_data = torch.load(args.train_data_path)
    test_data = torch.load(args.test_data_path)

    num_classes = len(train_data.tensors[1].unique())
    input_shape = train_data.tensors[0].shape[1:]

    clean_model = load_model(
        model_config=cm.model_conf, input_shape=input_shape, num_classes=num_classes
    )

    if cm.training_conf.get_random_seed() is not None:
        torch.manual_seed(cm.training_conf.get_random_seed())

    trainloader = DataLoader(
        train_data,
        batch_size=cm.training_conf.get_batch_size(),
        shuffle=True,
        num_workers=15,
    )

    testloader = DataLoader(
        test_data,
        batch_size=cm.training_conf.get_batch_size(),
        shuffle=False,
        num_workers=15,
    )

    clean_model, train_loss, test_loss, train_acc, test_acc = train(
        model=clean_model,
        epochs=cm.training_conf.get_epochs(),
        learning_rate=cm.training_conf.get_learning_rate(),
        reg_strength=cm.training_conf.model_regularization_strength(),
        save_dir=ckpt_fp,
        train_loader=trainloader,
        test_loader=testloader,
        device=args.device,
        save_ckpts=True,
    )

    info = {
        f"train_acc": train_acc,
        f"test_acc": test_acc,
        f"train_loss": train_loss,
        f"test_loss": test_loss,
    }

    with open(os.path.join(info_file_fp, "model_info.json"), "w") as f:
        json.dump(info, f)
