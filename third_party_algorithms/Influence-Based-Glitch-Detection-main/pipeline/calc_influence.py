import argparse
import os
from pathlib import Path

import numpy as np
import torch

from influence_functions.tracin_torch import TracInInfluenceTorch
from .config_manager import ConfigManager
from .utils import load_model


def int_or_none(value):
    if str(value).lower() in ["none", "null", "na", "nan"]:
        return None
    return int(value)


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
        help="Path to train dataset (TensorDataset)",
    )
    parser.add_argument(
        "--test_data_path",
        required=True,
        type=str,
        help="Path to test dataset (TensorDataset)",
    )
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--model_conf", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--inf_mat_savedir", required=False, type=str_or_none)
    return parser.parse_args()


def compute_inf_mat(
    tracin_cp,
    trainset,
    testset,
    unfrozen_layers,
    column_wise=False,
    e_id=None,
    batch_size=4096,
    fast_cp=False,
):
    dataset_for_si = trainset if not column_wise else testset

    self_if_arr = tracin_cp.compute_self_influence(
        dataset=dataset_for_si,
        load_pretrained_model=True,
        batch_size=batch_size,
        layers=unfrozen_layers,
        epoch_ids_to_consider=None if e_id is None else [e_id],
        fast_cp=fast_cp,
    )

    train_test_im = tracin_cp.compute_train_to_test_influence(
        train_set=trainset,
        test_set=testset,
        load_pretrained_model=True,
        batch_size=batch_size,
        layers=unfrozen_layers,
        epoch_ids_to_consider=None if e_id is None else [e_id],
        fast_cp=fast_cp,
    )

    return self_if_arr, train_test_im


if __name__ == "__main__":
    args = parse_args()

    cm = ConfigManager(model_conf=args.model_conf)

    train_data = torch.load(args.train_data_path)
    test_data = torch.load(args.test_data_path)

    num_classes = len(train_data.tensors[1].unique())
    input_shape = train_data.tensors[0].shape[1:]

    clean_model = load_model(
        model_config=cm.model_conf, input_shape=input_shape, num_classes=num_classes
    )

    savedir = args.inf_mat_savedir
    if savedir is None:
        savedir = Path(args.ckpt).parent
    else:
        Path(savedir).mkdir(parents=True, exist_ok=True)

    tracin_cp = TracInInfluenceTorch(
        model_instance=clean_model,
        save_models_path=args.ckpt,
        random_state=args.seed,
        input_shape=None,
        learning_rate=None,
        batch_size=None,
        epochs=None,
        reg_strength=None,
    )

    inf_mat = compute_inf_mat(
        tracin_cp,
        train_data,
        test_data,
        cm.training_conf.__trainable_layers(),
        e_id=None,
    )

    with open(os.path.join(savedir, "inf_mat.npy"), "wb") as f:
        np.save(f, inf_mat)
