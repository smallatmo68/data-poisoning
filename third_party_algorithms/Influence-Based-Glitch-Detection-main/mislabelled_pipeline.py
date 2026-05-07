import os.path
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset

from influence_functions import TracInInfluenceTorch
from utils import train_test_loaders, run_model, calc_f1, save_as_np, save_as_json

import torch
from signals_fast import InfluenceErrorSignals
from error_injection import mislabelled_uniform, mislabelled_cb
from pipeline.config_manager import ConfigManager
from pipeline.data_ops import dispatcher as data_dispatcher
from pipeline.subset_generator import gen_save_subset
from pipeline.utils import load_model
from pipeline.calc_influence import compute_inf_mat

def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--data_folder", required=True, type=str)
    parser.add_argument("--subset_ratio", required=False, type=float, default=None)
    parser.add_argument("--no_subset", action="store_true")
    # Model arguments
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--model_conf", required=True, type=str)
    parser.add_argument("--training_conf", required=False, type=str)
    # Error injection arguments
    parser.add_argument("--error", required=True, type=str)
    parser.add_argument("--contamination", required=True, type=float)
    parser.add_argument("--classes_to_cont", required=False, default=1, type=int)
    # Influence matrix arguments
    parser.add_argument("--inf_fn_conf", required=False, type=str, default=None)
    # Global arguments
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True, type=str, default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dfolder_name = (
        f"subset_{args.seed}_{args.subset_ratio}" if not args.no_subset else "full"
    )
    data_savedir = os.path.join("results", args.data_name, dfolder_name)

    # Generate Subset

    print(f"Generating the subset for {args.data_name} with ratio {args.subset_ratio}")

    trainset, testset, trainset_labels, testset_labels = data_dispatcher[
        args.data_name
    ]().load_data(args.data_folder)

    if not args.no_subset:
        trainset = gen_save_subset(
            trainset,
            trainset_labels,
            args.subset_ratio,
            data_savedir,
            "clean_train.pt",
            args.seed,
        )
        testset = gen_save_subset(
            testset,
            testset_labels,
            args.subset_ratio,
            data_savedir,
            "clean_test.pt",
            args.seed,
        )

        trainset_labels = trainset.tensors[1]
        testset_labels = testset.tensors[1]

    # Run clean model on subset

    model_savedir = os.path.join(
        data_savedir, args.model_name
    )

    cm = ConfigManager(model_conf=args.model_conf, training_conf=args.training_conf, inf_func_conf=args.inf_fn_conf)
    num_classes = len(trainset.tensors[1].unique())
    input_shape = trainset.tensors[0].shape[1:]

    # Add Errors
    dirty_folder_name = 'dirty_train'

    error_savedir = Path(
        model_savedir,
        dirty_folder_name,
        args.error,
        "contamination_" + str(args.contamination),
        f"cont_{args.classes_to_cont}_classes" if 'cb' in args.error else ''
    )

    error_col = None

    if args.error == 'mislabelled_uniform':
        dirty_dataset, error_col = mislabelled_uniform(
            dataset=trainset,
            labels=trainset_labels,
            contamination_ratio=args.contamination,
            random_seed=args.seed
        )
    elif args.error == 'mislabelled_cb':
        dirty_dataset, error_col = mislabelled_cb(
            dataset=trainset,
            labels=trainset_labels,
            contamination_ratio=args.contamination,
            random_seed=args.seed,
            classes_to_cont=args.classes_to_cont
        )
    else:
        raise AssertionError(f'{args.error} is unknown options are mislabelled_uni, milsabelled_cb or cont_label_one_class')

    print(sum(error_col))

    trainset = dirty_dataset
    trainset_labels = dirty_dataset.tensors[1]

    # Run Dirty Model

    dirty_ckpt_savedir = os.path.join(error_savedir, 'ckpts')

    dirty_model = load_model(
        model_config=cm.model_conf, input_shape=input_shape, num_classes=num_classes
    )

    dirty_trainloader, dirty_testloader = train_test_loaders(
        train_set=trainset,
        test_set=testset,
        batch_size=cm.training_conf.get_batch_size(),
        seed=args.seed,
    )

    Path(dirty_ckpt_savedir).mkdir(parents=True, exist_ok=True)

    dirty_model, dirty_m_info, dirty_preds_info = run_model(
        trainloader=dirty_trainloader,
        testloader=dirty_testloader,
        clean_model=dirty_model,
        cm=cm,
        ckpt_savedir=dirty_ckpt_savedir,
        device=args.device,
    )

    # Compute influence matrix

    self_inf_arr, train_test_im = None, None

    tracin_cp = TracInInfluenceTorch(
        model_instance=dirty_model,
        save_models_path=dirty_ckpt_savedir,
        random_state=args.seed,
        input_shape=None,
        learning_rate=None,
        batch_size=None,
        epochs=None,
        reg_strength=None,
    )

    fast_cp = False
    inf_batch_size = 4096

    if args.inf_fn_conf:
        inf_batch_size = cm.inf_func_conf.batch_size()
        fast_cp = cm.inf_func_conf.fast_cp()


    self_inf_arr, train_test_im = compute_inf_mat(
        tracin_cp,
        trainset,
        testset,
        dirty_model.trainable_layer_names(),
        e_id=None,
        column_wise=False,
        batch_size=inf_batch_size,
        fast_cp=fast_cp
    )

    # Computing

    ies = InfluenceErrorSignals()

    _, nil_opt_labels = ies.nil_opt_fast(train_test_inf_mat=train_test_im,
                     y_train=trainset_labels,
                     y_test=testset_labels)

    im_new = tracin_cp.compute_train_to_test_influence(
        train_set=TensorDataset(trainset.tensors[0], torch.tensor(nil_opt_labels)),
        test_set=testset,
        load_pretrained_model=True,
        batch_size=inf_batch_size,
        layers=dirty_model.trainable_layer_names(),
        epoch_ids_to_consider=[cm.training_conf.get_epochs()],
        fast_cp=True,
    )

    cnci, _ = ies.nil_fast(train_test_inf_mat=im_new, y_train=trainset_labels, y_test=testset_labels)

    cnci = -cnci # This is because NIL returns the absolute negative influence

    # Compute influence signals

    si = self_inf_arr
    mai, _ = ies.mai_fast(train_test_inf_mat=train_test_im)
    gd_class, _ = ies.gd_class_fast(train_test_inf_mat=train_test_im, y_train=trainset_labels, y_test=testset_labels)
    mi, _ = ies.mi_fast(train_test_inf_mat=train_test_im)

    contamination = sum(error_col) / len(error_col)

    cnci_f1 = calc_f1(scores=cnci, error_col=error_col, contamination=contamination)
    si_f1 = calc_f1(scores=si, error_col=error_col, contamination=contamination)
    mai_f1 = calc_f1(scores=mai, error_col=error_col, contamination=contamination)
    gd_class_f1 = calc_f1(scores=gd_class, error_col=error_col, contamination=contamination)
    mi_f1 = calc_f1(scores=mi, error_col=error_col, contamination=contamination)

    f1_perfs = [cnci_f1, si_f1, mai_f1, gd_class_f1, mi_f1]
    names = ['CNCI (ours)', 'SI', 'MAI', 'GD-class', 'MI']
    df = pd.DataFrame(f1_perfs, names)
    print(df)
    df.T.plot.bar(rot=0)
    plt.xticks([])
    plt.xlabel('Influence-based signals for ' + args.model_name)
    plt.ylabel('F1-Score')
    plt.ylim((0, 1))
    plt.title(args.data_name + ' ' + args.error)
    plt.savefig(args.error + '.png')
    plt.show()