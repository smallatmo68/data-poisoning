import itertools
import os.path
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, Subset, ConcatDataset

from anomaly_pipeline import compute_second_best_pil, compute_il_unreduced, calc_f1
from influence_functions import TracInInfluenceTorch
from signals_fast import InfluenceErrorSignals
from utils import train_test_loaders, run_model, get_last_ckpt, save_as_np, save_as_json
from pipeline.config_manager import ConfigManager
from pipeline.subset_generator import gen_save_subset
from pipeline.data_ops import dispatcher as data_dispatcher
from error_injection import mislabelled_uniform, inj_anomalies

import torch
from pipeline.utils import load_model
from pipeline.calc_influence import compute_inf_mat

ood_datasets_dict = {
    'mnist': 'fmnist',
    'fmnist': 'mnist',
    'cifar10': 'fmnist',
}

def parse_args():
	parser = argparse.ArgumentParser()
	# Dataset arguments
	parser.add_argument("--subset_ratio", required=True, type=float)
	parser.add_argument("--data_name", required=True, type=str)
	parser.add_argument("--data_folder", required=True, type=str)

	# Model arguments
	parser.add_argument("--model_name", required=True, type=str)
	parser.add_argument("--model_conf", required=True, type=str)
	parser.add_argument("--training_conf", required=False, type=str)

	# Influence matrix arguments
	parser.add_argument("--inf_fn_conf", required=False, type=str, default=None)

	# Global arguments
	parser.add_argument("--device", required=False, type=str, default="cpu")
	parser.add_argument("--savedir", required=False, type=str, default=None)
	parser.add_argument("--seed", required=True, type=int, default=42)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	dfolder_name = f"subset_{args.seed}_{args.subset_ratio}"

	data_savedir = os.path.join("results", args.data_name, dfolder_name)

	# Generate Subset

	print(f"Generating the subset for {args.data_name} with ratio {args.subset_ratio}")

	trainset, testset, trainset_labels, testset_labels = data_dispatcher[
		args.data_name
	]().load_data(args.data_folder)

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

	num_classes = len(torch.unique(trainset.tensors[1]))

	error_names = np.array(['clean'] * len(trainset))

	initial_sample_size = len(trainset)

	trainset, error_col = mislabelled_uniform(
		dataset=trainset,
		labels=trainset.tensors[1].numpy(),
		contamination_ratio=0.1,
		random_seed=args.seed
	)

	dirty_ids = np.where(error_col == 1)[0]
	error_names[dirty_ids] = 'mu'

	# inject ood

	_, ood_testset, _, _ = data_dispatcher[
		ood_datasets_dict[args.data_name]
	]().load_data(args.data_folder)

	trainset, _ = inj_anomalies(
		dataset=trainset,
		labels=trainset.tensors[1].numpy(),
		ood_dataset=ood_testset,
		contamination_ratio=0.1,
		random_seed=args.seed,
	)

	error_names = np.array([*error_names.tolist(), *['ua']*(len(trainset) - initial_sample_size)])

	error_col = np.array([0] * len(error_names))
	dirty_s = np.where(error_names != 'clean')[0]
	error_col[dirty_s] = 1

	trainset_labels = trainset.tensors[1]
	testset_labels = testset.tensors[1]

	# Run clean model on subset

	contamination = sum(error_col) / len(error_col)

	error_savedir = os.path.join(
		'results',
		args.data_name,
		f"subset_{args.seed}_{args.subset_ratio}",
		'dirty_train',
		'mixed',
		"contamination_" + str(contamination),
	)

	# Run Dirty Model

	dirty_ckpt_savedir = os.path.join(error_savedir, 'ckpts')

	input_shape = trainset.tensors[0].shape[1:]


	cm = ConfigManager(model_conf=args.model_conf, training_conf=args.training_conf, inf_func_conf=args.inf_fn_conf)

	dirty_model = load_model(
		model_config=cm.model_conf, input_shape=input_shape, num_classes=num_classes
	)

	print(f"Training for dirty data the model {cm.model_conf}")

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


	# calculate CFI, CFID

	ies = InfluenceErrorSignals()
	_, nil_opt_labels = ies.nil_opt_fast(train_test_inf_mat=train_test_im,
										 y_train=trainset_labels, y_test=testset_labels)

	print('Calculating counterfactual influence')

	im_new = tracin_cp.compute_train_to_test_influence(
		train_set=TensorDataset(trainset.tensors[0], torch.tensor(nil_opt_labels)),
		test_set=testset,
		load_pretrained_model=True,
		batch_size=inf_batch_size,
		layers=dirty_model.trainable_layer_names(),
		epoch_ids_to_consider=[cm.training_conf.get_epochs()],
		fast_cp=fast_cp,
	)

	# CNCI

	cnci, _ = ies.nil_fast(train_test_inf_mat=im_new, y_train=trainset_labels, y_test=testset_labels)
	cnci = -cnci

	# PCID

	ies = InfluenceErrorSignals()

	positive_counterfactuals = compute_second_best_pil(train_test_inf=train_test_im,
													   y_train=trainset_labels,
													   y_test=testset_labels)

	im_new = tracin_cp.compute_train_to_test_influence(
		train_set=TensorDataset(trainset.tensors[0], torch.tensor(positive_counterfactuals)),
		test_set=testset,
		load_pretrained_model=True,
		batch_size=inf_batch_size,
		layers=dirty_model.trainable_layer_names(),
		epoch_ids_to_consider=[cm.training_conf.get_epochs()],
		fast_cp=True,
	)

	pil_unreduced, _ = compute_il_unreduced(train_test_inf=train_test_im, y_train=trainset_labels, y_test=testset_labels)

	pil_unreduced_cf, _ = compute_il_unreduced(train_test_inf=im_new, y_train=positive_counterfactuals, y_test=testset_labels)

	norm = np.linalg.norm(pil_unreduced - pil_unreduced_cf, ord=np.inf, axis=0)

	w, _ = ies.mpi_fast(train_test_inf_mat=train_test_im)

	pcid = 1 / (w * norm)

	# CFRank

	cnci_rank = pd.DataFrame(np.arange(len(cnci)), index=np.argsort(cnci), columns=['cnci_r']).sort_index()
	pcfid_rank = pd.DataFrame(np.arange(len(pcid)), index=np.argsort(pcid), columns=['pcfid_r']).sort_index()
	cf_ranks = pd.concat([cnci_rank, pcfid_rank], axis=1).max(axis=1).sort_index()

	cnci_rank = cnci_rank.values.ravel()
	pcfid_rank = pcfid_rank.values.ravel()
	cf_ranks = cf_ranks.values.ravel()

	# Console Logging

	print('F1-score performance comparison')

	print('CNCI F1-score', calc_f1(scores=cnci, error_col=error_col, contamination=contamination))
	print('PCID F1-score', calc_f1(scores=pcid, error_col=error_col, contamination=contamination))
	print('CFRank F1-score', calc_f1(scores=cf_ranks, error_col=error_col, contamination=contamination))


	# Error characterization by CFRank

	print('Error characterization by CFRank')

	characterization_df = pd.DataFrame(error_names, columns=['err_name'])
	characterization_df['cf_rank'] = cf_ranks
	characterization_df['cnci_rank'] = cnci_rank
	characterization_df['pcfid_rank'] = pcfid_rank
	characterization_df = characterization_df.sort_values(by='cf_rank', ascending=False)
	characterization_df = characterization_df.iloc[:int(np.floor(contamination * len(characterization_df))), :]
	characterization_df = characterization_df[characterization_df['err_name'] != 'clean']
	characterization_df = characterization_df.replace({'mu': 0, 'ua': 1})
	characterization_df = characterization_df.rename(columns={'err_name': 'error_label'})
	pred_error_labels = np.argmax(characterization_df[['cnci_rank', 'pcfid_rank']].values, axis=1)
	error_labels = characterization_df['error_label'].values
	mislabelled = np.where(error_labels == 0)[0]
	mislabelled_acc = accuracy_score(error_labels[mislabelled], pred_error_labels[mislabelled])
	anom = np.where(error_labels == 1)[0]
	anom_acc = accuracy_score(error_labels[anom], pred_error_labels[anom])

	print('Accuracy of characterizing the detected mislabeled samples:', mislabelled_acc)
	print('Accuracy of characterizing the detected anomalous samples:', anom_acc)