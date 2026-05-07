import glob
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

from pipeline.model_train import train

import torch
from torch.utils.data import DataLoader


def load_model(model, ckpt_fp):
	state_dict = torch.load(ckpt_fp)
	model.load_state_dict(state_dict["model_state_dict"])
	model.eval()
	return model


def get_last_ckpt(ckpt_dir):
	file_list = glob.glob(os.path.join(ckpt_dir, "*.pt"))
	highest_number = -1
	highest_checkpoint = None
	for file_path in file_list:
		if file_path.endswith(".pt") and file_path.count("-") > 0:
			parts = file_path.split("-")
			try:
				checkpoint_number = int(parts[-1].split(".pt")[0])
				if checkpoint_number > highest_number:
					highest_number = checkpoint_number
					highest_checkpoint = file_path
			except ValueError:
				pass  # Ignore files with non-numeric numbers
	return highest_checkpoint


def train_test_loaders(train_set, test_set, batch_size, seed, workers=15):
	if seed is not None:
		torch.manual_seed(seed)
	trainloader = DataLoader(
		train_set,
		batch_size=batch_size,
		shuffle=True,
		num_workers=workers,
	)
	testloader = DataLoader(
		test_set,
		batch_size=batch_size,
		shuffle=False,
		num_workers=workers,
	)
	return trainloader, testloader


def run_model(trainloader, testloader, clean_model, cm, ckpt_savedir, device, save_ckpts=True):
	clean_model, info, preds_info = train(
		model=clean_model,
		epochs=cm.training_conf.get_epochs(),
		learning_rate=cm.training_conf.get_learning_rate(),
		reg_strength=cm.training_conf.model_regularization_strength(),
		save_dir=ckpt_savedir,
		train_loader=trainloader,
		test_loader=testloader,
		device=device,
		save_ckpts=save_ckpts,
	)
	return clean_model, info, preds_info


def save_as_json(data, fp):
	fp_folder = Path(fp).parent
	Path(fp_folder).mkdir(parents=True, exist_ok=True)
	with open(fp, "w") as f:
		json.dump(data, f)


def save_as_np(data, fp):
	fp_folder = Path(fp).parent
	Path(fp_folder).mkdir(parents=True, exist_ok=True)
	with open(fp, "wb") as f:
		np.save(f, data)


def calc_f1(scores, error_col, contamination):
	anoms = np.zeros(len(scores), dtype=int)
	score_thresh = np.sort(scores)[::-1][int(np.floor(contamination * len(scores)))]
	anoms[scores >= score_thresh] = 1
	return f1_score(error_col, anoms)