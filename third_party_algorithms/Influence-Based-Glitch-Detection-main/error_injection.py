import copy
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset, TensorDataset
import torch
from torchvision import transforms
from itertools import combinations


def flip_labels(labels, indices_to_flip, dataset):
	unq_labels = np.unique(labels)
	dirty_labels = copy.deepcopy(labels)
	samples = dataset.tensors[0]
	labels = dataset.tensors[1]
	# Flip the labels for the selected indices
	for counter, idx in enumerate(indices_to_flip):
		remaining_labels = list(set(unq_labels.tolist()).difference([labels[idx]]))
		np.random.seed(counter)
		flipped_label = np.random.choice(remaining_labels, 1)[0]
		dirty_labels[idx] = flipped_label
	if not isinstance(dirty_labels, torch.Tensor):
		dirty_labels = torch.tensor(dirty_labels)
	return TensorDataset(samples, dirty_labels)


def mislabelled_uniform(dataset, labels, random_seed=None, contamination_ratio=0.1):
	dataset = copy.deepcopy(dataset)
	indices_to_flip, _ = train_test_split(
		np.arange(len(labels)), train_size=contamination_ratio, random_state=random_seed
	)
	error_column = np.array([0] * len(dataset))
	error_column[indices_to_flip] = 1
	dataset = flip_labels(
		labels=labels, indices_to_flip=indices_to_flip, dataset=dataset
	)
	return dataset, error_column


def mislabelled_cb(dataset, labels, random_seed=None, contamination_ratio=0.1, classes_to_cont=1):
	dataset = copy.deepcopy(dataset)
	unq_labels = np.unique(labels)
	labels_tmp = labels.detach().cpu().numpy().copy()
	np.random.seed(random_seed)
	np.random.shuffle(unq_labels)
	np.random.seed(random_seed)
	error_column = np.array([0] * len(dataset))
	combs1 = {k: v for k,v in combinations(unq_labels, 2)}
	combs2 = {v: k for k, v in combinations(unq_labels, 2)}
	combs = {**combs1, **combs2}
	assert len(combs) == len(unq_labels)
	for i, (c1, c2) in enumerate(combs.items()):
		if i >= classes_to_cont:
			break
		base_class_ids = np.where(labels == c1)[0]
		np.random.seed(random_seed)
		indices_to_flip = np.random.choice(
			base_class_ids,
			size=int(np.floor(contamination_ratio * len(base_class_ids))),
			replace=False,
		)
		error_column[indices_to_flip] = 1
		labels_tmp[indices_to_flip] = [c2] * len(indices_to_flip)
	dataset = TensorDataset(dataset.tensors[0], torch.tensor(labels_tmp))
	return dataset, error_column

def inj_anomalies(dataset, labels, ood_dataset, contamination_ratio=0.1, random_seed=None):
	dataset = copy.deepcopy(dataset)
	ood_sample_size = min(len(ood_dataset), int(np.floor(contamination_ratio * len(dataset))))
	ood_labels = ood_dataset.targets
	if type(ood_labels) == torch.Tensor:
		ood_labels = ood_labels.numpy()
	else:
		ood_labels = np.array(ood_labels)
	sel_ood_ids, ood_in_class = _select_anom_ids(random_seed, labels, ood_labels, ood_sample_size)
	ood_subset = Subset(ood_dataset, sel_ood_ids)
	ood_subset = _prepare_anom(dataset, ood_subset, ood_in_class)
	final_dataset = ConcatDataset([dataset, ood_subset])
	data = torch.stack([sample[0] for sample in final_dataset])
	targets = torch.tensor([sample[1] for sample in final_dataset])
	error_column = np.array([0] * len(final_dataset))
	error_column[len(dataset):] = [1] * ood_sample_size
	return TensorDataset(data, targets), error_column

def _select_anom_ids(random_seed, labels, ood_labels, ood_sample_size):
	np.random.seed(random_seed)
	ood_class = np.random.choice(np.unique(ood_labels), size=1, replace=False)[0]
	ood_ids = np.where(ood_labels == ood_class)[0]
	np.random.seed(random_seed)
	sel_ood_ids = np.random.choice(ood_ids, size=ood_sample_size, replace=False)
	np.random.seed(random_seed)
	ood_in_class = np.array([np.random.choice(np.unique(labels), size=1, replace=False)[0]] * ood_sample_size)
	return sel_ood_ids, ood_in_class

def _prepare_anom(dataset, ood_dataset, ood_in_class):
	x_og, _ = dataset[0]
	x_ood, _ = ood_dataset[0]
	width = x_og.shape[1]
	height = x_og.shape[2]
	channels_og = x_og.shape[0]
	ood_transforms_list = [transforms.Resize((width, height))]
	if channels_og > x_ood.shape[0]:
		ood_transforms_list.append(lambda x: torch.cat([x, x, x], dim=0))
	ood_transforms = transforms.Compose(ood_transforms_list)
	ood_data = torch.stack([ood_transforms(sample) for sample, _ in ood_dataset])
	ood_labels = torch.tensor(ood_in_class)
	return TensorDataset(ood_data, ood_labels)
