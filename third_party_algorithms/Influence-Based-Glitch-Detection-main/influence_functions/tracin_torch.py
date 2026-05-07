import glob
import os
from typing import Any, List

import numpy as np
import torch
import torch.optim as optim
from captum.influence import TracInCP, TracInCPFast
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class TracInInfluenceTorch:
    def __init__(
        self,
        model_instance: nn.Module,
        input_shape: List,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        save_models_path: str,
        reg_strength: float = 0.0,
        random_state: int = None,
        loss_fn: Any = nn.CrossEntropyLoss(),
    ):
        # constructor arguments
        self.model_instance = model_instance
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.save_models_path = save_models_path
        self.random_state = random_state
        self.loss_fn = loss_fn
        self.reg_strength = reg_strength

        # class variables
        self.__models_are_built = False

    def build_influence_models(self, train_set, test_set):
        device = "cpu"
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        train_dataloader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        test_dataloader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.model_instance.to(device)

        if not os.path.exists(self.save_models_path):
            os.makedirs(self.save_models_path)

        optimizer = optim.SGD(
            self.model_instance.parameters(), lr=self.lr, weight_decay=self.reg_strength
        )

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:  # loop over the dataset multiple times
            epoch_loss = 0.0
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model_instance(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_dataloader)

            checkpoint_name = "-".join(["checkpoint", str(epoch) + ".pt"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model_instance.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "learning_rate": self.lr,
                },
                os.path.join(self.save_models_path, checkpoint_name),
            )

            # Calcualate validation accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # calculate outputs by running images through the network
                    outputs = self.model_instance(inputs)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = correct / total
            pbar.set_description(
                f"Epoch {epoch + 1:03d}, Train Loss: {epoch_loss:.4f}, Test Acc: {test_acc: .2f}"
            )

        self.__models_are_built = True

    def compute_train_to_test_influence(
        self,
        train_set,
        test_set,
        load_pretrained_model: bool = False,
        batch_size: int = 3500,
        epoch_ids_to_consider: List[int] = None,
        layers=None,
        fast_cp=False,
    ) -> np.ndarray:
        print("Computing Train to Test Influence")
        if not (load_pretrained_model or self.__models_are_built):
            raise AssertionError(
                "Model checkpoints are not created. Call the function build_influence_models"
            )
        if epoch_ids_to_consider is None:
            checkpoints_to_consider = glob.glob(
                os.path.join(self.save_models_path, "*.pt")
            )
        else:
            checkpoints_to_consider = [
                os.path.join(
                    self.save_models_path, "-".join(["checkpoint", str(e_id) + ".pt"])
                )
                for e_id in epoch_ids_to_consider
            ]
        if fast_cp:
            tracin_cp = TracInCPFast(
                model=self.model_instance,
                train_dataset=train_set,
                final_fc_layer=layers[-1],
                checkpoints=checkpoints_to_consider,
                checkpoints_load_func=TracInInfluenceTorch.checkpoints_load_func,
                loss_fn=self.loss_fn,
                batch_size=batch_size,
            )
        else:
            tracin_cp = TracInCP(
                self.model_instance,
                train_set,
                checkpoints_to_consider,
                checkpoints_load_func=TracInInfluenceTorch.checkpoints_load_func,
                loss_fn=self.loss_fn,
                batch_size=batch_size,
                sample_wise_grads_per_batch=True,
                layers=layers,
            )

        test_examples_features = torch.stack(
            [test_set[i][0] for i in range(len(test_set))]
        )
        test_examples_true_labels = torch.Tensor(
            [test_set[i][1] for i in range(len(test_set))]
        ).long()

        train_to_test_influence = tracin_cp.influence(
            (test_examples_features, test_examples_true_labels), show_progress=True
        )

        return np.array(train_to_test_influence).transpose()

    def compute_self_influence(
        self,
        dataset,
        load_pretrained_model: bool = False,
        batch_size: int = 3500,
        epoch_ids_to_consider: List[int] = None,
        layers=None,
        fast_cp=False,
    ) -> np.ndarray:
        print("Computing Self Influence")
        print(layers)
        if not (load_pretrained_model or self.__models_are_built):
            raise AssertionError(
                "Model checkpoints are not created. Call the function build_influence_models"
            )
        if epoch_ids_to_consider is None:
            checkpoints_to_consider = glob.glob(
                os.path.join(self.save_models_path, "*.pt")
            )
        else:
            checkpoints_to_consider = [
                os.path.join(
                    self.save_models_path, "-".join(["checkpoint", str(e_id) + ".pt"])
                )
                for e_id in epoch_ids_to_consider
            ]
        if fast_cp:
            tracin_cp = TracInCPFast(
                model=self.model_instance,
                train_dataset=dataset,
                final_fc_layer=layers[-1],
                checkpoints=checkpoints_to_consider,
                checkpoints_load_func=TracInInfluenceTorch.checkpoints_load_func,
                loss_fn=self.loss_fn,
                batch_size=batch_size,
            )
        else:
            tracin_cp = TracInCP(
                self.model_instance,
                dataset,
                checkpoints_to_consider,
                checkpoints_load_func=TracInInfluenceTorch.checkpoints_load_func,
                loss_fn=self.loss_fn,
                batch_size=batch_size,
                sample_wise_grads_per_batch=True,
                layers=layers,
            )
        self_influence = tracin_cp.self_influence(show_progress=True)
        return self_influence

    @staticmethod
    def checkpoints_load_func(model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint["learning_rate"]
