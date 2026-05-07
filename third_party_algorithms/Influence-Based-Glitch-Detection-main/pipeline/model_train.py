import os
from pathlib import Path

import numpy as np
import pkbar
import torch
from torch import nn, optim


def train(
    model,
    epochs,
    learning_rate,
    reg_strength,
    save_dir,
    train_loader,
    test_loader,
    device,
    save_ckpts=True,
    ckpt_number=None,
):
    ckpt_number = 0 if ckpt_number is None else ckpt_number

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=reg_strength
    )

    model = model.to(device)

    for epoch in range(0, epochs):
        # every epoch a new progressbar is created
        # also, depending on the epoch the learning rate gets adjusted before
        # the network is set into training mode
        kbar = pkbar.Kbar(
            target=len(train_loader) - 1,
            epoch=epoch,
            num_epochs=epochs,
            always_stateful=True,
        )
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        # iterates over a batch of training data
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            _, predicted = outputs.max(1)

            # calculate the current running loss as well as the total accuracy
            # and update the progressbar accordingly
            running_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            kbar.update(
                batch_idx,
                values=[
                    ("loss", running_loss / (batch_idx + 1)),
                    ("acc", 100.0 * correct / total),
                ],
            )

        # save the model in each epoch
        if save_ckpts:
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            checkpoint_name = "-".join(
                ["checkpoint", str(epoch + 1 + ckpt_number) + ".pt"]
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss,
                    "learning_rate": learning_rate,
                },
                os.path.join(save_dir, checkpoint_name),
            )

        # calculate the test accuracy of the network at the end of each epoch
        with torch.no_grad():
            model.eval()
            t_total = 0
            t_correct = 0
            for _, (inputs_t, targets_t) in enumerate(test_loader):
                inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
                # targets = torch.nn.functional.one_hot(targets)
                outputs_t = model(inputs_t)
                _, predicted_t = outputs_t.max(1)
                t_total += targets_t.size(0)
                t_correct += predicted_t.eq(targets_t).sum().item()
            print("-> test acc: {}".format(100.0 * t_correct / t_total))

    # calculate the test accuracy of the network at the end of the training
    test_preds_labels = None
    test_preds = None
    test_loss = 0
    with torch.no_grad():
        model.eval()
        t_total = 0
        t_correct = 0
        for _, (inputs_t, targets_t) in enumerate(test_loader):
            inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
            outputs_t = model(inputs_t)
            outputs_t = torch.softmax(outputs_t, dim=1)
            curr_preds, curr_preds_labels = outputs_t.max(1)
            t_total += targets_t.size(0)
            t_correct += curr_preds_labels.eq(targets_t).sum().item()
            loss = criterion(outputs_t, targets_t)
            test_loss += loss.item()
            if test_preds_labels is None:
                test_preds_labels = curr_preds_labels.cpu().numpy()
                test_preds = curr_preds.cpu().numpy()
            else:
                test_preds_labels = np.hstack(
                    (test_preds_labels, curr_preds_labels.cpu().numpy())
                )
                test_preds = np.hstack(
                    (test_preds, curr_preds.cpu().numpy())
                )

    # calculate the train accuracy of the network at the end of the training
    train_preds_labels = None
    train_loss = 0

    with torch.no_grad():
        model.eval()
        total_train = 0
        correct_train = 0
        for _, (inputs_t, targets_t) in enumerate(train_loader):
            inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
            outputs_t = model(inputs_t)
            _, curr_preds_labels = outputs_t.max(1)
            total_train += targets_t.size(0)
            correct_train += curr_preds_labels.eq(targets_t).sum().item()
            loss = criterion(outputs_t, targets_t)
            train_loss += loss.item()
            if train_preds_labels is None:
                train_preds_labels = curr_preds_labels.cpu().numpy()
            else:
                train_preds_labels = np.hstack(
                    (train_preds_labels, curr_preds_labels.cpu().numpy())
                )

    train_loss /= total_train
    test_loss /= t_total
    train_acc = correct_train / total_train
    test_acc = t_correct / t_total
    print(
        "Final accuracy: Train: {} | Test: {}".format(
            100.0 * train_acc, 100.0 * test_acc
        )
    )

    perf_info = {
        f"train_acc": train_acc,
        f"test_acc": test_acc,
        f"train_loss": train_loss,
        f"test_loss": test_loss,
    }

    preds_info = {
        'test_preds': test_preds,
        'test_preds_labels': test_preds_labels
    }

    return model, perf_info, preds_info
