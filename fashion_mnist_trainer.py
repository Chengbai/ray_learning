# import sys
# import os

# # Get the absolute path of the current script's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # modle/FashionMNIST => ./
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
# print(f"parent_dir1: {parent_dir}")

import torch
from tqdm import tqdm

from datetime import datetime
from model.FashionMNIST.dataset import get_dataloaders
from model.FashionMNIST.model import GarmentClassifier
from model.utils.config import Config
from torch.utils.data import DataLoader
from model.utils.utils import get_device
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(
    config: Config,
    training_loader: DataLoader,
    epoch_index: int,
    optimizer: torch.optim.Optimizer,
    tb_writer: SummaryWriter,
):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def train(
    config: Config, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn
):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
    epoch_number = 0
    best_vloss = 1_000_000.0

    training_loader, validation_loader = get_dataloaders()
    for epoch in tqdm(range(config.epochs)):
        print("EPOCH {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            config=config,
            training_loader=training_loader,
            epoch_index=epoch_number,
            optimizer=optimizer,
            tb_writer=writer,
        )

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(config.device)
                vlabels = vlabels.to(config.device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (epoch + 1)
        print(f"Epoch: {epoch}, LOSS: train {avg_loss} valid {avg_vloss}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "model_{}_{}.pt".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


config = Config(device=get_device(), epochs=5)
model = GarmentClassifier().to(config.device)
optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
loss_fn = torch.nn.CrossEntropyLoss()

train(config, model, optimizer, loss_fn)
