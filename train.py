import argparse
import datetime
from typing import Callable

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Accuracy, Metric
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from model import SimpleCNN

LossFunctionType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-e",
    "--epochs",
    default=200,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)

parser.add_argument(
    "-b", "--batch-size", default=128, type=int, metavar="N", help="Batch size"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="Learning rate",
    dest="lr",
)

args, _ = parser.parse_known_args()


def _train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: LossFunctionType,
    metrics_fn: Metric,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()
    num_batches = len(dataloader)
    train_loss = 0.0
    metrics_fn.reset()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        train_loss += loss.item()
        metrics_fn.update(outputs, labels)

        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / num_batches
    avg_train_accuracy = metrics_fn.compute()

    return avg_train_loss, avg_train_accuracy


def _evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: LossFunctionType,
    metrics_fn: Metric,
    device: torch.device,
):
    model.eval()
    num_batches = len(dataloader)
    validation_loss = 0.0
    metrics_fn.reset()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            validation_loss += loss.item()
            metrics_fn.update(outputs, labels)

    avg_validation_loss = validation_loss / num_batches
    avg_validation_accuracy = metrics_fn.compute()

    return avg_validation_loss, avg_validation_accuracy


def train(
    batch_size: int = 128,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
):
    # Mean and STD values calculated this way:
    # Stack all images into a single tensor
    # imgs = torch.stack([img for img, _ in dataset])  # Shape: (50000, 3, 32, 32)
    # Compute mean and std
    # mean = imgs.mean(dim=[0, 2, 3])  # Mean over (batch, height, width)
    # std = imgs.std(dim=[0, 2, 3])    # Std over (batch, height, width)
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    dataset_train = datasets.CIFAR10(
        root="./datasets/", train=True, download=True, transform=transform_train
    )

    train_size = int(0.8 * len(dataset_train))  # 80% train
    val_size = len(dataset_train) - train_size  # 20% validation
    dataset_train, dataset_val = random_split(dataset_train, [train_size, val_size])

    dataset_test = datasets.CIFAR10(
        root="./datasets/", train=False, download=True, transform=transform
    )

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_validation = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = SimpleCNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=3)

    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)

    best_val_loss = float("inf")
    best_model_path = "best_model.pth"
    writer = SummaryWriter(log_dir=f"./runs/{datetime.datetime.now():%Y%m%d_%H%M%S}")
    writer.add_graph(model, torch.randn(1, 3, 32, 32).to(device))
    for epoch in trange(epochs):
        avg_train_loss, avg_train_accuracy = _train(
            loader_train, model, criterion, metric_fn, optimizer, device
        )
        avg_val_loss, avg_val_accuracy = _evaluate(
            loader_validation, model, criterion, metric_fn, device
        )
        scheduler.step(avg_val_accuracy, epoch)

        writer.add_scalar("Training loss", avg_train_loss, epoch)
        writer.add_scalar("Training accuracy", avg_train_accuracy, epoch)
        writer.add_scalar("Validation loss", avg_val_loss, epoch)
        writer.add_scalar("Validation accuracy", avg_val_accuracy, epoch)

        tqdm.write(
            f"Epoch {epoch + 1}/{epochs} | train loss: {avg_train_loss:.6f} | train acc: {avg_train_accuracy:.6f} | val loss: {avg_val_loss:.6f} | val acc: {avg_val_accuracy:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    # Loading best weights for test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    avg_test_loss, avg_test_accuracy = _evaluate(
        loader_test, model, criterion, metric_fn, device
    )

    writer.add_scalar("Test loss", avg_test_loss, 0)
    writer.add_scalar("Test accuracy", avg_test_accuracy, 0)
    print(f"Best weights accuracy: {avg_test_accuracy:.6f}")


if __name__ == "__main__":
    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
