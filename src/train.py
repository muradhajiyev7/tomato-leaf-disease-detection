from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DEVICE


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def run_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = DEVICE,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return EpochMetrics(loss=total_loss / max(total, 1), accuracy=correct / max(total, 1))


@torch.no_grad()
def run_validation_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return EpochMetrics(loss=total_loss / max(total, 1), accuracy=correct / max(total, 1))

