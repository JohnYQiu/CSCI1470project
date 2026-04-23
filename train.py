"""
Training and validation loops with early stopping, checkpointing, and accuracy tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Adam


def _binary_accuracy(logits: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of correct predictions at ``threshold`` on sigmoid(logits)."""
    pred = (torch.sigmoid(logits) >= threshold).float()
    return (pred == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean loss, accuracy) for the epoch."""
    model.train()
    total_loss = 0.0
    n = 0
    correct = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        bs = len(xb)
        total_loss += loss.item() * bs
        n += bs
        with torch.no_grad():
            correct += _binary_accuracy(logits, yb) * bs
    denom = max(n, 1)
    return total_loss / denom, correct / denom


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean loss, accuracy) on the loader."""
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = len(xb)
        total_loss += loss.item() * bs
        n += bs
        correct += _binary_accuracy(logits, yb) * bs
    denom = max(n, 1)
    return total_loss / denom, correct / denom


def plot_training_history(history: dict[str, list[float]], path: str | Path) -> None:
    """
    Save train/val loss and accuracy curves (two subplots).
    Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", color="#2563eb", linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"], label="val", color="#dc2626", linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCEWithLogits)")
    axes[0].set_title("Training & validation loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train acc", color="#2563eb", linewidth=1.8)
    axes[1].plot(epochs, history["val_acc"], label="val acc", color="#dc2626", linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & validation accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    checkpoint_path: str | Path = "best_model.pt",
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
    verbose: bool = False,
    plot_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    BCEWithLogitsLoss + Adam. Saves state dict with lowest validation loss.

    Returns history with ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc`` per epoch.
    If ``verbose``, prints loss and accuracy each epoch. If ``plot_path`` is set, saves a
    figure when training finishes (including early stop).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    stale = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    ckpt = Path(checkpoint_path)

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate_metrics(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        metrics = {
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_acc": tr_acc,
            "val_acc": va_acc,
        }
        if on_epoch:
            on_epoch(epoch, metrics)
        if verbose:
            print(
                f"  epoch {epoch + 1:4d}/{epochs}  "
                f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
                f"train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}"
            )

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            stale = 0
            torch.save({"model_state_dict": model.state_dict(), "val_loss": va_loss}, ckpt)
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch + 1} (patience={patience})")
                break

    if ckpt.is_file():
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    if plot_path is not None:
        plot_training_history(history, plot_path)

    return history
