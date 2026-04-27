"""
Training and validation loops with early stopping, pos_weight, LR scheduling,
and accuracy tracking.

Loaders yield (x_tab, x_text, y) triples. Models accept (x_tab, x_text) and
return a [B, 1] logit — see models.py for the shared forward-call signature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def _binary_accuracy(logits: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute binary accuracy from logits and ground-truth labels.

    Parameters
    ----------
    logits
        Raw model logits for a batch of binary predictions.
    y
        Ground-truth binary labels aligned with ``logits``.
    threshold
        Probability cutoff used after the sigmoid transform to form predictions.

    Returns
    -------
    accuracy
        Fraction of correct predictions in the batch.
    """
    pred = (torch.sigmoid(logits) >= threshold).float()
    return (pred == y).float().mean().item()


def _compute_pos_weight(loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    """
    Compute the positive-class weight used by ``BCEWithLogitsLoss``.

    Parameters
    ----------
    loader
        Training DataLoader used to count positive and negative labels.
    device
        Device on which the resulting weight tensor should be allocated.

    Returns
    -------
    pos_weight
        Scalar tensor equal to ``n_negative / n_positive`` for the training data.
    """
    n_pos = 0
    n_neg = 0
    for batch in loader:
        yb = batch[2]  # (x_tab, x_text, y)
        n_pos += int((yb > 0.5).sum().item())
        n_neg += int((yb <= 0.5).sum().item())
    if n_pos == 0:
        return torch.tensor([1.0], device=device)
    return torch.tensor([n_neg / n_pos], device=device)


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    """
    Run one pass over a loader and return the mean loss and accuracy.

    Parameters
    ----------
    model
        Binary classifier to train or evaluate.
    loader
        DataLoader yielding ``(x_tab, x_text, y)`` batches.
    criterion
        Loss function applied to model logits and target labels.
    device
        Device on which the model and batch tensors should be processed.
    optimizer
        Optimizer used for parameter updates, or ``None`` to run evaluation only.

    Returns
    -------
    epoch_metrics
        Tuple containing the mean loss and accuracy for the loader pass.
    """
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for x_tab, x_text, yb in loader:
        x_tab = x_tab.to(device)
        x_text = x_text.to(device)
        yb = yb.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(x_tab, x_text)
            loss = criterion(logits, yb)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = len(x_tab)
        total_loss += loss.item() * batch_size
        with torch.no_grad():
            total_correct += _binary_accuracy(logits, yb) * batch_size
        total_examples += batch_size

    denom = max(total_examples, 1)
    return total_loss / denom, total_correct / denom


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch and summarize the batch-level results.

    Parameters
    ----------
    model
        Binary classifier to optimize for one full pass over ``loader``.
    loader
        Training DataLoader yielding ``(x_tab, x_text, y)`` batches.
    optimizer
        Optimizer used to update the model parameters.
    criterion
        Loss function applied to model logits and target labels.
    device
        Device on which the model and batch tensors should be processed.

    Returns
    -------
    epoch_metrics
        Tuple containing the mean training loss and accuracy for the epoch.
    """
    return _run_epoch(model, loader, criterion, device, optimizer=optimizer)


def evaluate_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on one loader and summarize the loss and accuracy.

    Parameters
    ----------
    model
        Binary classifier to evaluate without gradient updates.
    loader
        DataLoader yielding the split to score.
    criterion
        Loss function applied to model logits and target labels.
    device
        Device on which the model and batch tensors should be processed.

    Returns
    -------
    evaluation_metrics
        Tuple containing the mean loss and accuracy across the loader.
    """
    return _run_epoch(model, loader, criterion, device)


def plot_training_history(history: dict[str, list[float]], path: str | Path) -> None:
    """
    Save loss and accuracy curves for the recorded training history.

    Parameters
    ----------
    history
        Dictionary containing per-epoch train and validation losses and accuracies.
    path
        Output path where the training-curve image should be written.

    Returns
    -------
    None
        Writes the training-history figure to disk.
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", color="#2563eb", linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"],   label="val",   color="#dc2626", linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCEWithLogits)")
    axes[0].set_title("Training & validation loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train acc", color="#2563eb", linewidth=1.8)
    axes[1].plot(epochs, history["val_acc"],   label="val acc",   color="#dc2626", linewidth=1.8)
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
    Train a model with early stopping and restore the best validation checkpoint.

    Parameters
    ----------
    model
        Binary classifier to train on the provided loaders.
    train_loader
        DataLoader supplying the training batches.
    val_loader
        DataLoader supplying the validation batches used for checkpointing.
    epochs
        Maximum number of training epochs to run.
    lr
        Learning rate used by the Adam optimizer.
    patience
        Number of consecutive non-improving validation epochs tolerated before stopping.
    weight_decay
        L2 regularization strength applied by the optimizer.
    device
        Device on which the model and batch tensors should be processed.
    checkpoint_path
        Path where the best model checkpoint should be saved during training.
    on_epoch
        Optional callback invoked after each epoch with the epoch index and metric summary.
    verbose
        Whether to print per-epoch training progress and early-stopping messages.
    plot_path
        Optional output path for a training-history plot written after training finishes.

    Returns
    -------
    history
        Dictionary containing per-epoch train and validation loss and accuracy values.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    pos_weight = _compute_pos_weight(train_loader, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val = float("inf")
    stale    = 0
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
    }
    ckpt = Path(checkpoint_path)

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate_metrics(model, val_loader,   criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        metrics = {"train_loss": tr_loss, "val_loss": va_loss,
                   "train_acc": tr_acc,   "val_acc":  va_acc}
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
            stale    = 0
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
