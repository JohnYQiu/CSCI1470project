"""
Classification metrics and confusion matrix (numpy / sklearn).

Loaders yield (x_tab, x_text, y) triples; models accept (x_tab, x_text).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over a loader and collect probabilities with ground-truth labels.

    Parameters
    ----------
    model
        Trained binary classifier that accepts tabular and text inputs.
    loader
        DataLoader yielding ``(x_tab, x_text, y)`` batches to evaluate.
    device
        Device on which the model and input tensors should be evaluated.

    Returns
    -------
    predictions
        Tuple containing the predicted probabilities and the corresponding true labels
        as NumPy arrays.
    """
    model.eval()
    probs:  list[float] = []
    labels: list[float] = []
    for x_tab, x_text, yb in loader:
        x_tab  = x_tab.to(device)
        x_text = x_text.to(device)
        logits = model(x_tab, x_text).view(-1)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(yb.view(-1).cpu().numpy().tolist())
    return np.asarray(probs), np.asarray(labels)


def find_best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Sweep candidate thresholds and select the one that maximizes F1 score.

    Parameters
    ----------
    probs
        Predicted positive-class probabilities for each example.
    y_true
        Ground-truth binary labels aligned with ``probs``.

    Returns
    -------
    best_threshold
        Threshold that yields the highest F1 score on the provided predictions.
    """
    y_true_i = y_true.astype(int)
    best_t  = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (probs >= t).astype(int)
        f = float(f1_score(y_true_i, y_pred, zero_division=0))
        if f > best_f1:
            best_f1 = f
            best_t  = float(t)
    return best_t


def evaluate_binary(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float | None = None,
    val_loader: torch.utils.data.DataLoader | None = None,
) -> dict:
    """
    Evaluate a binary classifier and compute standard classification metrics.

    Parameters
    ----------
    model
        Trained binary classifier that accepts tabular and text inputs.
    loader
        DataLoader yielding the evaluation split to score.
    device
        Device on which the model and input tensors should be evaluated.
    threshold
        Decision threshold used to convert probabilities into class predictions.
    val_loader
        Optional validation loader used to tune the decision threshold when
        ``threshold`` is not provided.

    Returns
    -------
    metrics
        Dictionary containing the threshold, scalar classification metrics, and
        confusion matrix for the evaluated split.
    """
    probs, y_true = collect_predictions(model, loader, device)

    if threshold is None:
        if val_loader is not None:
            val_probs, val_true = collect_predictions(model, val_loader, device)
            threshold = find_best_threshold(val_probs, val_true)
        else:
            threshold = 0.5

    y_pred    = (probs >= threshold).astype(int)
    y_true_i  = y_true.astype(int)

    out = {
        "threshold":        threshold,
        "accuracy":         float(accuracy_score(y_true_i, y_pred)),
        "precision":        float(precision_score(y_true_i, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true_i, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true_i, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_i, y_pred).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true_i, probs))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def print_metrics(metrics: dict, title: str = "Evaluation") -> None:
    """
    Print a formatted summary of evaluation metrics to standard output.

    Parameters
    ----------
    metrics
        Dictionary of evaluation outputs such as threshold, accuracy, F1, and confusion matrix.
    title
        Heading displayed above the metric summary.

    Returns
    -------
    None
        Writes the formatted metric report to standard output.
    """
    print(f"\n=== {title} ===")
    thr = metrics.get("threshold")
    if thr is not None:
        print(f"  threshold: {thr:.2f}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if k in metrics:
            v = metrics[k]
            print(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        print("  confusion_matrix [[TN, FP], [FN, TP]]:")
        for row in cm:
            print(f"    {row}")
