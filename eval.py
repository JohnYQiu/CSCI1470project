"""
Classification metrics and confusion matrix (numpy / sklearn).
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
    model.eval()
    probs: list[float] = []
    labels: list[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).view(-1)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(yb.view(-1).cpu().numpy().tolist())
    return np.asarray(probs), np.asarray(labels)


def evaluate_binary(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    probs, y_true = collect_predictions(model, loader, device)
    y_pred = (probs >= threshold).astype(int)
    y_true_i = y_true.astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true_i, y_pred)),
        "precision": float(precision_score(y_true_i, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_i, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_i, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_i, y_pred).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true_i, probs))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def print_metrics(metrics: dict, title: str = "Evaluation") -> None:
    print(f"\n=== {title} ===")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if k in metrics:
            v = metrics[k]
            print(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        print("  confusion_matrix [[TN, FP], [FN, TP]]:")
        for row in cm:
            print(f"    {row}")
