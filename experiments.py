"""
Feature ablations, model comparison, and noise robustness experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

import eval as eval_mod
import models
import preprocessing
import train as train_mod


def run_feature_ablations(df: pd.DataFrame, device: torch.device, workdir: Path) -> pd.DataFrame:
    rows = []
    for group in ("vitals_only", "demographics_only", "all"):
        pr = preprocessing.preprocess_and_loaders(df, feature_group=group)  # type: ignore[arg-type]
        m = models.MLPClassifier(pr.input_dim)
        ckpt = workdir / f"ablation_{group}_mlp.pt"
        train_mod.train_with_early_stopping(
            m,
            pr.train_loader,
            pr.val_loader,
            checkpoint_path=ckpt,
            device=device,
            verbose=False,
        )
        metrics = eval_mod.evaluate_binary(m, pr.test_loader, device)
        rows.append({"experiment": "ablation", "feature_group": group, **metrics})
    return pd.DataFrame(rows)


def run_model_comparison(df: pd.DataFrame, device: torch.device, workdir: Path) -> pd.DataFrame:
    pr = preprocessing.preprocess_and_loaders(df, feature_group="all")
    results = []

    log_model = models.LogisticRegression(pr.input_dim)
    train_mod.train_with_early_stopping(
        log_model,
        pr.train_loader,
        pr.val_loader,
        checkpoint_path=workdir / "compare_logreg.pt",
        device=device,
        verbose=False,
    )
    results.append(
        {
            "experiment": "model_compare",
            "model": "logistic_regression",
            **eval_mod.evaluate_binary(log_model, pr.test_loader, device),
        }
    )

    mlp = models.MLPClassifier(pr.input_dim)
    train_mod.train_with_early_stopping(
        mlp,
        pr.train_loader,
        pr.val_loader,
        checkpoint_path=workdir / "compare_mlp.pt",
        device=device,
        verbose=False,
    )
    results.append(
        {
            "experiment": "model_compare",
            "model": "mlp",
            **eval_mod.evaluate_binary(mlp, pr.test_loader, device),
        }
    )
    return pd.DataFrame(results)


def run_noise_robustness(
    df: pd.DataFrame,
    base_model: torch.nn.Module,
    device: torch.device,
    noise_levels: Sequence[float] = (0.0, 0.05, 0.15, 0.3),
) -> pd.DataFrame:
    """
    Evaluate a fixed trained model on test sets built with increasing Gaussian noise
    added to *raw* vital columns before scaling (train/val untouched per level).
    """
    rows = []
    for std in noise_levels:
        pr = preprocessing.preprocess_and_loaders(
            df,
            feature_group="all",
            test_vitals_noise_std=std,
            test_noise_columns=preprocessing.VITAL_COLS,
        )
        metrics = eval_mod.evaluate_binary(base_model, pr.test_loader, device)
        rows.append({"experiment": "noise", "noise_std": std, **metrics})
    return pd.DataFrame(rows)
