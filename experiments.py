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


def _train_and_evaluate(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_path: Path,
) -> dict:
    """
    Train one model and return its test-set evaluation metrics.

    Parameters
    ----------
    model
        Model instance to train and evaluate.
    train_loader
        DataLoader supplying the training batches.
    val_loader
        DataLoader supplying the validation batches for early stopping and threshold tuning.
    test_loader
        DataLoader supplying the held-out test batches.
    device
        Device on which the model should be trained and evaluated.
    checkpoint_path
        Path where the best checkpoint should be written.

    Returns
    -------
    metrics
        Evaluation metrics computed on the held-out test split.
    """
    train_mod.train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        checkpoint_path=checkpoint_path,
        device=device,
        verbose=False,
    )
    return eval_mod.evaluate_binary(model, test_loader, device, val_loader=val_loader)


def run_feature_ablations(df: pd.DataFrame, device: torch.device, workdir: Path) -> pd.DataFrame:
    """
    Train the main EMS classifier on several feature subsets and compare outcomes.

    Parameters
    ----------
    df
        Encounter-level dataframe used to build the ablation datasets.
    device
        Device on which the ablation models should be trained and evaluated.
    workdir
        Directory where ablation checkpoints should be written.

    Returns
    -------
    results
        Dataframe containing evaluation metrics for each feature-group ablation.
    """
    feature_groups: tuple[preprocessing.FeatureGroup, ...] = (
        "vitals_only",
        "demographics_only",
        "all",
    )
    rows = []
    for group in feature_groups:
        pr = preprocessing.preprocess_and_loaders(df, feature_group=group)
        model = models.EmsClassifier(pr.input_dim, pr.vocab_size)
        metrics = _train_and_evaluate(
            model,
            pr.train_loader,
            pr.val_loader,
            pr.test_loader,
            device,
            workdir / f"ablation_{group}_ems.pt",
        )
        rows.append({"experiment": "ablation", "feature_group": group, **metrics})
    return pd.DataFrame(rows)


def run_model_comparison(df: pd.DataFrame, device: torch.device, workdir: Path) -> pd.DataFrame:
    """
    Train several baseline architectures and compare their held-out performance.

    Parameters
    ----------
    df
        Encounter-level dataframe used to build the comparison dataset.
    device
        Device on which the comparison models should be trained and evaluated.
    workdir
        Directory where model-comparison checkpoints should be written.

    Returns
    -------
    results
        Dataframe containing evaluation metrics for each compared model.
    """
    pr      = preprocessing.preprocess_and_loaders(df, feature_group="all")
    results = []

    model_configs = (
        (
            "logistic_regression",
            workdir / "compare_logreg.pt",
            lambda: models.LogisticRegression(pr.input_dim),
        ),
        (
            "mlp",
            workdir / "compare_mlp.pt",
            lambda: models.MLPClassifier(pr.input_dim),
        ),
        (
            "ems_classifier",
            workdir / "compare_ems.pt",
            lambda: models.EmsClassifier(pr.input_dim, pr.vocab_size),
        ),
    )

    for model_name, checkpoint_path, model_factory in model_configs:
        metrics = _train_and_evaluate(
            model_factory(),
            pr.train_loader,
            pr.val_loader,
            pr.test_loader,
            device,
            checkpoint_path,
        )
        results.append({
            "experiment": "model_compare",
            "model": model_name,
            **metrics,
        })

    return pd.DataFrame(results)


def run_noise_robustness(
    df: pd.DataFrame,
    base_model: torch.nn.Module,
    device: torch.device,
    noise_levels: Sequence[float] = (0.0, 0.05, 0.15, 0.3),
) -> pd.DataFrame:
    """
    Evaluate a trained model on test splits with increasing amounts of vital-sign noise.

    Parameters
    ----------
    df
        Encounter-level dataframe used to regenerate noisy preprocessing splits.
    base_model
        Previously trained model that will be evaluated on each noisy test split.
    device
        Device on which the model and test batches should be evaluated.
    noise_levels
        Sequence of Gaussian standard deviations to add to the selected raw test vitals.

    Returns
    -------
    results
        Dataframe containing evaluation metrics for each tested noise level.
    """
    rows = []
    for std in noise_levels:
        pr = preprocessing.preprocess_and_loaders(
            df,
            feature_group="all",
            test_vitals_noise_std=std,
            test_noise_columns=preprocessing.VITAL_COLS,
        )
        metrics = eval_mod.evaluate_binary(base_model, pr.test_loader, device,
                                           val_loader=pr.val_loader)
        rows.append({"experiment": "noise", "noise_std": std, **metrics})
    return pd.DataFrame(rows)
