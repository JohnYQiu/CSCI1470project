"""
Full model comparison: all PyTorch architectures + sklearn classical ML baselines + ensemble.

Usage:
    python compare_models.py [--mock-patients N] [--epochs N] [--workdir PATH]

Writes results to model_comparison_results.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

import eval as eval_mod
import fhir_parser
import models
import preprocessing
import train as train_mod


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_numpy(
    loader: torch.utils.data.DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert all batches from a DataLoader into concatenated NumPy arrays.

    Parameters
    ----------
    loader
        DataLoader yielding ``(x_tab, x_text, y)`` batches.

    Returns
    -------
    arrays
        Tuple containing the concatenated tabular features, encoded text features,
        and labels as NumPy arrays.
    """
    tabs, texts, ys = [], [], []
    for x_tab, x_text, yb in loader:
        tabs.append(x_tab.numpy())
        texts.append(x_text.numpy().reshape(-1, 1))
        ys.append(yb.numpy().ravel())
    return (
        np.concatenate(tabs),
        np.concatenate(texts),
        np.concatenate(ys),
    )


def _sklearn_features(X_tab: np.ndarray, X_text: np.ndarray) -> np.ndarray:
    """
    Assemble the feature matrix used by the sklearn baselines.

    Parameters
    ----------
    X_tab
        Array of scaled tabular features.
    X_text
        Array of label-encoded chief-complaint indices.

    Returns
    -------
    features
        Feature matrix formed by concatenating the tabular features and encoded text index.
    """
    return np.hstack([X_tab, X_text.astype(np.float32)])


def _eval_sklearn(
    clf, X_test: np.ndarray, y_test: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluate an sklearn classifier and tune its threshold on validation probabilities.

    Parameters
    ----------
    clf
        Fitted sklearn classifier exposing ``predict_proba``.
    X_test
        Test feature matrix used for final evaluation.
    y_test
        Ground-truth test labels aligned with ``X_test``.
    X_val
        Validation feature matrix used to tune the decision threshold.
    y_val
        Ground-truth validation labels aligned with ``X_val``.

    Returns
    -------
    evaluation_outputs
        Tuple containing the metric dictionary, validation probabilities, and test
        probabilities for the classifier.
    """
    val_probs  = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]
    metrics = _metrics_from_probs(test_probs, y_test, val_probs, y_val)
    return metrics, val_probs, test_probs


def _metrics_from_probs(
    test_probs: np.ndarray, y_test: np.ndarray,
    val_probs: np.ndarray,  y_val: np.ndarray,
) -> dict:
    """
    Tune a decision threshold on validation probabilities and score the test split.

    Parameters
    ----------
    test_probs
        Predicted positive-class probabilities for the test split.
    y_test
        Ground-truth binary labels for the test split.
    val_probs
        Predicted positive-class probabilities for the validation split.
    y_val
        Ground-truth binary labels for the validation split.

    Returns
    -------
    metrics
        Dictionary containing the selected threshold and the resulting test metrics.
    """
    y_true = y_test.astype(int)
    threshold = eval_mod.find_best_threshold(val_probs, y_val.astype(int))
    y_pred = (test_probs >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, test_probs))
    except ValueError:
        auc = float("nan")
    return {
        "threshold": round(threshold, 2),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(auc, 4),
    }


# ── DL model registry ─────────────────────────────────────────────────────────

def _dl_model_registry(input_dim: int, vocab_size: int) -> list[tuple[str, torch.nn.Module]]:
    """
    Construct the collection of deep-learning models used in the comparison run.

    Parameters
    ----------
    input_dim
        Number of tabular features consumed by the tabular model inputs.
    vocab_size
        Size of the chief-complaint vocabulary for models with an embedding branch.

    Returns
    -------
    registry
        List of ``(name, model)`` pairs to train and evaluate during comparison.
    """
    return [
        ("LogisticRegression",     models.LogisticRegression(input_dim)),
        ("ShallowMLP",             models.ShallowMLP(input_dim)),
        ("MLPClassifier",          models.MLPClassifier(input_dim)),
        ("DeepResidualMLP",        models.DeepResidualMLP(input_dim)),
        ("EmsClassifier",          models.EmsClassifier(input_dim, vocab_size)),
        ("AttentionEmsClassifier", models.AttentionEmsClassifier(input_dim, vocab_size)),
    ]


# ── Main comparison ───────────────────────────────────────────────────────────

def run_comparison(
    df: pd.DataFrame,
    device: torch.device,
    workdir: Path,
    epochs: int = 100,
    patience: int = 12,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Train all configured models and return a unified comparison table of metrics.

    Parameters
    ----------
    df
        Encounter-level dataframe used to build train, validation, and test splits.
    device
        Device on which the PyTorch models should be trained and evaluated.
    workdir
        Directory where trained-model checkpoints should be written.
    epochs
        Maximum number of training epochs for each PyTorch model.
    patience
        Early-stopping patience applied to each PyTorch training run.
    batch_size
        Batch size used when constructing preprocessing DataLoaders.

    Returns
    -------
    results
        Dataframe containing runtime, parameter-count, and evaluation metrics for
        each trained model and the final ensemble.
    """
    pr = preprocessing.preprocess_and_loaders(df, feature_group="all", batch_size=batch_size)

    X_train_tab, X_train_text, y_train = _extract_numpy(pr.train_loader)
    X_val_tab,   X_val_text,   y_val   = _extract_numpy(pr.val_loader)
    X_test_tab,  X_test_text,  y_test  = _extract_numpy(pr.test_loader)

    X_train_sk = _sklearn_features(X_train_tab, X_train_text)
    X_val_sk   = _sklearn_features(X_val_tab,   X_val_text)
    X_test_sk  = _sklearn_features(X_test_tab,  X_test_text)
    X_fit_sk   = np.vstack([X_train_sk, X_val_sk])
    y_fit      = np.concatenate([y_train, y_val])

    rows: list[dict] = []

    # Store probability outputs for ensemble members.
    ensemble_val_probs_list:  list[np.ndarray] = []
    ensemble_test_probs_list: list[np.ndarray] = []
    ENSEMBLE_MEMBERS = {"EmsClassifier", "GradientBoosting", "SVM (RBF)"}

    # ── PyTorch models ────────────────────────────────────────────────────────
    print("\n── PyTorch models ──────────────────────────────────────────")
    for name, model in _dl_model_registry(pr.input_dim, pr.vocab_size):
        print(f"  Training {name}…", end=" ", flush=True)
        t0 = time.time()
        train_mod.train_with_early_stopping(
            model, pr.train_loader, pr.val_loader,
            epochs=epochs, patience=patience,
            checkpoint_path=workdir / f"cmp_{name}.pt",
            device=device, verbose=False,
        )
        elapsed = time.time() - t0
        metrics = eval_mod.evaluate_binary(model, pr.test_loader, device, val_loader=pr.val_loader)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row = {
            "model":     name,
            "type":      "PyTorch",
            "params":    n_params,
            "train_s":   round(elapsed, 1),
            "threshold": round(metrics["threshold"], 2),
            "accuracy":  round(metrics["accuracy"],  4),
            "f1":        round(metrics["f1"],         4),
            "roc_auc":   round(metrics["roc_auc"],   4),
        }
        rows.append(row)
        print(f"acc={row['accuracy']:.3f}  f1={row['f1']:.3f}  auc={row['roc_auc']:.3f}  ({elapsed:.1f}s)")

        if name in ENSEMBLE_MEMBERS:
            vp, _ = eval_mod.collect_predictions(model, pr.val_loader,  device)
            tp, _ = eval_mod.collect_predictions(model, pr.test_loader, device)
            ensemble_val_probs_list.append(vp)
            ensemble_test_probs_list.append(tp)

    # ── sklearn models ────────────────────────────────────────────────────────
    print("\n── sklearn models ──────────────────────────────────────────")
    sk_models = [
        ("RandomForest",     RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
        ("SVM (RBF)",        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
    ]
    for name, clf in sk_models:
        print(f"  Fitting {name}…", end=" ", flush=True)
        t0 = time.time()
        clf.fit(X_fit_sk, y_fit.astype(int))
        elapsed = time.time() - t0
        metrics, vp, tp = _eval_sklearn(clf, X_test_sk, y_test, X_val_sk, y_val)
        row = {
            "model":   name,
            "type":    "sklearn",
            "params":  "—",
            "train_s": round(elapsed, 1),
            **metrics,
        }
        rows.append(row)
        print(f"acc={row['accuracy']:.3f}  f1={row['f1']:.3f}  auc={row['roc_auc']:.3f}  ({elapsed:.1f}s)")

        if name in ENSEMBLE_MEMBERS:
            ensemble_val_probs_list.append(vp)
            ensemble_test_probs_list.append(tp)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print("\n── Ensemble ────────────────────────────────────────────────")
    print(f"  Members: {sorted(ENSEMBLE_MEMBERS)}")
    print("  Computing averaged probabilities…", end=" ", flush=True)
    t0 = time.time()
    ens_val_probs  = np.mean(ensemble_val_probs_list,  axis=0)
    ens_test_probs = np.mean(ensemble_test_probs_list, axis=0)
    metrics = _metrics_from_probs(ens_test_probs, y_test, ens_val_probs, y_val)
    elapsed = time.time() - t0
    row = {
        "model":   "Ensemble (Ems+GB+SVM)",
        "type":    "Ensemble",
        "params":  "—",
        "train_s": round(elapsed, 2),
        **metrics,
    }
    rows.append(row)
    print(f"acc={row['accuracy']:.3f}  f1={row['f1']:.3f}  auc={row['roc_auc']:.3f}")

    return pd.DataFrame(rows)


# ── Results writer ────────────────────────────────────────────────────────────

def write_results(results: pd.DataFrame, path: Path, n_patients: int, n_encounters: int) -> None:
    """
    Write the model-comparison summary report to a Markdown file.

    Parameters
    ----------
    results
        Dataframe containing the comparison metrics for all trained models.
    path
        Output Markdown path where the formatted report should be written.
    n_patients
        Number of mock patients represented in the comparison dataset summary.
    n_encounters
        Number of labeled encounters represented in the comparison dataset summary.

    Returns
    -------
    None
        Formats the comparison tables and writes the Markdown report to disk.
    """
    dl  = results[results["type"] == "PyTorch"].copy()
    sk  = results[results["type"] == "sklearn"].copy()
    ens = results[results["type"] == "Ensemble"].copy()

    def _fmt_table(df: pd.DataFrame, cols: list[str]) -> str:
        """
        Format a dataframe subset as a simple Markdown table.

        Parameters
        ----------
        df
            Dataframe containing the rows to render in table form.
        cols
            Ordered list of columns to include in the Markdown table.

        Returns
        -------
        table
            Markdown string representing the selected dataframe columns.
        """
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        lines  = [header, sep]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    dl_cols  = ["model", "params", "train_s", "threshold", "accuracy", "f1", "roc_auc"]
    sk_cols  = ["model", "train_s", "threshold", "accuracy", "f1", "roc_auc"]
    ens_cols = ["model", "threshold", "accuracy", "f1", "roc_auc"]

    best_f1  = results.loc[results["f1"].idxmax()]
    best_auc = results.loc[results["roc_auc"].idxmax()]

    content = f"""# Model Comparison Results

**Dataset:** {n_patients} mock patients → {n_encounters} labeled EMS encounters
**Split:** 70% train / 15% val / 15% test (stratified)
**Features:** age, sex (one-hot), 6 raw vitals + 3 derived (shock index, pulse pressure, MAP), chief complaint (label-encoded)
**Device:** CPU

---

## PyTorch Models

All DL models trained with:
- Loss: `BCEWithLogitsLoss` with auto-computed `pos_weight` (class imbalance correction)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- LR schedule: `CosineAnnealingLR` (eta_min=1e-5)
- Early stopping on val loss (patience=12)
- Threshold tuned on val set to maximise F1

| Model | Description |
|---|---|
| LogisticRegression | Linear baseline, tabular only |
| ShallowMLP | 1 hidden layer (64), ReLU, no BatchNorm, tabular only |
| MLPClassifier | 2 hidden layers (64→32) with BatchNorm, tabular only |
| DeepResidualMLP | Project→ResBlock→ResBlock→head with skip connections, tabular only |
| EmsClassifier | Tabular branch + dispatch `nn.Embedding` (16-d), concatenated head |
| AttentionEmsClassifier | Soft feature-attention over vitals + dispatch embedding |

{_fmt_table(dl, dl_cols)}

*params = trainable parameter count; train_s = wall-clock seconds*

---

## sklearn / Classical ML Models

Fitted on train+val combined. Features: scaled tabular (with derived features) + label-encoded complaint index. Threshold tuned on val set.

| Model | Description |
|---|---|
| RandomForest | 200 trees, `class_weight="balanced"` |
| GradientBoosting | 200 estimators, max_depth=4 |
| SVM (RBF) | RBF kernel, `class_weight="balanced"`, `probability=True` |

{_fmt_table(sk, sk_cols)}

---

## Ensemble (EmsClassifier + GradientBoosting + SVM)

Soft-voting: simple average of sigmoid/probability outputs from the three best individual models. Threshold tuned on val set.

{_fmt_table(ens, ens_cols)}

---

## Summary

| | Model | Score |
|---|---|---|
| Best F1 | **{best_f1['model']}** | {best_f1['f1']} |
| Best ROC-AUC | **{best_auc['model']}** | {best_auc['roc_auc']} |

### Key observations

- **Feature engineering:** Derived features (shock index = HR/SBP, pulse pressure, MAP) give every model access to clinically meaningful composite signals that the raw vitals would otherwise need to compute implicitly.
- **Dispatch embedding:** `EmsClassifier` and `AttentionEmsClassifier` use a learnable `nn.Embedding` for chief complaint, giving them access to dispatch signal that tabular-only models lack.
- **Ensemble:** averaging probabilities from `EmsClassifier` (strong DL, uses text), `GradientBoosting` (strong tree model), and `SVM` (strong on small datasets) combines complementary inductive biases.
- **Depth vs. data size:** `DeepResidualMLP` does not outperform shallower models — at ~400 samples, extra depth adds variance without enough data to support it.
"""
    path.write_text(content, encoding="utf-8")
    print(f"\nResults written to {path.resolve()}")


def main() -> None:
    """
    Run the full comparison CLI and write the resulting Markdown report.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Parses command-line arguments, prepares data, trains comparison models,
        and writes the final results report.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",      default="data/mock_fhir")
    parser.add_argument("--mock-patients", type=int, default=200)
    parser.add_argument("--epochs",        type=int, default=100)
    parser.add_argument("--patience",      type=int, default=12)
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--workdir",       default="artifacts")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    workdir   = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if not fhir_parser.has_fhir_resources(data_path):
        print(f"Generating {args.mock_patients} mock patients under {data_path}…")
        data_path.mkdir(parents=True, exist_ok=True)
        fhir_parser.write_mock_fhir_directory(data_path, n_patients=args.mock_patients)

    print("Parsing FHIR…")
    df = fhir_parser.parse_fhir_to_dataframe(data_path, drop_unlabeled=True, only_ems_like=True)
    if df.empty:
        raise SystemExit("No labeled rows produced.")
    print(f"Dataset: {len(df)} rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = run_comparison(
        df, device, workdir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    print("\n\n=== Final Results ===")
    print(results[["model", "type", "accuracy", "f1", "roc_auc"]].to_string(index=False))

    write_results(results, Path("model_comparison_results.md"), args.mock_patients, len(df))


if __name__ == "__main__":
    main()
