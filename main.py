"""
End-to-end pipeline: load (or synthesize) FHIR → tabular dataset → preprocess → train → evaluate → experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

import eval as eval_mod
import experiments
import fhir_parser
import models
import preprocessing
import train as train_mod


def main() -> None:
    parser = argparse.ArgumentParser(description="EMS transport vs refusal — FHIR → PyTorch")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/mock_fhir",
        help="Directory (or single JSON file) with Synthea-style FHIR resources",
    )
    parser.add_argument(
        "--mock-patients",
        type=int,
        default=120,
        help="If no JSON is found under --data-dir, generate this many mock patients",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-experiments", action="store_true", help="Only train + evaluate main model")
    parser.add_argument("--workdir", type=str, default="artifacts", help="Checkpoints and experiment outputs")
    parser.add_argument(
        "--quiet-train",
        action="store_true",
        help="Do not print per-epoch metrics (plot/csv still saved unless --no-plot)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip saving training loss/accuracy figure")
    args = parser.parse_args()

    data_path = Path(args.data_dir)

    def _data_ready(p: Path) -> bool:
        if p.is_file() and p.suffix.lower() in {".json", ".ndjson"}:
            return True
        if p.is_dir():
            return any(p.glob("**/*.json")) or any(p.glob("**/*.ndjson"))
        return False

    if not _data_ready(data_path):
        print(f"No FHIR JSON/NDJSON under {data_path}; generating mock Synthea-like bundles…")
        data_path.mkdir(parents=True, exist_ok=True)
        fhir_parser.write_mock_fhir_directory(data_path, n_patients=args.mock_patients)

    print("Parsing FHIR into tabular rows (one row per EMS-like encounter)…")
    df = fhir_parser.parse_fhir_to_dataframe(data_path, drop_unlabeled=True, only_ems_like=True)
    if df.empty:
        raise SystemExit(
            "No labeled rows produced. Add Encounter.extension transport labels or use mock data."
        )
    print(f"Dataset: {len(df)} rows, columns: {list(df.columns)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("Preprocessing (impute, scale, one-hot) and building DataLoaders…")
    pr = preprocessing.preprocess_and_loaders(df, feature_group="all", batch_size=args.batch_size)

    print(f"Training MLP (input_dim={pr.input_dim})…")
    if not args.quiet_train:
        print("  (per-epoch train/val loss and accuracy; figure saved to workdir when done)")
    model = models.MLPClassifier(pr.input_dim)
    plot_path = None if args.no_plot else workdir / "training_curves.png"
    history = train_mod.train_with_early_stopping(
        model,
        pr.train_loader,
        pr.val_loader,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=workdir / "best_mlp.pt",
        device=device,
        verbose=not args.quiet_train,
        plot_path=plot_path,
    )
    if not args.no_plot and plot_path is not None:
        print(f"Saved training curves: {plot_path.resolve()}")
    pd.DataFrame(history).to_csv(workdir / "training_history.csv", index_label="epoch")
    print(f"Saved per-epoch metrics: {(workdir / 'training_history.csv').resolve()}")

    metrics = eval_mod.evaluate_binary(model, pr.test_loader, device)
    eval_mod.print_metrics(metrics, title="Primary MLP — test set")

    if args.skip_experiments:
        print("Skipping experiments (--skip-experiments).")
        return

    print("\n--- Feature ablations (MLP) ---")
    ab_df = experiments.run_feature_ablations(df, device, workdir)
    print(ab_df[["feature_group", "accuracy", "f1", "roc_auc"]].to_string(index=False))

    print("\n--- Model comparison (full features) ---")
    cmp_df = experiments.run_model_comparison(df, device, workdir)
    print(cmp_df[["model", "accuracy", "f1", "roc_auc"]].to_string(index=False))

    print("\n--- Noise robustness (MLP trained clean; noisy test vitals) ---")
    noise_df = experiments.run_noise_robustness(df, model, device)
    print(noise_df[["noise_std", "accuracy", "f1", "roc_auc"]].to_string(index=False))

    ab_df.to_csv(workdir / "ablations.csv", index=False)
    cmp_df.to_csv(workdir / "model_compare.csv", index=False)
    noise_df.to_csv(workdir / "noise_robustness.csv", index=False)
    print(f"\nSaved experiment tables under {workdir.resolve()}")


if __name__ == "__main__":
    main()
