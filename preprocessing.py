"""
Tabular preprocessing: imputation, scaling, encodings, and PyTorch DataLoaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FeatureGroup = Literal["all", "vitals_only", "demographics_only"]

VITAL_COLS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
    "temperature",
]
DEMO_COLS = ["age", "sex"]
TEXT_COLS = ["chief_complaint"]


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


@dataclass
class PreprocessResult:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    input_dim: int
    feature_names: list[str]
    preprocessor: ColumnTransformer


def _select_columns(df: pd.DataFrame, group: FeatureGroup) -> pd.DataFrame:
    if group == "all":
        return df[DEMO_COLS + VITAL_COLS + TEXT_COLS].copy()
    if group == "vitals_only":
        return df[VITAL_COLS].copy()
    if group == "demographics_only":
        return df[DEMO_COLS + TEXT_COLS].copy()
    raise ValueError(f"Unknown feature group: {group}")


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    if not numeric_cols and not categorical_cols:
        raise ValueError("At least one numeric or categorical column is required")
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers)


def preprocess_and_loaders(
    df: pd.DataFrame,
    *,
    feature_group: FeatureGroup = "all",
    batch_size: int = 32,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    test_vitals_noise_std: float = 0.0,
    test_noise_columns: list[str] | None = None,
) -> PreprocessResult:
    """
    Train/val/test split (stratified), fit preprocessor on train only, return DataLoaders.

    If ``test_vitals_noise_std`` > 0, Gaussian noise is added to raw vital columns in the
    *test split only* before ``transform`` (train/val stay clean). Used to study
    robustness of a model trained on clean data.
    """
    if "transport" not in df.columns:
        raise ValueError("DataFrame must include a 'transport' column")

    sub = _select_columns(df, feature_group)
    y = df["transport"].astype(int).values

    numeric_cols = [c for c in sub.columns if c in VITAL_COLS or c == "age"]
    categorical_cols = [c for c in sub.columns if c not in numeric_cols]

    X_train, X_temp, y_train, y_temp = train_test_split(
        sub, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state, stratify=y_temp
    )

    pre = build_preprocessor(numeric_cols, categorical_cols)
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)

    X_test_eval = X_test.copy()
    if test_vitals_noise_std > 0 and test_noise_columns:
        rng = np.random.default_rng(random_state + 17)
        for col in test_noise_columns:
            if col in X_test_eval.columns:
                noise = rng.normal(0, test_vitals_noise_std, size=len(X_test_eval))
                X_test_eval[col] = X_test_eval[col].astype(float) + noise

    X_test_t = pre.transform(X_test_eval)

    feature_names = list(pre.get_feature_names_out())
    input_dim = X_train_t.shape[1]

    train_ds = TabularDataset(np.asarray(X_train_t, dtype=np.float32), y_train)
    val_ds = TabularDataset(np.asarray(X_val_t, dtype=np.float32), y_val)
    test_ds = TabularDataset(np.asarray(X_test_t, dtype=np.float32), y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return PreprocessResult(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        feature_names=feature_names,
        preprocessor=pre,
    )
