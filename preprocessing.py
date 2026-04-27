"""
Tabular preprocessing: imputation, scaling, encodings, and PyTorch DataLoaders.

Three encoding streams:
  numeric  → SimpleImputer(median) + StandardScaler         → float32 tensor
  categorical (sex) → SimpleImputer(mode) + OneHotEncoder   → float32 tensor (appended to numeric)
  text (chief_complaint) → LabelEncoder                     → int64 tensor   (separate branch)
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

FeatureGroup = Literal["all", "vitals_only", "demographics_only"]

VITAL_COLS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
    "temperature",
]
# Clinically derived features computed from raw vitals before scaling.
DERIVED_COLS = [
    "shock_index",    # HR / systolic_BP  — higher → more haemodynamic compromise
    "pulse_pressure", # systolic - diastolic — reflects stroke volume
    "map_bp",         # mean arterial pressure = diastolic + pulse_pressure/3
]
DEMO_COLS = ["age", "sex"]
NUMERIC_COLS = ["age", *VITAL_COLS, *DERIVED_COLS]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived clinical features to a copy of the encounter dataframe.

    Parameters
    ----------
    df
        Encounter-level dataframe containing the raw vital-sign columns used to derive
        shock index, pulse pressure, and mean arterial pressure.

    Returns
    -------
    engineered_df
        Copy of the input dataframe with the derived clinical feature columns added.
    """
    df = df.copy()
    hr  = df.get("heart_rate")
    sbp = df.get("systolic_bp")
    dbp = df.get("diastolic_bp")

    pp = sbp - dbp if sbp is not None and dbp is not None else None
    df["shock_index"]    = (hr / sbp.replace(0, np.nan)) if hr is not None and sbp is not None else np.nan
    df["pulse_pressure"] = pp if pp is not None else np.nan
    df["map_bp"]         = (dbp + pp / 3.0)             if pp is not None and dbp is not None else np.nan
    return df


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X_tab: np.ndarray, X_text: np.ndarray, y: np.ndarray):
        """
        Store tabular features, encoded text, and labels as tensors.

        Parameters
        ----------
        X_tab
            Array of tabular features for each sample.
        X_text
            Array of label-encoded chief-complaint indices for each sample.
        y
            Array of binary transport labels for each sample.

        Returns
        -------
        None
            Converts the provided arrays into tensors stored on the dataset instance.
        """
        self.X_tab  = torch.as_tensor(X_tab,  dtype=torch.float32)
        self.X_text = torch.as_tensor(X_text, dtype=torch.long)
        self.y      = torch.as_tensor(y,       dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        """
        Return the number of samples stored in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        length
            Total number of rows available for indexing in the dataset.
        """
        return len(self.X_tab)

    def __getitem__(self, idx: int):
        """
        Return one tabular sample, text token, and label tuple by index.

        Parameters
        ----------
        idx
            Position of the sample to retrieve from the dataset.

        Returns
        -------
        sample
            Tuple containing the tabular tensor, encoded text tensor, and target label
            tensor for the requested sample.
        """
        return self.X_tab[idx], self.X_text[idx], self.y[idx]


@dataclass
class PreprocessResult:
    train_loader:  torch.utils.data.DataLoader
    val_loader:    torch.utils.data.DataLoader
    test_loader:   torch.utils.data.DataLoader
    input_dim:     int       # tabular feature dimension (numeric + one-hot cat)
    vocab_size:    int       # number of unique chief_complaint categories + 1 (for padding/unknown)
    feature_names: list[str]
    preprocessor:  ColumnTransformer
    label_encoder: LabelEncoder


def _make_loader(
    X_tab: np.ndarray,
    X_text: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader from tabular features, encoded text, and labels.

    Parameters
    ----------
    X_tab
        Tabular feature array for one dataset split.
    X_text
        Encoded chief-complaint array for the same split.
    y
        Label array aligned with the provided features.
    batch_size
        Number of samples to include in each batch.
    shuffle
        Whether the resulting DataLoader should shuffle its samples.

    Returns
    -------
    loader
        DataLoader wrapping the provided split as a ``TabularDataset``.
    """
    dataset = TabularDataset(
        np.asarray(X_tab, dtype=np.float32),
        np.asarray(X_text, dtype=np.int64),
        np.asarray(y, dtype=np.float32),
    )
    drop_last = shuffle and len(dataset) > batch_size
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def _select_columns(df: pd.DataFrame, group: FeatureGroup) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select the tabular and text features associated with a requested feature group.

    Parameters
    ----------
    df
        Encounter-level dataframe containing demographics, vitals, and chief complaint.
    group
        Feature subset to construct for downstream preprocessing and modeling.

    Returns
    -------
    selected_features
        Tuple containing the tabular dataframe and chief-complaint series corresponding
        to the requested feature group.
    """
    df = engineer_features(df)
    if group == "all":
        tab = df[DEMO_COLS + VITAL_COLS + DERIVED_COLS].copy()
        txt = df["chief_complaint"].copy()
    elif group == "vitals_only":
        tab = df[VITAL_COLS + DERIVED_COLS].copy()
        txt = pd.Series(["unknown"] * len(df), index=df.index, name="chief_complaint")
    elif group == "demographics_only":
        tab = df[DEMO_COLS].copy()
        txt = df["chief_complaint"].copy()
    else:
        raise ValueError(f"Unknown feature group: {group}")
    return tab, txt


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Build the sklearn preprocessing pipeline for tabular model inputs.

    Parameters
    ----------
    numeric_cols
        Names of numeric columns that should be median-imputed and standardized.
    categorical_cols
        Names of categorical columns that should be mode-imputed and one-hot encoded.

    Returns
    -------
    preprocessor
        Column transformer that applies the configured numeric and categorical pipelines.
    """
    if not numeric_cols and not categorical_cols:
        raise ValueError("At least one numeric or categorical column is required")
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers)


def _encode_text(
    le: LabelEncoder,
    series: pd.Series,
    fit: bool = False,
) -> np.ndarray:
    """
    Label-encode the chief-complaint text column with an explicit unknown token.

    Parameters
    ----------
    le
        Label encoder used to learn or apply the text-category mapping.
    series
        Text values to encode into integer chief-complaint indices.
    fit
        Whether to fit the encoder on the provided series before transforming it.

    Returns
    -------
    encoded_text
        Integer array of chief-complaint indices where unseen categories map to 0.
    """
    filled = series.fillna("unknown").astype(str)
    if fit:
        # Reserve index 0 for unknown/padding by prepending a sentinel class.
        classes = ["<unknown>"] + sorted(filled.unique().tolist())
        le.fit(classes)
    # Map unseen categories to index 0 (<unknown>).
    known = set(le.classes_)
    mapped = filled.map(lambda x: x if x in known else "<unknown>")
    return le.transform(mapped).astype(np.int64)


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
    Split the dataset, fit preprocessing on the training split, and build DataLoaders.

    Parameters
    ----------
    df
        Encounter-level dataframe containing the transport label and raw feature columns.
    feature_group
        Feature subset to use when selecting columns for preprocessing.
    batch_size
        Number of samples to include in each DataLoader batch.
    test_size
        Fraction of examples reserved for the held-out test split.
    val_size
        Fraction of examples reserved for the validation split.
    random_state
        Random seed used for reproducible train, validation, and test splits.
    test_vitals_noise_std
        Standard deviation of Gaussian noise added to selected test-set vital columns.
    test_noise_columns
        Names of test-set columns that should receive Gaussian noise before transformation.

    Returns
    -------
    preprocess_result
        Dataclass containing the fitted preprocessing objects, feature metadata, and
        train, validation, and test DataLoaders.
    """
    if "transport" not in df.columns:
        raise ValueError("DataFrame must include a 'transport' column")

    tab_df, txt_series = _select_columns(df, feature_group)
    y = df["transport"].astype(int).values

    numeric_cols     = [c for c in tab_df.columns if c in NUMERIC_COLS]
    categorical_cols = [c for c in tab_df.columns if c not in numeric_cols]

    # ── Split ──────────────────────────────────────────────────────────────────
    idx = np.arange(len(df))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
        stratify=y_temp,
    )

    X_tab_train = tab_df.iloc[idx_train]
    X_tab_val   = tab_df.iloc[idx_val]
    X_tab_test  = tab_df.iloc[idx_test].copy()

    txt_train = txt_series.iloc[idx_train]
    txt_val   = txt_series.iloc[idx_val]
    txt_test  = txt_series.iloc[idx_test]

    # ── Tabular preprocessing ──────────────────────────────────────────────────
    pre = build_preprocessor(numeric_cols, categorical_cols)
    X_tab_train_t = pre.fit_transform(X_tab_train)
    X_tab_val_t   = pre.transform(X_tab_val)

    if test_vitals_noise_std > 0 and test_noise_columns:
        rng = np.random.default_rng(random_state + 17)
        for col in test_noise_columns:
            if col in X_tab_test.columns:
                noise = rng.normal(0, test_vitals_noise_std, size=len(X_tab_test))
                X_tab_test[col] = X_tab_test[col].astype(float) + noise

    X_tab_test_t = pre.transform(X_tab_test)

    feature_names = list(pre.get_feature_names_out())
    input_dim     = X_tab_train_t.shape[1]

    # ── Text (label) encoding ─────────────────────────────────────────────────
    le = LabelEncoder()
    X_text_train = _encode_text(le, txt_train, fit=True)
    X_text_val   = _encode_text(le, txt_val)
    X_text_test  = _encode_text(le, txt_test)
    vocab_size   = len(le.classes_)  # includes <unknown> at index 0

    train_loader = _make_loader(
        X_tab_train_t,
        X_text_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = _make_loader(
        X_tab_val_t,
        X_text_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = _make_loader(
        X_tab_test_t,
        X_text_test,
        y_test,
        batch_size=batch_size,
        shuffle=False,
    )

    return PreprocessResult(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        vocab_size=vocab_size,
        feature_names=feature_names,
        preprocessor=pre,
        label_encoder=le,
    )
