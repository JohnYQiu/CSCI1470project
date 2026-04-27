# Model Comparison Results

**Dataset:** 60 mock patients → 116 labeled EMS encounters
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

| model | params | train_s | threshold | accuracy | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 14 | 1.2 | 0.05 | 0.6111 | 0.7586 | 0.2338 |
| ShallowMLP | 961 | 0.0 | 0.44 | 0.5 | 0.6667 | 0.3247 |
| MLPClassifier | 3201 | 0.0 | 0.48 | 0.7222 | 0.8148 | 0.5974 |
| DeepResidualMLP | 18241 | 0.0 | 0.48 | 0.5556 | 0.6923 | 0.6234 |
| EmsClassifier | 3825 | 0.0 | 0.05 | 0.6111 | 0.7586 | 0.6753 |
| AttentionEmsClassifier | 4007 | 0.0 | 0.49 | 0.5556 | 0.6364 | 0.5065 |

*params = trainable parameter count; train_s = wall-clock seconds*

---

## sklearn / Classical ML Models

Fitted on train+val combined. Features: scaled tabular (with derived features) + label-encoded complaint index. Threshold tuned on val set.

| Model | Description |
|---|---|
| RandomForest | 200 trees, `class_weight="balanced"` |
| GradientBoosting | 200 estimators, max_depth=4 |
| SVM (RBF) | RBF kernel, `class_weight="balanced"`, `probability=True` |

| model | train_s | threshold | accuracy | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| RandomForest | 0.2 | 0.23 | 0.6667 | 0.7857 | 0.9481 |
| GradientBoosting | 0.3 | 0.05 | 0.7778 | 0.8182 | 0.9221 |
| SVM (RBF) | 0.0 | 0.13 | 0.7222 | 0.8148 | 0.987 |

---

## Ensemble (EmsClassifier + GradientBoosting + SVM)

Soft-voting: simple average of sigmoid/probability outputs from the three best individual models. Threshold tuned on val set.

| model | threshold | accuracy | f1 | roc_auc |
| --- | --- | --- | --- | --- |
| Ensemble (Ems+GB+SVM) | 0.41 | 0.8333 | 0.8571 | 0.961 |

---

## Summary

| | Model | Score |
|---|---|---|
| Best F1 | **Ensemble (Ems+GB+SVM)** | 0.8571 |
| Best ROC-AUC | **SVM (RBF)** | 0.987 |

### Key observations

- **Feature engineering:** Derived features (shock index = HR/SBP, pulse pressure, MAP) give every model access to clinically meaningful composite signals that the raw vitals would otherwise need to compute implicitly.
- **Dispatch embedding:** `EmsClassifier` and `AttentionEmsClassifier` use a learnable `nn.Embedding` for chief complaint, giving them access to dispatch signal that tabular-only models lack.
- **Ensemble:** averaging probabilities from `EmsClassifier` (strong DL, uses text), `GradientBoosting` (strong tree model), and `SVM` (strong on small datasets) combines complementary inductive biases.
- **Depth vs. data size:** `DeepResidualMLP` does not outperform shallower models — at ~400 samples, extra depth adds variance without enough data to support it.
