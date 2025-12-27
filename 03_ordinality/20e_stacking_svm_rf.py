import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error

"""
Evaluate a stacking ensemble of a two-stage SVM model and a Random Forest model for TRL classification.
The two-stage SVM uses a coarse model to predict TRL bands and fine models for specific TRL levels within those bands.
The Random Forest is trained on full-text TF-IDF features.

Both models are trained on a silver dataset and tested on both an internal silver test set and an external gold standard (IEA).
"""
# ==========================================
# 0. Load datasets & define paths
# ==========================================
SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./20e_stacking_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHAS = [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]  # 0.00, 0.10, ..., 1.00

# ==========================================
# 1. Metric and band definitions
# ==========================================
def trl_to_band(y):
    if y <= 3:
        return 0
    elif y <= 6:
        return 1
    else:
        return 2


BAND_TO_CLASSES = {
    0: [1, 2, 3],
    1: [4, 5, 6],
    2: [7, 8, 9],
}


def relaxed_accuracy(y_true, y_pred, tol=1):
    return np.mean(np.abs(y_true - y_pred) <= tol)


def catastrophic_rate(y_true, y_pred, thr=3):
    return np.mean(np.abs(y_true - y_pred) >= thr)


def _softmax_logits(logits, axis=1):
    """Convert LinearSVC decision_function scores to pseudo-probabilities."""
    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = logits[:, None]
    logits = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=axis, keepdims=True)


def eval_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rel = relaxed_accuracy(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    cat = catastrophic_rate(y_true, y_pred)
    return acc, rel, mae, cat


# ==========================================
# 2. Dataset preparation - Load data
# ==========================================
print("Loading datasets")

# Silver
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=["text", "label"])
df_silver["label"] = df_silver["label"].astype(int)

X_train_text, X_test_silver_text, y_train, y_test_silver = train_test_split(
    df_silver["text"],
    df_silver["label"],
    test_size=0.2,
    random_state=42,
    stratify=df_silver["label"],
)

# Gold (IEA)
df_gold = pd.read_csv(IEA_PATH)
df_gold = df_gold.dropna(subset=["text", "trl_final"])
df_gold = df_gold[(df_gold["trl_final"] >= 1) & (df_gold["trl_final"] <= 9)]
df_gold["label"] = df_gold["trl_final"].astype(int)

X_test_gold_text = df_gold["text"]
y_test_gold = df_gold["label"].values

print(
    f"Train Silver: {len(X_train_text)} | "
    f"Test Silver: {len(X_test_silver_text)} | "
    f"Test Gold: {len(X_test_gold_text)}"
)

# vectorization
vect = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50000,
    sublinear_tf=True,
)
print("Vectorization TF-IDF (fit on Silver train)...")
X_train = vect.fit_transform(X_train_text)
X_test_silver = vect.transform(X_test_silver_text)
X_test_gold = vect.transform(X_test_gold_text)

# ==========================================
# 3. TWO-STAGE SVM (SOFT)
# ==========================================
print("\n=== Training two-stage SVM (soft) ===")

# Coarse model (bands)
y_train_band = np.array([trl_to_band(v) for v in y_train])

svm_coarse = LinearSVC(
    C=0.1,
    class_weight="balanced",
    dual="auto",
    random_state=42,
)

t0 = time.time()
svm_coarse.fit(X_train, y_train_band)
print(f"  -> SVM coarse trained in {time.time() - t0:.2f}s")

# Fine models per band
fine_models = {}
print("\n  Training fine SVMs per band...")
for band, classes in BAND_TO_CLASSES.items():
    mask = np.isin(y_train, classes)
    X_band = X_train[mask]
    y_band = y_train[mask]
    print(f"    Band {band} (classes {classes}) : {X_band.shape[0]} examples")

    clf = LinearSVC(
        C=0.1,
        class_weight="balanced",
        dual="auto",
        random_state=42 + band,
    )
    clf.fit(X_band, y_band)
    fine_models[band] = clf


def predict_two_stage_soft(X):
    """
    Probabilistic routing:
    P(band | x) via svm_coarse, then mixing fine SVMs.
    """
    band_scores = svm_coarse.decision_function(X)      # (n_samples, 3)
    p_band = _softmax_logits(band_scores, axis=1)      # (n_samples, 3)

    n_samples = X.shape[0]
    n_trl_classes = 9
    p_y = np.zeros((n_samples, n_trl_classes), dtype=float)  # P(TRL=k | x)

    for band, classes in BAND_TO_CLASSES.items():
        clf = fine_models[band]
        fine_scores = clf.decision_function(X)         # (n_samples, 3)
        p_fine = _softmax_logits(fine_scores, axis=1)  # P(TRL in classes | x, band)

        weight = p_band[:, band][:, None]              # (n_samples, 1)
        contrib = p_fine * weight                      # (n_samples, 3)

        for j, cls in enumerate(classes):
            p_y[:, cls - 1] += contrib[:, j]           # TRL 1..9 -> index 0..8

    y_pred = p_y.argmax(axis=1) + 1                    # labels 1..9
    return y_pred


# ==========================================
# 4. RANDOM FOREST (FULL-TEXT)
# ==========================================
print("\n=== Training RandomForest (full-text TF-IDF) ===")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
t0 = time.time()
rf.fit(X_train, y_train)
print(f"  -> RF trained in {time.time() - t0:.2f}s")

# ==========================================
# 5. STACKING ON ALPHA GRID
# ==========================================
print("\n=== Stacking Two-stage SVM (soft) + RF ===")
rows = []

for name, X, y_true in [
    ("Silver test", X_test_silver, y_test_silver.values),
    ("Gold IEA", X_test_gold, y_test_gold),
]:
    print(f"\n--- {name} ---")

    # Base predictions
    y_svm = predict_two_stage_soft(X)
    y_rf = rf.predict(X)

    acc_svm, rel_svm, mae_svm, cat_svm = eval_metrics(y_true, y_svm)
    acc_rf, rel_rf, mae_rf, cat_rf = eval_metrics(y_true, y_rf)

    print(
        f"SVM soft only : strict={acc_svm:.3f} | ±1={rel_svm:.3f} "
        f"| MAE={mae_svm:.3f} | cat(≥3)={cat_svm:.3f}"
    )
    print(
        f"RF only       : strict={acc_rf:.3f} | ±1={rel_rf:.3f} "
        f"| MAE={mae_rf:.3f} | cat(≥3)={cat_rf:.3f}"
    )

    dataset_rows = []

    for alpha in ALPHAS:
        # alpha = weight SVM, (1 - alpha) = weight RF
        y_stack_real = alpha * y_svm + (1.0 - alpha) * y_rf
        y_stack = np.rint(y_stack_real).astype(int)
        y_stack = np.clip(y_stack, 1, 9)

        acc, rel, mae, cat = eval_metrics(y_true, y_stack)

        dataset_rows.append({
            "Dataset": name,
            "Alpha_SVM": float(alpha),
            "Strict": acc,
            "Relaxed": rel,
            "MAE": mae,
            "Catastrophic>=3": cat,
        })

    # Display best alpha on MAE and catastrophic rate
    best = min(
        dataset_rows,
        key=lambda r: (r["MAE"], r["Catastrophic>=3"])
    )
    print(
        f"Best alpha (SVM) according to MAE : {best['Alpha_SVM']:.2f} | "
        f"MAE={best['MAE']:.3f} | cat(≥3)={best['Catastrophic>=3']:.3f}"
    )

    rows.extend(dataset_rows)

df_out = pd.DataFrame(rows)
out_path = os.path.join(OUTPUT_DIR, "metrics_stacking_svm_rf.csv")
df_out.to_csv(out_path, index=False)
print(f"\n Stacking results saved to {out_path}")