import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, mean_absolute_error

"""
Evaluate a two-stage SVM model for TRL classification:
1. A coarse SVM classifies texts into three bands: TRL 1-3, 4-6, 7-9.
2. For each band, a fine SVM classifies texts into the specific TRL levels within that band.
The model is trained on a silver dataset and tested on both an internal silver test set and an external gold standard (IEA).

"""

#=========================================
# 0. Load datasets & define paths
#=========================================

SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./20c_two_stage_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. Dataset preparation - Band definition
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

# ==========================================
# 2. Dataset preparation - Load data
# ==========================================
print("Loading datasets")
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=["text", "label"])
df_silver["label"] = df_silver["label"].astype(int)

X_train_text, X_test_silver_text, y_train, y_test_silver = train_test_split(
    df_silver["text"],
    df_silver["label"],
    test_size=0.2,
    random_state=42,
    stratify=df_silver["label"],
)

df_gold = pd.read_csv(IEA_PATH)
df_gold = df_gold.dropna(subset=["text", "trl_final"])
df_gold = df_gold[(df_gold["trl_final"] >= 1) & (df_gold["trl_final"] <= 9)]
df_gold["label"] = df_gold["trl_final"].astype(int)

X_test_gold_text = df_gold["text"]
y_test_gold = df_gold["label"].values

print(f"Train Silver: {len(X_train_text)} | Test Silver: {len(X_test_silver_text)} | Test Gold: {len(X_test_gold_text)}")

# ==========================================
# 3. Vectorization
# ==========================================    
vect = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50000,
    sublinear_tf=True,
)
print("Vectorization TF-IDF (fit on Silver train)")
X_train = vect.fit_transform(X_train_text)
X_test_silver = vect.transform(X_test_silver_text)
X_test_gold = vect.transform(X_test_gold_text)

# ==========================================
# 4. Baseline one-shot SVM
# ==========================================
print("\n=== Baseline : SVM one-shot (1-9) ===")
svm_baseline = LinearSVC(
    C=0.1,
    class_weight="balanced",
    dual="auto",
    random_state=42,
)

t0 = time.time()
svm_baseline.fit(X_train, y_train)
print(f"   -> Entraîné en {time.time() - t0:.2f}s")

for name, X, y in [
    ("Silver test", X_test_silver, y_test_silver.values),
    ("Gold IEA", X_test_gold, y_test_gold),
]:
    y_pred = svm_baseline.predict(X)
    acc = accuracy_score(y, y_pred)
    rel = relaxed_accuracy(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    cat = catastrophic_rate(y, y_pred)
    print(f"[Baseline] {name} : strict={acc:.3f} | ±1={rel:.3f} | MAE={mae:.3f} | cat(≥3)={cat:.3f}")

# ==========================================
# 5. Coarse SVM (bands)
# ==========================================
print("\n=== Step 1: SVM coarse (bands 1-3 / 4-6 / 7-9) ===")
y_train_band = np.array([trl_to_band(v) for v in y_train])

svm_coarse = LinearSVC(
    C=0.1,
    class_weight="balanced",
    dual="auto",
    random_state=42,
)

t0 = time.time()
svm_coarse.fit(X_train, y_train_band)
print(f"   -> Trained in {time.time() - t0:.2f}s")

# ==========================================
# 6. Fine SVMs per band
# ==========================================
print("\n=== Step 2: SVM fine models per band ===")
fine_models = {}
for band, classes in BAND_TO_CLASSES.items():
    mask = np.isin(y_train, classes)
    X_band = X_train[mask]
    y_band = y_train[mask]
    print(f"  Band {band} (classes {classes}) : {X_band.shape[0]} examples")

    clf = LinearSVC(
        C=0.1,
        class_weight="balanced",
        dual="auto",
        random_state=42 + band,
    )
    clf.fit(X_band, y_band)
    fine_models[band] = clf

def _softmax_logits(logits, axis=1):
    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = logits[:, None]
    logits = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=axis, keepdims=True)

def predict_two_stage_soft(X):
    # P(band | x)
    band_scores = svm_coarse.decision_function(X)              # (n_samples, 3)
    p_band = _softmax_logits(band_scores, axis=1)              # (n_samples, 3)

    n_samples = X.shape[0]
    n_trl_classes = 9
    p_y = np.zeros((n_samples, n_trl_classes), dtype=float)    # P(TRL=k | x)

    # For each band, P(TRL | x, band) then weighting by P(band | x)
    for band, classes in BAND_TO_CLASSES.items():
        clf = fine_models[band]
        fine_scores = clf.decision_function(X)                 # (n_samples, 3)
        p_fine = _softmax_logits(fine_scores, axis=1)          # P(TRL in classes | x, band)

        weight = p_band[:, band][:, None]                      # (n_samples, 1)
        contrib = p_fine * weight                              # (n_samples, 3)

        for j, cls in enumerate(classes):
            p_y[:, cls - 1] += contrib[:, j]                   # TRL 1..9 -> index 0..8

    y_pred = p_y.argmax(axis=1) + 1                            # labels 1..9
    return y_pred

def predict_two_stage(X):
    band_pred = svm_coarse.predict(X)
    y_final = np.zeros(len(band_pred), dtype=int)

    for band, classes in BAND_TO_CLASSES.items():
        idx = np.where(band_pred == band)[0]
        if len(idx) == 0:
            continue
        clf = fine_models[band]
        y_final[idx] = clf.predict(X[idx])

    return y_final

print("\n=== Evaluation two-stage vs baseline ===")
rows = []
for name, X, y in [
    ("Silver test", X_test_silver, y_test_silver.values),
    ("Gold IEA", X_test_gold, y_test_gold),
]:
    # baseline
    y_baseline = svm_baseline.predict(X)
    acc_b = accuracy_score(y, y_baseline)
    rel_b = relaxed_accuracy(y, y_baseline)
    mae_b = mean_absolute_error(y, y_baseline)
    cat_b = catastrophic_rate(y, y_baseline)

    # two-stage (hard)
    y_two = predict_two_stage(X)
    acc_t = accuracy_score(y, y_two)
    rel_t = relaxed_accuracy(y, y_two)
    mae_t = mean_absolute_error(y, y_two)
    cat_t = catastrophic_rate(y, y_two)

    # two-stage (soft / probabilistic)
    y_soft = predict_two_stage_soft(X)
    acc_s = accuracy_score(y, y_soft)
    rel_s = relaxed_accuracy(y, y_soft)
    mae_s = mean_absolute_error(y, y_soft)
    cat_s = catastrophic_rate(y, y_soft)

    print(f"\n--- {name} ---")
    print(f"Baseline      : strict={acc_b:.3f} | ±1={rel_b:.3f} | MAE={mae_b:.3f} | cat(≥3)={cat_b:.3f}")
    print(f"Two-stage dur : strict={acc_t:.3f} | ±1={rel_t:.3f} | MAE={mae_t:.3f} | cat(≥3)={cat_t:.3f}")
    print(f"Two-stage soft: strict={acc_s:.3f} | ±1={rel_s:.3f} | MAE={mae_s:.3f} | cat(≥3)={cat_s:.3f}")

    rows.append({
        "Dataset": name,
        "Model": "Baseline SVM",
        "Strict": acc_b,
        "Relaxed": rel_b,
        "MAE": mae_b,
        "Catastrophic>=3": cat_b,
    })
    rows.append({
        "Dataset": name,
        "Model": "Two-stage SVM (hard)",
        "Strict": acc_t,
        "Relaxed": rel_t,
        "MAE": mae_t,
        "Catastrophic>=3": cat_t,
    })
    rows.append({
        "Dataset": name,
        "Model": "Two-stage SVM (soft)",
        "Strict": acc_s,
        "Relaxed": rel_s,
        "MAE": mae_s,
        "Catastrophic>=3": cat_s,
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(os.path.join(OUTPUT_DIR, "metrics_two_stage_vs_baseline.csv"), index=False)
print(f"\n Results saved in {OUTPUT_DIR}/metrics_two_stage_vs_baseline.csv")