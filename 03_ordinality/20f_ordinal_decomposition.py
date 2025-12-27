import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

"""
Evaluate an ordinal decomposition approach for TRL classification using multiple binary logistic regression models.
For each threshold k=2..9, a binary model estimates P(y >= k).
The final TRL prediction is reconstructed from these probabilities.
"""

#=========================================
# 0. Load datasets & define paths
#=========================================
SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./20f_ordinal_decomposition_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# metrics definitions
def relaxed_accuracy(y_true, y_pred, tol=1):
    return np.mean(np.abs(y_true - y_pred) <= tol)

def catastrophic_rate(y_true, y_pred, thr=3):
    return np.mean(np.abs(y_true - y_pred) >= thr)

#=========================================
# 1. Dataset preparation - Load data
#=========================================
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

vect = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50000,
    sublinear_tf=True,
)
print("Vectorization TF-IDF (fit on Silver train)...")
X_train = vect.fit_transform(X_train_text)
X_test_silver = vect.transform(X_test_silver_text)
X_test_gold = vect.transform(X_test_gold_text)

# =======================
# 8 models P(y >= k), k=2..9
# =======================
print("\n=== Training (binary Logistic Regression) ===")
models = {}
Ks = list(range(2, 10))

t0_global = time.time()
for k in Ks:
    print(f"  -> Model for P(y >= {k})")
    y_bin = (y_train >= k).astype(int)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42 + k,
        n_jobs=-1,
    )
    t0 = time.time()
    clf.fit(X_train, y_bin)
    print(f"     Trained in {time.time() - t0:.2f}s")
    models[k] = clf

print(f"Total training time (8 models): {time.time() - t0_global:.2f}s")

def predict_ordinal_corallike(X):
    """
    Reconstruct y^ from the 8 models P(y>=k):
    y^ = 1 + sum_k 1[P(y>=k) >= 0.5], k=2..9.
    """
    n = X.shape[0]
    K = len(Ks)
    P = np.zeros((n, K), dtype=float)

    for j, k in enumerate(Ks):
        clf = models[k]
        # P(y>=k | x) = proba class 1
        proba = clf.predict_proba(X)[:, 1]
        P[:, j] = proba

    decisions = (P >= 0.5).astype(int)   # 1 if y>=k judged true
    counts = decisions.sum(axis=1)       # number of thresholds exceeded
    y_hat = 1 + counts                   # in [1..9]
    return y_hat

def eval_split(name, X, y):
    y_pred = predict_ordinal_corallike(X)
    acc = accuracy_score(y, y_pred)
    rel = relaxed_accuracy(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    cat = catastrophic_rate(y, y_pred)
    print(f"[Ordinal] {name} : strict={acc:.3f} | ±1={rel:.3f} | MAE={mae:.3f} | cat(≥3)={cat:.3f}")
    return acc, rel, mae, cat

print("\n=== Ordinal decomposition evaluation ===")
rows = []
for name, X, y in [
    ("Silver test", X_test_silver, y_test_silver.values),
    ("Gold IEA", X_test_gold, y_test_gold),
]:
    acc, rel, mae, cat = eval_split(name, X, y)
    rows.append({
        "Dataset": name,
        "Model": "Ordinal_LogReg_CORN",
        "Strict": acc,
        "Relaxed": rel,
        "MAE": mae,
        "Catastrophic>=3": cat,
    })

df_out = pd.DataFrame(rows)
out_path = os.path.join(OUTPUT_DIR, "metrics_ordinal_decomposition.csv")
df_out.to_csv(out_path, index=False)
print(f"\n RResults saved in {out_path}")