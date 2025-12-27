import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

"""
Evaluate SVM and Logistic Regression models using character-level n-grams
trained on a silver dataset and tested on both an internal silver test set and an external gold standard (IEA).
"""

# ==========================================
# 0. Load datasets & define paths
# ==========================================

SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./20d_char_ngrams_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define metric: relaxed accuracy (±1)
def relaxed_accuracy(y_true, y_pred, tol=1):
    return np.mean(np.abs(y_true - y_pred) <= tol)

def evaluate_model(model_name, analyzer, ngram_range,
                   X_train_text, y_train,
                   X_test_silver_text, y_test_silver,
                   X_test_gold_text, y_test_gold):
    print(f"\n=== {model_name} | analyzer={analyzer} | ngram={ngram_range} ===")

    vect = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=50000,
        sublinear_tf=True,
    )

    t0 = time.time()
    X_train = vect.fit_transform(X_train_text)
    X_test_silver = vect.transform(X_test_silver_text)
    X_test_gold = vect.transform(X_test_gold_text)
    print(f"   -> TF-IDF fit+transform en {time.time() - t0:.2f}s")

    clf = None
    if model_name.startswith("SVM"):
        C = 0.1
        clf = LinearSVC(
            C=C,
            class_weight="balanced",
            dual="auto",
            random_state=42,
        )
    else:
        C = 0.1
        clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"   -> Modèle entraîné en {train_time:.2f}s")

    rows = []
    for name, X, y in [
        ("Silver test", X_test_silver, y_test_silver),
        ("Gold IEA", X_test_gold, y_test_gold),
    ]:
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        rel = relaxed_accuracy(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        print(f"   [{name}] strict={acc:.3f} | ±1={rel:.3f} | MAE={mae:.3f}")

        rows.append({
            "Model": model_name,
            "Analyzer": analyzer,
            "Ngram": str(ngram_range),
            "Dataset": name,
            "Train Time (s)": round(train_time, 2),
            "Strict": acc,
            "Relaxed": rel,
            "MAE": mae,
        })

    return rows

# ==========================================
# 1. Dataset preparation
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
# 2. Experiments: Char n-grams vs Word n-grams
# ==========================================
results = []

# 1) Word-level baseline (1,3)
for model_name in ["SVM word", "LogReg word"]:
    rows = evaluate_model(
        model_name=model_name,
        analyzer="word",
        ngram_range=(1, 3),
        X_train_text=X_train_text,
        y_train=y_train,
        X_test_silver_text=X_test_silver_text,
        y_test_silver=y_test_silver.values,
        X_test_gold_text=X_test_gold_text,
        y_test_gold=y_test_gold,
    )
    results.extend(rows)

# 2) Char 5-grams (char_wb)
for model_name in ["SVM char5", "LogReg char5"]:
    rows = evaluate_model(
        model_name=model_name,
        analyzer="char_wb",
        ngram_range=(5, 5),
        X_train_text=X_train_text,
        y_train=y_train,
        X_test_silver_text=X_test_silver_text,
        y_test_silver=y_test_silver.values,
        X_test_gold_text=X_test_gold_text,
        y_test_gold=y_test_gold,
    )
    results.extend(rows)

df_res = pd.DataFrame(results)
out_path = os.path.join(OUTPUT_DIR, "char_vs_word_results.csv")
df_res.to_csv(out_path, index=False)
print(f"Results saved in {out_path}")