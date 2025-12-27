import pandas as pd
import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

"""
Evaluate SVM model trained on the gold standard (IEA) dataset
and tested on the silver dataset (Cordis).

"""

# ==========================================
# 0. Load datasets & define paths
# ==========================================
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
SILVER_PATH = "silver_dataset_master.csv"
OUTPUT_DIR = "./50_results_svm_iea_to_silver"
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_VALUE = 0.1
NGRAM_RANGE = (1, 3)
MAX_FEATURES = 30000
BOOST_FACTOR = 10  

def plot_cm(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=range(1, 10))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap='Blues',
        xticklabels=range(1, 10),
        yticklabels=range(1, 10)
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    print("=== 50_train_svm_on_IEA : train IEA -> test Silver ===")

    # ------------------------------------------
    # 1. LOADING IEA (TRAIN)
    # ------------------------------------------
    if not os.path.exists(IEA_PATH):
        raise FileNotFoundError(f"{IEA_PATH} not found")

    df_iea = pd.read_csv(IEA_PATH)
    df_iea = df_iea.dropna(subset=["text", "trl_final"])
    df_iea = df_iea[(df_iea["trl_final"] >= 1) & (df_iea["trl_final"] <= 9)]
    df_iea["label"] = df_iea["trl_final"].astype(int)

    # BOOST GLOBAL
    df_iea_rare = df_iea[df_iea["label"].isin([1, 2, 3])]
    df_iea_boosted = pd.concat(
        [df_iea] + [df_iea_rare] * (BOOST_FACTOR - 1),
        ignore_index=True
    )

    X_train = df_iea_boosted["text"]
    y_train = df_iea_boosted["label"].values

    print(f"Train IEA (boosted x{BOOST_FACTOR}) : {len(X_train)} documents")

    # ------------------------------------------
    # 2. LOADING SILVER (TEST)
    # ------------------------------------------
    if not os.path.exists(SILVER_PATH):
        raise FileNotFoundError(f"{SILVER_PATH} not found")

    df_silver = pd.read_csv(SILVER_PATH).dropna(subset=["text", "label"])
    df_silver["label"] = df_silver["label"].astype(int)

    X_test = df_silver["text"]
    y_test = df_silver["label"].values

    print(f"Test Silver : {len(X_test)} documents")

    # ------------------------------------------
    # 3. TF-IDF VECTOR + SVM
    # ------------------------------------------
    tfidf = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        max_features=MAX_FEATURES
    )

    print("\nTF-IDF vectorization on IEA")
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    svm = LinearSVC(
        class_weight="balanced",
        C=C_VALUE,
        dual="auto",
        random_state=42
    )

    print(f"Training SVM (C={C_VALUE}, ngram={NGRAM_RANGE}, max_features={MAX_FEATURES})...")
    t0 = time.time()
    svm.fit(X_train_vec, y_train)
    train_time = time.time() - t0
    print(f"   -> Trained in {train_time:.2f}s")
    # ------------------------------------------
    # 4. EVALUATION ON SILVER
    # ------------------------------------------
    print("\nEval on Silver (Cordis)")
    y_pred = svm.predict(X_test_vec)
    diff = np.abs(y_test - y_pred)

    acc_strict = accuracy_score(y_test, y_pred)
    acc_relaxed = np.mean(diff <= 1)

    print(f"   Strict  : {acc_strict:.3f}")
    print(f"   Relaxed : {acc_relaxed:.3f}")

    # Save results
    results = pd.DataFrame([{
        "Model": "SVM IEA->Silver",
        "C": C_VALUE,
        "ngram_range": str(NGRAM_RANGE),
        "max_features": MAX_FEATURES,
        "Train Time (s)": round(train_time, 1),
        "Silver Strict": acc_strict,
        "Silver Relaxed": acc_relaxed,
    }])
    results.to_csv(os.path.join(OUTPUT_DIR, "metrics_svm_iea_to_silver.csv"), index=False)

    # Confusion matrix
    plot_cm(
        y_test, y_pred,
        "CM SVM trained on IEA, tested on Silver",
        os.path.join(OUTPUT_DIR, "cm_svm_iea_to_silver.png")
    )

    print("\nRResults saved in", OUTPUT_DIR)