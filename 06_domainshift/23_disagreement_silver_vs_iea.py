import os
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

"""
Analyze disagreements between two SVM models:
1. SVM trained on the silver dataset.
2. SVM trained on a few-shot subset of the IEA dataset.
Both models are evaluated on the IEA test set to identify instances where their predictions differ.
"""

#=========================================
# 0. Load datasets & define paths
#=========================================
SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./23_results_disagreement"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEW_SHOT_SIZE = 50  

print("Loading datasets")

# Silver
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=["text", "label"])
df_silver["label"] = df_silver["label"].astype(int)

# IEA
df_iea = pd.read_csv(IEA_PATH)
df_iea = df_iea.dropna(subset=["text", "trl_final"])
df_iea = df_iea[(df_iea["trl_final"] >= 1) & (df_iea["trl_final"] <= 9)]
df_iea["label"] = df_iea["trl_final"].astype(int)

print(f"Silver total: {len(df_silver)} | IEA total: {len(df_iea)}")

# Split IEA : few-shot train / test
iea_train, iea_test = train_test_split(
    df_iea,
    train_size=FEW_SHOT_SIZE,
    random_state=42,
    stratify=df_iea["label"]
)

print(f"IEA few-shot train: {len(iea_train)} | IEA test: {len(iea_test)}")

def make_svm_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            #stop_words="english",
            max_features=50000
        )),
        ("clf", LinearSVC(
            class_weight="balanced",
            C=0.1,
            dual="auto",
            random_state=42
        ))
    ])

# 1. Model trained only on the silver dataset
print("\n--- Training SVM on Silver ---")
svm_silver = make_svm_pipeline()
t0 = time.time()
svm_silver.fit(df_silver["text"], df_silver["label"])
print(f"   -> Trained in {time.time() - t0:.2f}s")

# 2. Model trained only on IEA few-shot
print("\n--- Training SVM on IEA few-shot ---")
svm_iea = make_svm_pipeline()
t0 = time.time()
svm_iea.fit(iea_train["text"], iea_train["label"])
print(f"   -> Trained in {time.time() - t0:.2f}s")
# 3. Cross predictions on the same IEA test set
print("\n--- Cross predictions on IEA test ---")
X_test = iea_test["text"]
y_true = iea_test["label"].values

pred_silver = svm_silver.predict(X_test)
pred_iea = svm_iea.predict(X_test)

# Individual accuracies
acc_silver_on_iea = accuracy_score(y_true, pred_silver)
acc_iea_on_iea = accuracy_score(y_true, pred_iea)

print(f"Accuracy SVM_silver on IEA test: {acc_silver_on_iea:.4f}")
print(f"Accuracy SVM_iea (few-shot) on IEA test: {acc_iea_on_iea:.4f}")

# 4. Disagreement analysis
disagree_mask = pred_silver != pred_iea
agree_mask = ~disagree_mask

n_total = len(y_true)
n_disagree = disagree_mask.sum()
n_agree = agree_mask.sum()

print(f"\nTotal number of IEA test examples: {n_total}")
print(f"Agreements: {n_agree} ({n_agree / n_total:.1%})")
print(f"Disagreements: {n_disagree} ({n_disagree / n_total:.1%})")

# Detailed save for inspection
df_out = iea_test.copy()
df_out["pred_silver"] = pred_silver
df_out["pred_iea"] = pred_iea
df_out["agree"] = df_out["pred_silver"] == df_out["pred_iea"]

# Highlight disagreements
df_disagree = df_out[~df_out["agree"]].copy()
df_disagree.to_csv(os.path.join(OUTPUT_DIR, "iea_test_disagreements_silver_vs_iea.csv"), index=False)

print(f"\nDisagreements saved in {OUTPUT_DIR}/iea_test_disagreements_silver_vs_iea.csv")
summary = df_disagree.groupby("label")[["pred_silver", "pred_iea"]].agg(["count"])
summary.to_csv(os.path.join(OUTPUT_DIR, "iea_disagreement_summary_by_label.csv"))
print(f"Summary by label saved in {OUTPUT_DIR}/iea_disagreement_summary_by_label.csv")