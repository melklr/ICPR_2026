import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

"""
Compare the prediction errors of two classification models:
- R1: SVM trained on full text TF-IDF features
- R2: SVM trained on grammatical structure features + TRL-specific terms
Both models are evaluated on the external gold standard (IEA).
The comparison includes overall error metrics, error distributions, and analysis by linguistic patterns.

"""
# ==========================================
# 0. Paths & load predictions
# ==========================================
BASE_DIR = "."
PRED_DIR = "./25_results_structural_grammar"
OUTPUT_DIR = "./26_results_full_vs_grammar"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) LOADING PREDICTIONS R1/R2
r1_path = os.path.join(PRED_DIR, "svm_R1_fulltext_preds_iea.csv")
r2_path = os.path.join(PRED_DIR, "svm_R2_grammar_preds_iea.csv")

print("Loading predictions R1 and R2 on IEA")
df_r1 = pd.read_csv(r1_path)
df_r2 = pd.read_csv(r2_path)

df_r1["idx"] = np.arange(len(df_r1))
df_r2["idx"] = np.arange(len(df_r2))
df = df_r1.merge(df_r2[["idx", "pred_R2"]], on="idx", how="inner")

df["label"] = df["label"].astype(int)
df["pred_R1"] = df["pred_R1"].astype(int)
df["pred_R2"] = df["pred_R2"].astype(int)

# 2) ERRORS AND GLOBAL COMPARISON
df["err_R1"] = df["pred_R1"] - df["label"]
df["err_R2"] = df["pred_R2"] - df["label"]

print("\n--- Global summary R1 and R2(IEA) ---")
for name, col in [("R1_fulltext", "err_R1"), ("R2_grammar", "err_R2")]:
    mae = df[col].abs().mean()
    relaxed = (df[col].abs() <= 1).mean()
    catastrophic = (df[col].abs() >= 3).mean()
    print(f"{name}: MAE={mae:.2f} | ±1={relaxed:.3f} | Catastrophic>=3={catastrophic:.3f}")

# 3) ERROR DISTRIBUTION
plt.figure(figsize=(8,4))
sns.kdeplot(df["err_R1"], label="R1 fulltext", shade=True)
sns.kdeplot(df["err_R2"], label="R2 grammar+TRL", shade=True)
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Error (prediction - IEA label)")
plt.title("Error distribution SVM R1 vs R2 (IEA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "error_distribution_R1_vs_R2.png"), dpi=200)
plt.close()

# 4) ANALYSIS BY LINGUISTIC PATTERNS 
PATTERNS = {
    "futur": r"\b(will|shall|going to)\b",
    "modal": r"\b(may|might|could|should)\b",
    "demo": r"\b(demonstration|demonstrated|pilot|field testing)\b",
    "deploy": r"\b(deployed|deployment|commercial|in operation|operational)\b",
    "research": r"\b(research|exploratory|early-stage|laboratory|lab)\b",
}

print("\n--- Bias by linguistic patterns (on raw text R1) ---")
for name, pat in PATTERNS.items():
    df[f"has_{name}"] = df["text"].apply(lambda t: 1 if re.search(pat, str(t).lower()) else 0)

stats = []
for name in PATTERNS.keys():
    sub = df[df[f"has_{name}"] == 1]
    if len(sub) == 0:
        continue
    stats.append({
        "pattern": name,
        "n_examples": len(sub),
        "mean_err_R1": sub["err_R1"].mean(),
        "mean_err_R2": sub["err_R2"].mean()
    })

df_stats = pd.DataFrame(stats)
df_stats.to_csv(os.path.join(OUTPUT_DIR, "pattern_error_stats_R1_vs_R2.csv"), index=False)
print(df_stats)

# 5) VISUALIZATION OF BIAS BY PATTERN
plt.figure(figsize=(8,4))
sns.barplot(data=df_stats, x="pattern", y="mean_err_R1", color="#1f77b4")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Mean error (R1 - label)")
plt.title("Mean bias R1 (fulltext) by pattern (IEA)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bias_R1_by_pattern.png"), dpi=200)
plt.close()

plt.figure(figsize=(8,4))
sns.barplot(data=df_stats, x="pattern", y="mean_err_R2", color="#ff7f0e")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Mean error (R2 - label)")
plt.title("Mean bias R2 (grammar+TRL) by pattern (IEA)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bias_R2_by_pattern.png"), dpi=200)
plt.close()

print("\nVisualizations saved in", OUTPUT_DIR)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("26_results_full_vs_grammar/pattern_error_stats_R1_vs_R2.csv")
# columns: pattern, mean_err_R1, mean_err_R2

df_long = df.melt(
    id_vars="pattern",
    value_vars=["mean_err_R1", "mean_err_R2"],
    var_name="model",
    value_name="mean_error"
)
df_long["model"] = df_long["model"].map({
    "mean_err_R1": "Full-text",
    "mean_err_R2": "Grammar+TRL"
})

plt.figure(figsize=(6, 4))
sns.barplot(
    data=df_long,
    x="pattern", y="mean_error",
    hue="model", palette="Set2"
)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Mean bias (pred - true)")
plt.title("Bias by linguistic pattern – Full vs Grammar")
plt.tight_layout()
plt.savefig("fig_bias_patterns_full_vs_grammar.png", dpi=300)

