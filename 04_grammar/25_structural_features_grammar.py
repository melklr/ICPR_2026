import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

"""
Evaluate classification models using structural grammatical features and TRL-related lexicon.
R1, R2, R3, R4 are evaluated on both an internal silver test set and an external gold standard (IEA).
"""
# ==========================================
# 0. Datasets & paths
# ==========================================
SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./25_results_structural_grammar"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results_table = []

# ==========================================
# 1. TRL LEXICON + GRAMMATICAL PATTERNS
# ==========================================
TRL_LONG = {
    1: "basic principles observed fundamental research theoretical studies lowest maturity scientific discovery idea generation",
    2: "technology concept formulated hypothesis speculative application potential use cases theoretical framework no validation",
    3: "experimental proof of concept critical function validation feasibility study lab experiment analytical studies early prototype",
    4: "technology validation in lab component testing prototype in controlled environment laboratory verification low-fidelity",
    5: "technology validation in relevant environment high-fidelity prototype simulation industrial validation risk reduction",
    6: "system demonstration in relevant environment pilot functional prototype near-operational performance testing",
    7: "system demonstration in operational environment field testing real world conditions deployment pre-commercial",
    8: "system complete and qualified certification standards production readiness commercial ready final system",
    9: "system proven in operational environment market deployment full scale commercialized sales mission ready"
}

TRL_TERMS = sorted(set(
    w
    for desc in TRL_LONG.values()
    for w in desc.lower().split()
))

FUTURE_PATTERNS = [r"\bwill\b", r"\bshall\b", r"\bgoing to\b"]
MODAL_PATTERNS = [r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bshould\b"]
PASSIVE_PATTERNS = [r"\b(is|are|was|were|been|being)\s+\w+ed\b"]

PHASE_RESEARCH = [r"fundamental research", r"early[- ]stage research", r"exploratory (study|research)"]
PHASE_POC_LAB = [r"proof of concept", r"lab(oratory)? experiment", r"in the laboratory"]
PHASE_RELEVANT = [r"relevant environment", r"high[- ]fidelity prototype"]
PHASE_DEMO = [r"pilot (plant|project)", r"field testing", r"demonstration (project|plant|in)"]
PHASE_DEPLOY = [r"in operation", r"commercial deployment", r"commercially available", r"full[- ]scale operation", r"deployed"]


def add_grammar_phase_tags(text: str) -> str:
    t = text.lower()
    tokens = []

    if any(re.search(p, t) for p in FUTURE_PATTERNS):
        tokens.append("__FUTURE__")
    if any(re.search(p, t) for p in MODAL_PATTERNS):
        tokens.append("__MODAL__")
    if any(re.search(p, t) for p in PASSIVE_PATTERNS):
        tokens.append("__PASSIVE__")

    if any(re.search(p, t) for p in PHASE_RESEARCH):
        tokens.append("__PHASE_RESEARCH__")
    if any(re.search(p, t) for p in PHASE_POC_LAB):
        tokens.append("__PHASE_POC_LAB__")
    if any(re.search(p, t) for p in PHASE_RELEVANT):
        tokens.append("__PHASE_RELEVANT__")
    if any(re.search(p, t) for p in PHASE_DEMO):
        tokens.append("__PHASE_DEMO__")
    if any(re.search(p, t) for p in PHASE_DEPLOY):
        tokens.append("__PHASE_DEPLOY__")

    return " ".join(tokens)


def text_grammar_view(text: str) -> str:
    """R2 : tags grammaticaux + mots du lexique TRL."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    tags = add_grammar_phase_tags(t)
    words = re.findall(r"\b[a-z]+\b", t)
    trl_words = [w for w in words if w in TRL_TERMS]
    return (tags + " " + " ".join(trl_words)).strip()


def extract_structural_features(text: str) -> dict:
    """R3 : vecteur numérique de features structuraux + phases TRL."""
    if not isinstance(text, str):
        t = ""
    else:
        t = text.lower()

    n_tokens = len(re.findall(r"\b\w+\b", t))

    def count(patts):
        return sum(1 for p in patts if re.search(p, t))

    feats = {
        "n_tokens": n_tokens,
        "n_future": count(FUTURE_PATTERNS),
        "n_modal": count(MODAL_PATTERNS),
        "n_passive": count(PASSIVE_PATTERNS),
        "has_phase_research": 1 if any(re.search(p, t) for p in PHASE_RESEARCH) else 0,
        "has_phase_poc_lab": 1 if any(re.search(p, t) for p in PHASE_POC_LAB) else 0,
        "has_phase_relevant": 1 if any(re.search(p, t) for p in PHASE_RELEVANT) else 0,
        "has_phase_demo": 1 if any(re.search(p, t) for p in PHASE_DEMO) else 0,
        "has_phase_deploy": 1 if any(re.search(p, t) for p in PHASE_DEPLOY) else 0,
    }
    return feats


def plot_cm(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=range(1, 10))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=False, cmap="Blues",
                xticklabels=range(1, 10), yticklabels=range(1, 10))
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def evaluate_representation(rep_name, model_name, make_pipeline,
                            X_train, y_train,
                            X_silver, y_silver,
                            X_gold, y_gold):
    print(f"\n=== {rep_name} | {model_name} ===")

    pipe = make_pipeline()
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  -> Train in {train_time:.2f}s")

    def eval_split(X, y, split_name):
        preds = pipe.predict(X)
        diff = np.abs(y - preds)
        acc = accuracy_score(y, preds)
        relaxed = np.mean(diff <= 1)
        mae = mean_absolute_error(y, preds)
        catastrophic = np.mean(diff >= 3)

        print(f"  [{split_name}] Strict={acc:.3f} | ±1={relaxed:.3f} | MAE={mae:.2f} | Catastrophic(>=3)={catastrophic:.3f}")

        # Confusion matrix 
        if split_name == "Gold":
            fname = f"{OUTPUT_DIR}/cm_{rep_name}_{model_name}_gold.png".replace(" ", "_")
            plot_cm(y, preds, f"{rep_name} - {model_name} (Gold)", fname)

        return acc, relaxed, mae, catastrophic

    acc_s, rel_s, mae_s, cat_s = eval_split(X_silver, y_silver, "Silver")
    acc_g, rel_g, mae_g, cat_g = eval_split(X_gold, y_gold, "Gold")

    results_table.append({
        "Representation": rep_name,
        "Model": model_name,
        "Train_Time_s": round(train_time, 2),
        "Silver_Strict": acc_s,
        "Silver_Relaxed": rel_s,
        "Silver_MAE": mae_s,
        "Silver_Catastrophic>=3": cat_s,
        "Gold_Strict": acc_g,
        "Gold_Relaxed": rel_g,
        "Gold_MAE": mae_g,
        "Gold_Catastrophic>=3": cat_g
    })


# ==========================================
# 2. LOADING + VIEWS R1–R4
# ==========================================
print("Loading datasets")

# Silver
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=["text", "label"])
df_silver["label"] = df_silver["label"].astype(int)

X_text = df_silver["text"].values
y = df_silver["label"].values

X_train_text, X_test_silver_text, y_train, y_test_silver = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

train_idx = df_silver.sample(frac=1, random_state=42).index[:len(X_train_text)]
df_silver = df_silver.reset_index(drop=True)
X_text = df_silver["text"].values
y = df_silver["label"].values

X_train_text, X_test_silver_text, y_train, y_test_silver, idx_train, idx_test_silver = train_test_split(
    X_text, y, np.arange(len(df_silver)),
    test_size=0.2, random_state=42, stratify=y
)

# Gold
df_gold = pd.read_csv(IEA_PATH)
df_gold = df_gold.dropna(subset=["text", "trl_final"])
df_gold = df_gold[(df_gold["trl_final"] >= 1) & (df_gold["trl_final"] <= 9)]
df_gold["label"] = df_gold["trl_final"].astype(int)
X_test_gold_text = df_gold["text"].values
y_test_gold = df_gold["label"].values

# R1 : full-text
X_train_R1 = X_train_text
X_silver_R1 = X_test_silver_text
X_gold_R1 = X_test_gold_text

# R2 : tags grammar + TRL_TERMS
df_silver["text_grammar"] = df_silver["text"].apply(text_grammar_view)
df_gold["text_grammar"] = df_gold["text"].apply(text_grammar_view)

X_train_R2 = df_silver.loc[idx_train, "text_grammar"].values
X_silver_R2 = df_silver.loc[idx_test_silver, "text_grammar"].values
X_gold_R2 = df_gold["text_grammar"].values

# R3 : structural_features
struct_silver = df_silver["text"].apply(extract_structural_features)
df_struct_silver = pd.DataFrame(list(struct_silver))

struct_gold = df_gold["text"].apply(extract_structural_features)
df_struct_gold = pd.DataFrame(list(struct_gold))

struct_cols = df_struct_silver.columns.tolist()

X_train_R3 = df_struct_silver.loc[idx_train, struct_cols]
X_silver_R3 = df_struct_silver.loc[idx_test_silver, struct_cols]
X_gold_R3 = df_struct_gold[struct_cols]

# R4 : fusion full-text + structural_features
df_silver_fusion = pd.concat([df_silver[["text"]], df_struct_silver], axis=1)
df_gold_fusion = pd.concat([df_gold[["text"]], df_struct_gold], axis=1)

X_train_R4 = df_silver_fusion.loc[idx_train]
X_silver_R4 = df_silver_fusion.loc[idx_test_silver]
X_gold_R4 = df_gold_fusion


# ==========================================
# 3. Pipelines R1–R4
# ==========================================
def make_pipeline_R1(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            # stop_words="english",
            max_features=50000
        )),
        ("clf", model)
    ])


def make_pipeline_R2(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=50000
        )),
        ("clf", model)
    ])


def make_pipeline_R3(model):
    return Pipeline([
        ("clf", model)
    ])


def make_pipeline_R4(model):
    transformer = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                max_features=50000
            ), "text"),
            ("struct", "passthrough", struct_cols),
        ],
        remainder="drop"
    )
    return Pipeline([
        ("features", transformer),
        ("clf", model)
    ])


models = [
    ("SVM", LinearSVC(class_weight="balanced", C=0.1, dual="auto", random_state=42)),
    ("LogReg", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ("RF", RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)),
]

# ==========================================
# 4. R1–R4 Experiments
# ==========================================
for model_name, clf in models:
    # R1
    evaluate_representation(
        "R1_fulltext", model_name,
        lambda clf=clf: make_pipeline_R1(clf),
        X_train_R1, y_train,
        X_silver_R1, y_test_silver,
        X_gold_R1, y_test_gold
    )

    # R2
    evaluate_representation(
        "R2_grammar_TFIDF", model_name,
        lambda clf=clf: make_pipeline_R2(clf),
        X_train_R2, y_train,
        X_silver_R2, y_test_silver,
        X_gold_R2, y_test_gold
    )

    # R3
    evaluate_representation(
        "R3_struct_features", model_name,
        lambda clf=clf: make_pipeline_R3(clf),
        X_train_R3, y_train,
        X_silver_R3, y_test_silver,
        X_gold_R3, y_test_gold
    )

    # R4
    evaluate_representation(
        "R4_full_plus_struct", model_name,
        lambda clf=clf: make_pipeline_R4(clf),
        X_train_R4, y_train,
        X_silver_R4, y_test_silver,
        X_gold_R4, y_test_gold
    )


# ==========================================
# 4bis. save prediction SVM R1/R2 on IEA
# ==========================================
from sklearn.base import clone

print("\n--- Saving SVM R1/R2 predictions on IEA ---")
svm_base = LinearSVC(class_weight="balanced", C=0.1, dual="auto", random_state=42)

# R1 : fulltext
svm_R1 = make_pipeline_R1(clone(svm_base))
svm_R1.fit(X_train_R1, y_train)
pred_gold_R1 = svm_R1.predict(X_gold_R1)

df_pred_R1 = pd.DataFrame({
    "text": X_gold_R1,
    "label": y_test_gold,
    "pred_R1": pred_gold_R1
})
df_pred_R1.to_csv(os.path.join(OUTPUT_DIR, "svm_R1_fulltext_preds_iea.csv"), index=False)

# R2 : grammar_TFIDF
svm_R2 = make_pipeline_R2(clone(svm_base))
svm_R2.fit(X_train_R2, y_train)
pred_gold_R2 = svm_R2.predict(X_gold_R2)

df_pred_R2 = pd.DataFrame({
    "text": X_gold_R2,         
    "label": y_test_gold,
    "pred_R2": pred_gold_R2
})
df_pred_R2.to_csv(os.path.join(OUTPUT_DIR, "svm_R2_grammar_preds_iea.csv"), index=False)

print("Saving SVM R1/R2 predictions on IEA in", OUTPUT_DIR)

# ==========================================
# 5. SAVE RESULTS
# ==========================================
df_res = pd.DataFrame(results_table)
df_res.to_csv(os.path.join(OUTPUT_DIR, "metrics_structural_grammar.csv"), index=False)
print("\n Results saved in", os.path.join(OUTPUT_DIR, "metrics_structural_grammar.csv"))
print(df_res.to_string(index=False))