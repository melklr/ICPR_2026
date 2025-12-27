import pandas as pd
import numpy as np
import time
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix

"""
Evaluate classification models using a fusion of full text and grammatical structure features.
"""

# ==========================================
# 0. Datasets & paths
# ==========================================
SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./22_results_dual_eval_fusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results_table = []

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

# Grammatical patterns
FUTURE_PATTERNS = [r"\bwill\b", r"\bshall\b", r"\bgoing to\b"]
MODAL_PATTERNS = [r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bshould\b"]
PASSIVE_PATTERNS = [r"\b(is|are|was|were|been|being)\s+\w+ed\b"]

# Phases TRL (macro-tags)
PHASE_RESEARCH = [r"fundamental research", r"early[- ]stage research", r"exploratory (study|research)"]
PHASE_POC_LAB = [r"proof of concept", r"lab(oratory)? experiment", r"in the laboratory"]
PHASE_RELEVANT = [r"relevant environment", r"high[- ]fidelity prototype"]
PHASE_DEMO = [r"pilot (plant|project)", r"field testing", r"demonstration (project|plant|in)"]
PHASE_DEPLOY = [r"in operation", r"commercial deployment", r"commercially available", r"full[- ]scale operation", r"deployed"]

def add_grammar_phase_tags(text: str) -> str:
    t = text.lower()
    tokens = []

    # Future / modal / passive
    if any(re.search(p, t) for p in FUTURE_PATTERNS):
        tokens.append("__FUTURE__")
    if any(re.search(p, t) for p in MODAL_PATTERNS):
        tokens.append("__MODAL__")
    if any(re.search(p, t) for p in PASSIVE_PATTERNS):
        tokens.append("__PASSIVE__")

    # Phases TRL
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
    """ 'structure+TRL': tags + words from the TRL lexicon present in the text."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    tags = add_grammar_phase_tags(t)
    words = re.findall(r"\b[a-z]+\b", t)
    trl_words_in_text = [w for w in words if w in TRL_TERMS]
    return (tags + " " + " ".join(trl_words_in_text)).strip()

def plot_cm(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=range(1, 10))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=False, cmap='Blues',
                xticklabels=range(1, 10), yticklabels=range(1, 10))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def make_fusion_transformer():
    """
    ColumnTransformer with two views:
    - full text 
    - grammar+TRL view 
    """
    full_vect = TfidfVectorizer(
        ngram_range=(1, 3),
        sublinear_tf=True,
        #stop_words='english',
        max_features=50000
    )
    grammar_vect = TfidfVectorizer(
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=20000  # smaller
    )

    transformer = ColumnTransformer(
        transformers=[
            ("full", full_vect, "text_full"),
            ("grammar", grammar_vect, "text_grammar")
        ],
        remainder="drop"
    )
    return transformer

def evaluate_dual(model, df_train, df_test_silver, df_test_gold, model_name):
    print(f"\n--- Model (full+grammar) : {model_name} ---")

    transformer = make_fusion_transformer()

    pipeline = Pipeline([
        ("features", transformer),
        ("clf", model)
    ])

    t0 = time.time()
    pipeline.fit(df_train[["text_full", "text_grammar"]], df_train["label"])
    train_time = time.time() - t0
    print(f"   -> Train in {train_time:.2f}s")

    # Silver
    preds_silver = pipeline.predict(df_test_silver[["text_full", "text_grammar"]])
    y_silver = df_test_silver["label"].values
    diff_silver = np.abs(y_silver - preds_silver)
    acc_silver = accuracy_score(y_silver, preds_silver)
    rel_silver = np.mean(diff_silver <= 1)

    # Gold
    preds_gold = pipeline.predict(df_test_gold[["text_full", "text_grammar"]])
    y_gold = df_test_gold["label"].values
    diff_gold = np.abs(y_gold - preds_gold)
    acc_gold = accuracy_score(y_gold, preds_gold)
    rel_gold = np.mean(diff_gold <= 1)

    print(f"   -> Silver Relaxed: {rel_silver:.1%} | Gold Relaxed: {rel_gold:.1%}")

    results_table.append({
        "Model": model_name,
        "Train Time": f"{train_time:.1f}s",
        "Silver Strict": f"{acc_silver:.1%}",
        "Silver Relaxed": f"{rel_silver:.1%}",
        "Gold Strict": f"{acc_gold:.1%}",
        "Gold Relaxed": f"{rel_gold:.1%}",
        "Drop (Domain Shift)": f"-{(rel_silver - rel_gold)*100:.1f} pts"
    })

    plot_cm(y_silver, preds_silver,
            f"CM {model_name} (Internal Test, Fusion)",
            f"{OUTPUT_DIR}/cm_{model_name}_internal_fusion.png")
    plot_cm(y_gold, preds_gold,
            f"CM {model_name} (External Gold, Fusion)",
            f"{OUTPUT_DIR}/cm_{model_name}_external_fusion.png")

# ==========================================
# 1. Data preparation
# ==========================================
print("Loading datasets")

# Silver
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=["text", "label"])
df_silver["label"] = df_silver["label"].astype(int)
df_silver["text_full"] = df_silver["text"]
df_silver["text_grammar"] = df_silver["text"].apply(text_grammar_view)

train_df, test_silver_df = train_test_split(
    df_silver,
    test_size=0.2,
    random_state=42,
    stratify=df_silver["label"]
)

# Gold
df_gold = pd.read_csv(IEA_PATH)
df_gold = df_gold.dropna(subset=["text", "trl_final"])
df_gold = df_gold[(df_gold["trl_final"] >= 1) & (df_gold["trl_final"] <= 9)]
df_gold["label"] = df_gold["trl_final"].astype(int)
df_gold["text_full"] = df_gold["text"]
df_gold["text_grammar"] = df_gold["text"].apply(text_grammar_view)

test_gold_df = df_gold

classes = np.unique(train_df["label"].values)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["label"].values)
print("\n--- Class weights (balanced) ---")
for c, w in zip(classes, weights):
    print(f"Class {c}: {w:.2f}")

print(f"Train Set (Silver, fusion): {len(train_df)}")
print(f"Internal Test (Silver, fusion): {len(test_silver_df)}")
print(f"External Test (Gold, fusion): {len(test_gold_df)}")

# ==========================================
# 2. Experiments: Fusion full text + grammar+TRL
# ==========================================
models = [
    ("SVM (Linear)", LinearSVC(class_weight="balanced", C=0.1, dual="auto", random_state=42)),
    ("Logistic Regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ("Naive Bayes", MultinomialNB(alpha=0.1)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)),
]

for name, clf in models:
    evaluate_dual(clf, train_df, test_silver_df, test_gold_df, name)

# ==========================================
# 3. Results
# ==========================================
print("\n" + "="*80)
print("Results summary (Fusion full + grammar)")
print("="*80)
df_res = pd.DataFrame(results_table)
print(df_res.to_string(index=False))
df_res.to_csv(f"{OUTPUT_DIR}/dual_metrics_fusion.csv", index=False)