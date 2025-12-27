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
from sklearn.metrics import accuracy_score, confusion_matrix

"""
Evaluate classification models using grammatical structure features combined with TRL-specific terms.
"""

# ==========================================
# 0. load datasets & define paths
# ==========================================
SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./21_results_dual_eval_grammar"
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

# Building blocks for grammatical structure tagging
FUTURE_PATTERNS = [
    r"\bwill\b", r"\bshall\b", r"\bgoing to\b"
]
MODAL_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bshould\b"
]
PASSIVE_PATTERNS = [
    r"\b(is|are|was|were|been|being)\s+\w+ed\b"
]
PHASE_RESEARCH = [r"fundamental research", r"early-stage research", r"exploratory study"]
PHASE_POC_LAB = [r"proof of concept", r"lab experiment", r"in the laboratory"]
PHASE_RELEVANT = [r"relevant environment", r"high[- ]fidelity prototype"]
PHASE_DEMO = [r"pilot plant", r"field testing", r"demonstration (project|plant)"]
PHASE_DEPLOY = [r"in operation", r"commercial deployment", r"deployed", r"full-scale operation"]

def add_grammar_tags(text: str) -> str:
    """Grammatical structure tagging based on regex patterns."""
    t = text.lower()
    tokens = []

    # Futur tags
    if any(re.search(p, t) for p in FUTURE_PATTERNS):
        tokens.append("__FUTURE__")

    # Modal tags
    if any(re.search(p, t) for p in MODAL_PATTERNS):
        tokens.append("__MODAL__")

    # Passive tags
    if any(re.search(p, t) for p in PASSIVE_PATTERNS):
        tokens.append("__PASSIVE__")

    # Additional tags (objective, demonstration, etc.)
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

def keep_trl_terms_and_structure(text: str) -> str:
    """
    Build 'reduced' text that keeps only:
    - grammatical structure tags
    - words belonging to the TRL lexicon (TRL_LONG)
    """
    if not isinstance(text, str):
        return ""

    t = text.lower()
    # 1) Structure tags
    tags = add_grammar_tags(t)

    # 2) Words from the TRL lexicon present in the text
    words = re.findall(r"\b[a-z]+\b", t)
    trl_words_in_text = [w for w in words if w in TRL_TERMS]

    # 3) Final text = tags + TRL words
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

def evaluate_dual(model, X_train, y_train,
                  X_test_silver, y_test_silver,
                  X_test_gold, y_test_gold,
                  model_name):
    print(f"\n--- Model (Grammar+TRL) : {model_name} ---")
    
    # 1. TRAINING: TF-IDF on preprocessed text
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=50000
        )),
        ('clf', model)
    ])
    
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"   -> Trained in {train_time:.2f}s")

    # 2. INTERNAL EVAL (Silver Test)
    preds_silver = pipeline.predict(X_test_silver)
    diff_silver = np.abs(y_test_silver - preds_silver)
    acc_silver = accuracy_score(y_test_silver, preds_silver)
    rel_silver = np.mean(diff_silver <= 1)

    # 3. EXTERNAL EVAL (Gold Test)
    preds_gold = pipeline.predict(X_test_gold)
    diff_gold = np.abs(y_test_gold - preds_gold)
    acc_gold = accuracy_score(y_test_gold, preds_gold)
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
    
    # Confusion matrices
    plot_cm(y_test_silver, preds_silver,
            f"CM {model_name} (Internal Test, Grammar+TRL)",
            f"{OUTPUT_DIR}/cm_{model_name}_internal_grammar.png")
    plot_cm(y_test_gold, preds_gold,
            f"CM {model_name} (External Gold, Grammar+TRL)",
            f"{OUTPUT_DIR}/cm_{model_name}_external_grammar.png")

# ==========================================
# 1. DATA PREPARATION
# ==========================================
print("Loading datasets")

# A. Silver
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=['text', 'label'])
df_silver['label'] = df_silver['label'].astype(int)

# grammatical structure + TRL terms version
df_silver['text_grammar'] = df_silver['text'].apply(keep_trl_terms_and_structure)

X_train_raw, X_test_silver_raw, y_train, y_test_silver = train_test_split(
    df_silver['text_grammar'],
    df_silver['label'],
    test_size=0.2,
    random_state=42,
    stratify=df_silver['label']
)

# B. Gold (IEA)
df_gold = pd.read_csv(IEA_PATH)
df_gold = df_gold.dropna(subset=['text', 'trl_final'])
df_gold = df_gold[(df_gold['trl_final'] >= 1) & (df_gold['trl_final'] <= 9)]
df_gold['label'] = df_gold['trl_final'].astype(int)
df_gold['text_grammar'] = df_gold['text'].apply(keep_trl_terms_and_structure)

X_test_gold_raw = df_gold['text_grammar']
y_test_gold = df_gold['label'].values

# Class weights information
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
print("\n--- Class weight (balanced) ---")
for c, w in zip(classes, weights):
    print(f"Class {c}: {w:.2f}")

print(f"Train Set (Silver, grammar): {len(X_train_raw)}")
print(f"Internal Test (Silver, grammar): {len(X_test_silver_raw)}")
print(f"External Test (Gold, grammar): {len(X_test_gold_raw)}")

# ==========================================
# 2. EXÉCUTION
# ==========================================
models = [
    ("SVM (Linear)", LinearSVC(class_weight='balanced', C=0.1, dual='auto', random_state=42)),
    ("Logistic Regression", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
    ("Naive Bayes", MultinomialNB(alpha=0.1)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42))
]

for name, clf in models:
    evaluate_dual(clf,
                  X_train_raw, y_train,
                  X_test_silver_raw, y_test_silver,
                  X_test_gold_raw, y_test_gold,
                  name)

# ==========================================
# 3. Results
# ==========================================
print("\n" + "="*80)
print("Results summary (Grammar + TRL terms)")
print("="*80)
df_res = pd.DataFrame(results_table)
print(df_res.to_string(index=False))
df_res.to_csv(f"{OUTPUT_DIR}/dual_metrics_grammar.csv", index=False)