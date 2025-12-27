import pandas as pd
import numpy as np
import time
import os
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error

"""
Evaluate classification models trained on a silver dataset
and tested on both an internal silver test set and an external gold standard (IEA).
"""
# ==========================================
# 0. Load datasets & define paths
# ==========================================
SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./20_results_dual_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results_table = []

def plot_cm(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=range(1, 10))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=range(1, 10), yticklabels=range(1, 10))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_dual(model, X_train, y_train, X_test_silver, y_test_silver, X_test_gold, y_test_gold, model_name):
    print(f"\n--- Modèle : {model_name} ---")
    
    # 1. training
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=50000)),
        ('clf', model)
    ])
    
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"   -> Entraîné en {train_time:.2f}s")

    # 2. Internal test on Silver set
    preds_silver = pipeline.predict(X_test_silver)
    diff_silver = np.abs(y_test_silver - preds_silver)
    acc_silver = accuracy_score(y_test_silver, preds_silver)
    rel_silver = np.mean(diff_silver <= 1)
    
    # 3. External test on Gold set
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
    
    # Sauvegarde des matrices pour analyse visuelle
    plot_cm(y_test_silver, preds_silver, f"CM {model_name} (Internal Test)", f"{OUTPUT_DIR}/cm_{model_name}_internal.png")
    plot_cm(y_test_gold, preds_gold, f"CM {model_name} (External Gold)", f"{OUTPUT_DIR}/cm_{model_name}_external.png")

# ==========================================
# 1. DATA PREPARATION
# ==========================================
print("Loading datasets...")

# A. Dataset Silver (to split)
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=['text', 'label'])
df_silver['label'] = df_silver['label'].astype(int)

# Split 80% Train / 20% Internal Test
X_train, X_test_silver, y_train, y_test_silver = train_test_split(
    df_silver['text'], df_silver['label'], test_size=0.2, random_state=42, stratify=df_silver['label']
)

# B. Dataset Gold (IEA) - External Test
df_gold = pd.read_csv(IEA_PATH)
df_gold = df_gold.dropna(subset=['text', 'trl_final'])
df_gold = df_gold[(df_gold['trl_final'] >= 1) & (df_gold['trl_final'] <= 9)]
df_gold['label'] = df_gold['trl_final'].astype(int)

X_test_gold = df_gold['text']
y_test_gold = df_gold['label'].values

# Calculate and display class weights for information
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
print("\n--- Class Weights (balanced) ---")
for c, w in zip(classes, weights):
    print(f"Class {c}: {w:.2f}")

print(f"Train Set (Silver): {len(X_train)}")
print(f"Internal Test (Silver): {len(X_test_silver)}")
print(f"External Test (Gold): {len(X_test_gold)}")

# ==========================================
# 2. MODEL TRAINING & EVALUATION
# ==========================================
models = [
    ("SVM (Linear)", LinearSVC(class_weight='balanced', C=0.1, dual='auto', random_state=42)),
    ("Logistic Regression", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
    ("Naive Bayes", MultinomialNB(alpha=0.1)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42))
]

for name, clf in models:
    evaluate_dual(clf, X_train, y_train, X_test_silver, y_test_silver, X_test_gold, y_test_gold, name)

# ==========================================
# 3. RESULTS
# ==========================================
print("\n" + "="*80)
print("Results Summary:")
print("="*80)
df_res = pd.DataFrame(results_table)
print(df_res.to_string(index=False))
df_res.to_csv(f"{OUTPUT_DIR}/dual_metrics.csv", index=False)