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
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

"""
Evaluate SVM and Logistic Regression models with various hyperparameter configurations
trained on a silver dataset and tested on both an internal silver test set and an external gold standard (IEA).

"""

# ==========================================
# 0. Load datasets & define paths
# ==========================================
SILVER_PATH = "silver_dataset_master.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./20b_sweep_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

def plot_cm(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=range(1, 10))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(
        cm_norm, annot=False, cmap='Blues',
        xticklabels=range(1, 10), yticklabels=range(1, 10)
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_one(model_name, clf, ngram_range, max_features,
                 X_train, y_train, X_test_silver, y_test_silver,
                 X_test_gold, y_test_gold):
    print(f"\n--- {model_name} | ngram={ngram_range} | max_feat={max_features} ---")

    vect = TfidfVectorizer(
        ngram_range=ngram_range,
        sublinear_tf=True,
        max_features=max_features
    )
    pipeline = Pipeline([
        ('vect', vect),
        ('clf', clf)
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"   -> Entraîné en {train_time:.2f}s")

    # Silver
    preds_silver = pipeline.predict(X_test_silver)
    diff_silver = np.abs(y_test_silver - preds_silver)
    acc_silver = accuracy_score(y_test_silver, preds_silver)
    rel_silver = np.mean(diff_silver <= 1)

    # Gold
    preds_gold = pipeline.predict(X_test_gold)
    diff_gold = np.abs(y_test_gold - preds_gold)
    acc_gold = accuracy_score(y_test_gold, preds_gold)
    rel_gold = np.mean(diff_gold <= 1)

    print(f"   Silver strict={acc_silver:.2%} | ±1={rel_silver:.2%}")
    print(f"   Gold   strict={acc_gold:.2%} | ±1={rel_gold:.2%}")

    results.append({
        "Model": model_name,
        "ngram_range": str(ngram_range),
        "max_features": max_features,
        "Train Time (s)": round(train_time, 1),
        "Silver Strict": acc_silver,
        "Silver Relaxed": rel_silver,
        "Gold Strict": acc_gold,
        "Gold Relaxed": rel_gold,
        "Drop Relaxed (pts)": (rel_silver - rel_gold) * 100,
    })

    base_name = f"{model_name.replace(' ', '_')}_n{ngram_range[0]}-{ngram_range[1]}_mf{max_features}"
    plot_cm(
        y_test_silver, preds_silver,
        f"CM Silver {base_name}",
        os.path.join(OUTPUT_DIR, f"cm_silver_{base_name}.png")
    )
    plot_cm(
        y_test_gold, preds_gold,
        f"CM Gold {base_name}",
        os.path.join(OUTPUT_DIR, f"cm_gold_{base_name}.png")
    )

# ==========================================
# 1. Dataset preparation
# ==========================================
print("Loading datasets")

df_silver = pd.read_csv(SILVER_PATH).dropna(subset=['text', 'label'])
df_silver['label'] = df_silver['label'].astype(int)

X_train, X_test_silver, y_train, y_test_silver = train_test_split(
    df_silver['text'], df_silver['label'],
    test_size=0.2,
    random_state=42,
    stratify=df_silver['label']
)

df_gold = pd.read_csv(IEA_PATH)
df_gold = df_gold.dropna(subset=['text', 'trl_final'])
df_gold = df_gold[(df_gold['trl_final'] >= 1) & (df_gold['trl_final'] <= 9)]
df_gold['label'] = df_gold['trl_final'].astype(int)

X_test_gold = df_gold['text']
y_test_gold = df_gold['label'].values

print(f"Train Silver: {len(X_train)} | Test Silver: {len(X_test_silver)} | Test Gold: {len(X_test_gold)}")

# ==========================================
# 2. HYPERPARAMETER GRID
# ==========================================
svm_C_values = [0.1, 1.0, 2.0]
logreg_C_values = [0.1, 1.0, 5.0]
ngram_configs = [(1, 2), (1, 3)]
max_features_list = [30000, 50000]

# ==========================================
# 3. Optimization loop
# ==========================================
for C in svm_C_values:
    for ngram in ngram_configs:
        for mf in max_features_list:
            clf = LinearSVC(
                class_weight='balanced',
                C=C,
                dual='auto',
                random_state=42
            )
            name = f"SVM C={C}"
            evaluate_one(
                name, clf, ngram, mf,
                X_train, y_train,
                X_test_silver, y_test_silver,
                X_test_gold, y_test_gold
            )

for C in logreg_C_values:
    for ngram in ngram_configs:
        for mf in max_features_list:
            clf = LogisticRegression(
                class_weight='balanced',
                C=C,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            name = f"LogReg C={C}"
            evaluate_one(
                name, clf, ngram, mf,
                X_train, y_train,
                X_test_silver, y_test_silver,
                X_test_gold, y_test_gold
            )

# ==========================================
# 4. Results saving
# ==========================================
df_res = pd.DataFrame(results)
df_res.sort_values(
    by=["Gold Relaxed", "Silver Relaxed"],
    ascending=[False, False],
    inplace=True
)
df_res.to_csv(os.path.join(OUTPUT_DIR, "sweep_svm_logreg.csv"), index=False)
print("\n=== MEILLEURES CONFIGS (triées par Gold Relaxed) ===")
print(df_res.head(10).to_string(index=False))

# ==========================================
# 5. SCATTER Silver Relaxed vs Gold Relaxed
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=df_res,
    x="Silver Relaxed",
    y="Gold Relaxed",
    hue="Model",
    style="Model"
)
plt.xlabel("Silver Relaxed accuracy (±1)")
plt.ylabel("Gold Relaxed accuracy (±1)")
plt.title("Domain shift : Silver vs IEA (SVM / LogReg TF-IDF)")
plt.xlim(0.4, 0.7)   
plt.ylim(0.3, 0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_silver_vs_gold_relaxed.png"), dpi=300)
plt.close()
print("Figure saved: scatter_silver_vs_gold_relaxed.png")