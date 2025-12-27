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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix

"""
Evaluate regression and classification models trained on a silver dataset
and tested on both an internal silver test set and an external gold standard (IEA).
"""
# ==========================================
# 0. Load datasets & define paths
# ==========================================
SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./30_results_regression_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results_table = []

def evaluate_dual(model, X_train, y_train, X_test_silver, y_test_silver, X_test_gold, y_test_gold, model_name, task_type):
    print(f"\n=== Model: {model_name} ({task_type}) ===")
    
    # 1. Create and Train Pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(
            ngram_range=(1, 3), 
            sublinear_tf=True, 
            #stop_words='english', 
            max_features=50000
        )),
        ('model', model)
    ])
    
    print("  -> Training on Silver train")
    t0 = time.time()
    
    if task_type == "regression":
        y_train_formatted = y_train.astype(float)
    else:
        y_train_formatted = y_train.astype(int)
        
    pipeline.fit(X_train, y_train_formatted)
    train_time = time.time() - t0
    
    
    def process_evaluation(X, y, dataset_name):
        raw_preds = pipeline.predict(X)
        
        
        if task_type == "regression":
            preds = np.round(raw_preds).astype(int)
            preds = np.clip(preds, 1, 9)
        else:
            preds = raw_preds.astype(int)
        
        # Metrics
        acc = accuracy_score(y, preds)
        diff = np.abs(y - preds)
        relaxed_acc = np.mean(diff <= 1)
        mae = mean_absolute_error(y, preds)
        f1 = f1_score(y, preds, average='macro')
        
        results_table.append({
            "Model": model_name,
            "Type": task_type,
            "Test Set": dataset_name,
            "Time (Train)": f"{train_time:.1f}s",
            "Relaxed Acc": f"{relaxed_acc:.1%}",
            "Strict Acc": f"{acc:.1%}",
            "MAE": f"{mae:.2f}",
            "F1 Macro": f"{f1:.3f}"
        })
        
        print(f"  -> {dataset_name}: Relaxed={relaxed_acc:.1%} | MAE={mae:.2f}")
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y, preds, labels=range(1, 10))
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=range(1, 10), yticklabels=range(1, 10))
        plt.title(f"CM: {model_name} on {dataset_name}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        safe_name = model_name.replace(' ', '_')
        safe_set = dataset_name.replace(' ', '_')
        plt.savefig(f"{OUTPUT_DIR}/cm_{safe_name}_{safe_set}.png")
        plt.close()

    # 2. Evaluation Silver Test
    process_evaluation(X_test_silver, y_test_silver, "Silver Test")
    
    # 3. Evaluation Gold Test
    process_evaluation(X_test_gold, y_test_gold, "Gold Test")

# ==========================================
# 1. LOADING DATASETS
# ==========================================
print("Loading datasets")

# A. Silver Dataset (Train + Internal Test)
if not os.path.exists(SILVER_PATH):
    print(f"Error: {SILVER_PATH} not found.")
    exit()
    
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=['text', 'label'])
df_silver['label'] = df_silver['label'].astype(int)

# Split 80% Train / 20% Test
X_train, X_test_silver, y_train, y_test_silver = train_test_split(
    df_silver['text'], 
    df_silver['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df_silver['label'] 
)

# B. Gold Dataset (External Test - IEA)
df_gold = pd.read_csv(IEA_PATH)
def clean_iea(val):
    try:
        s = str(val).strip()
        if '-' in s: return int(float(s.split('-')[0]))
        return int(float(s))
    except: return None
df_gold['label'] = df_gold['trl_final'].apply(clean_iea)
df_gold = df_gold.dropna(subset=['text', 'label'])
df_gold = df_gold[(df_gold['label'] >= 1) & (df_gold['label'] <= 9)]

X_test_gold = df_gold['text']
y_test_gold = df_gold['label'].values

print(f"Silver Train: {len(X_train)}")
print(f"Silver Test : {len(X_test_silver)}")
print(f"Gold Test   : {len(X_test_gold)}")

# ==========================================
# 2. LIST OF MODELS
# ==========================================
models_to_test = [
    # 1. Ridge Regression
    (Ridge(alpha=1.0, random_state=42), "Ridge Regression", "regression"),
    
    # 2. Logistic Regression (Probabilistic Classification version)
    (LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), "Logistic Regression", "classification"),
    
    # 3. Linear SVR (Support Vector Regression)
    (LinearSVR(C=1.0, dual='auto', random_state=42, max_iter=10000), "Linear SVR", "regression")
]

# ==========================================
# 3. Run 
# ==========================================
for model, name, task in models_to_test:
    evaluate_dual(
        model, 
        X_train, y_train, 
        X_test_silver, y_test_silver, 
        X_test_gold, y_test_gold, 
        name, task
    )

# ==========================================
# 4.  RESULTS
# ==========================================
print("\n" + "="*80)
print("Results summary: REGRESSION vs CLASSIFICATION (Silver & Gold)")
print("="*80)
df_res = pd.DataFrame(results_table)
# Sort to have Silver and Gold side by side for each model
df_res = df_res.sort_values(by=['Model', 'Test Set'], ascending=[True, False]) 
print(df_res.to_string(index=False))
df_res.to_csv(f"{OUTPUT_DIR}/metrics_regression_comparison.csv", index=False)
print(f"\nRResults saved in {OUTPUT_DIR}/")