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
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix

"""
Evaluate regression models with few-shot adaptation trained on a silver dataset
and adapted with a few examples from the gold standard (IEA), then tested on the remaining gold standard data.
"""
# ==========================================
# 0. Load datasets & define paths
# ==========================================
SILVER_PATH = "silver_dataset_master.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./31_results_regression_fewshot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Few-Shot Parameters
FEW_SHOT_SIZE = 50
BOOST_FACTOR = 100

results_table = []

def evaluate_regression_adaptation(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n--- Training: {model_name}  ---")
    
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(
            ngram_range=(1, 3), 
            sublinear_tf=True, 
            #stop_words='english', 
            max_features=50000
        )),
        ('reg', model)
    ])
    
    t0 = time.time()
    # Training on continuous labels (float)
    pipeline.fit(X_train, y_train.astype(float))
    train_time = time.time() - t0
    
    # Continuous predictions
    raw_preds = pipeline.predict(X_test)
    
    # Post-processing (Rounding for classification metrics)
    preds = np.round(raw_preds).astype(int)
    preds = np.clip(preds, 1, 9)
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    diff = np.abs(y_test - preds)
    relaxed_acc = np.mean(diff <= 1)
    mae = mean_absolute_error(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    
    results_table.append({
        "Model": model_name,
        "Type": "Regression (Few-Shot)",
        "Relaxed Acc": f"{relaxed_acc:.1%}",
        "Strict Acc": f"{acc:.1%}",
        "MAE": f"{mae:.2f}",
        "F1 Macro": f"{f1:.3f}",
        "Time": f"{train_time:.1f}s"
    })
    
    print(f" {model_name}: Relaxed={relaxed_acc:.1%} | MAE={mae:.2f}")

# ==========================================
# 1. DATA PREPARATION (Few-Shot Setup)
# ==========================================
print("Loading and Merging (Few-Shot Mode)")

# A. Silver
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=['text', 'label'])
df_silver['label'] = df_silver['label'].astype(int)

# B. Gold (IEA)
df_iea = pd.read_csv(IEA_PATH)
def clean_iea(val):
    try:
        s = str(val).strip()
        if '-' in s: return int(float(s.split('-')[0]))
        return int(float(s))
    except: return None
df_iea['label'] = df_iea['trl_final'].apply(clean_iea)
df_iea = df_iea.dropna(subset=['text', 'label'])
df_iea = df_iea[(df_iea['label'] >= 1) & (df_iea['label'] <= 9)]

label_counts = df_iea['label'].value_counts()
rare_labels = label_counts[label_counts < 2].index
if len(rare_labels) > 0:
    print(f" Removing rare classes (<2 examples) to allow stratified split: {list(rare_labels)}")
    df_iea = df_iea[~df_iea['label'].isin(rare_labels)]
# ---------------------------------------------------------

# C. Stratified Split (50 Train / Remainder Test)
iea_train, iea_test = train_test_split(
    df_iea, 
    train_size=FEW_SHOT_SIZE, 
    random_state=42, 
    stratify=df_iea['label']
)

# D. BOOSTING TARGET EXAMPLES
# multiply the 50 examples by 100 so they "weigh" against the 35k
iea_boosted = pd.concat([iea_train.rename(columns={'text': 'text'})[['text', 'label']]] * BOOST_FACTOR)

# Final Fusion
df_train_final = pd.concat([
    df_silver[['text', 'label']], 
    iea_boosted
]).sample(frac=1, random_state=42) # Shuffle
print(f"Train Set Final : {len(df_train_final)} rows (including {len(iea_boosted)} boosted)")
print(f"Test Set (IEA Hidden) : {len(iea_test)} rows")

# ==========================================
# 2. BENCHMARK REGRESSION
# ==========================================
models = [
    # Ridge Regression 
    (Ridge(alpha=1.0, random_state=42), "Ridge Regression"),
    
    # Linear SVR 
    (LinearSVR(C=1.0, dual='auto', random_state=42, max_iter=10000), "Linear SVR")
]

for model, name in models:
    evaluate_regression_adaptation(
        model, 
        df_train_final['text'], df_train_final['label'], 
        iea_test['text'], iea_test['label'].values, 
        name
    )

# ==========================================
# 3. RESULTS
# ==========================================
print("\n" + "="*70)
print("Result summary: REGRESSION WITH ADAPTATION (Few-Shot)")
print("="*70)
df_res = pd.DataFrame(results_table)
print(df_res.to_string(index=False))
df_res.to_csv(f"{OUTPUT_DIR}/metrics_regression_fewshot.csv", index=False)