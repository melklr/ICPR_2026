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
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error

"""
Evaluate classification models with few-shot adaptation trained on a silver dataset
and adapted with a few examples from the gold standard (IEA), then tested on the remaining gold standard data.
"""
# ==========================================
# 0. Load datasets & define paths
# ==========================================
SILVER_PATH = "C:\\Users\\Melusine\\.venv\\silver_dataset_master.csv"
IEA_PATH = "C:\\Users\\Melusine\\.venv\\IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./21_results_fewshot_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Few-Shot Parameters
FEW_SHOT_SIZE = 50      # Number of Gold examples
BOOST_FACTOR = 100      

results_table = []

def evaluate_adaptation(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n--- Training: {model_name}  ---")
    
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(
            ngram_range=(1, 3), 
            sublinear_tf=True, 
            #stop_words='english', 
            max_features=50000
        )),
        ('clf', model)
    ])
    
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Predictions on the TEST SET 
    preds = pipeline.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    diff = np.abs(y_test - preds)
    relaxed_acc = np.mean(diff <= 1)
    f1 = f1_score(y_test, preds, average='macro')
    mae = mean_absolute_error(y_test, preds)
    
    results_table.append({
        "Model": model_name,
        "Relaxed Acc": f"{relaxed_acc:.1%}",
        "Strict Acc": f"{acc:.1%}",
        "F1 Macro": f"{f1:.3f}",
        "MAE": f"{mae:.2f}",
        "Time": f"{train_time:.1f}s"
    })
    
    print(f" {model_name}: Relaxed={relaxed_acc:.1%} | Probable gain thanks to Few-Shot")

# ==========================================
# 1. DATA PREPARATION 
# ==========================================
print("Loading and merging...")

# A. Silver
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=['text', 'label'])
df_silver['label'] = df_silver['label'].astype(int)

# B. Gold
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

print(f"Silver Base: {len(df_silver)}")
print(f"Gold Boost (Train): {len(iea_train)} x {BOOST_FACTOR} copies")
print(f"Gold Test (Hidden): {len(iea_test)}")

# D. CREATION OF THE HYBRID DATASET (BOOSTED)
iea_boosted = pd.concat([iea_train.rename(columns={'text': 'text'})[['text', 'label']]] * BOOST_FACTOR)

# Concatenate 
df_train_final = pd.concat([
    df_silver[['text', 'label']], 
    iea_boosted
]).sample(frac=1, random_state=42) # Shuffle

print(f"Final Training Dataset: {len(df_train_final)} rows")

# ==========================================
# 2. BENCHMARK MODELS
# ==========================================
models = [
    ("SVM (Linear)", LinearSVC(class_weight='balanced', C=0.1, dual='auto', random_state=42)),
    ("Naive Bayes", MultinomialNB(alpha=0.1)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)),
    ("Logistic Reg.", LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000))
]

for name, clf in models:
    evaluate_adaptation(
        clf, 
        df_train_final['text'], df_train_final['label'], 
        iea_test['text'], iea_test['label'].values, 
        name
    )

# ==========================================
# 3. RESULTS
# ==========================================
print("\n" + "="*60)
print("Result summary: FEW-SHOT ADAPTATION (50 EXAMPLES)")
print("="*60)
df_res = pd.DataFrame(results_table)
print(df_res.to_string(index=False))
df_res.to_csv(f"{OUTPUT_DIR}/metrics_fewshot.csv", index=False)