import pandas as pd
import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

"""
Compare BERT and SciBERT models on a classification task using a silver dataset for training
and an external gold standard (IEA) for testing.
"""
# ==========================================
# 0. Load datasets & define paths
# ==========================================
SILVER_PATH = "silver_dataset_master.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./40_results_bert_vs_scibert"
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("="*40)
if torch.cuda.is_available():
    print(f" GPU DETECTED : {torch.cuda.get_device_name(0)}")
    print(f"   Available VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("NO GPU DETECTED. The code will run on CPU (slower).")
print("="*40)


MODELS_TO_COMPARE = [
    ("BERT (Base)", "bert-base-uncased"),
    ("SciBERT", "allenai/scibert_scivocab_uncased")
]

# Training parameters 
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 256

results_table = []

# ==========================================
# 1. EVALUATION FUNCTION
# ==========================================
def train_and_evaluate(model_name, model_id, ds_train, ds_test, y_true):
    print(f"\n--- Training : {model_name} ({model_id}) ---")
    
    # 1. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=9)
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)
    
    ds_train_tok = ds_train.map(tokenize, batched=True)
    ds_test_tok = ds_test.map(tokenize, batched=True)
    
    # 2. Training
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{model_name.replace(' ', '_')}",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=500
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train_tok
    )
    
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    
    # 3. Prediction
    print("Prediction on the Gold Standard...")
    preds_output = trainer.predict(ds_test_tok)
    y_pred = np.argmax(preds_output.predictions, axis=1) + 1 # Back to 1-9
    
    # 4. Metrics
    acc = accuracy_score(y_true, y_pred)
    diff = np.abs(y_true - y_pred)
    relaxed_acc = np.mean(diff <= 1)
    f1 = f1_score(y_true, y_pred, average='macro')
    mae = mean_absolute_error(y_true, y_pred)
    
    results_table.append({
        "Model": model_name,
        "Relaxed Acc": f"{relaxed_acc:.1%}",
        "Strict Acc": f"{acc:.1%}",
        "F1 Macro": f"{f1:.3f}",
        "MAE": f"{mae:.2f}",
        "Time": f"{train_time:.0f}s"
    })
    
    print(f" {model_name}: Relaxed={relaxed_acc:.1%} | MAE={mae:.2f}")
    
    # 5. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=range(1, 10))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=range(1, 10), yticklabels=range(1, 10))
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cm_{model_name.replace(' ', '_')}.png")
    plt.close()

# ==========================================
# 2. load DATA
# ==========================================
print("Loading datasets")

# Silver (Train)
df_train = pd.read_csv(SILVER_PATH).dropna(subset=['text', 'label'])
df_train['label'] = df_train['label'].astype(int)
# Mapping 0-8 for HuggingFace
ds_train = Dataset.from_pandas(df_train[['text', 'label']]).map(lambda x: {'label': x['label'] - 1})

# Gold (Test)
df_test = pd.read_csv(IEA_PATH)
def clean_iea(val):
    try:
        s = str(val).strip()
        if '-' in s: return int(float(s.split('-')[0]))
        return int(float(s))
    except: return None
df_test['label'] = df_test['trl_final'].apply(clean_iea)
df_test = df_test.dropna(subset=['text', 'label'])
df_test = df_test[(df_test['label'] >= 1) & (df_test['label'] <= 9)]

y_true = df_test['label'].values
ds_test = Dataset.from_pandas(df_test[['text', 'label']].rename(columns={'text': 'text'})).map(lambda x: {'label': x['label'] - 1})

print(f"Train: {len(ds_train)} | Test: {len(ds_test)}")

# ==========================================
# 3. Compare
# ==========================================
for name, model_id in MODELS_TO_COMPARE:
    train_and_evaluate(name, model_id, ds_train, ds_test, y_true)

# ==========================================
# 4. FINAL RESULTS
# ==========================================
print("\n" + "="*60)
print("COMPARISON: BERT vs SCIBERT")
print("="*60)
df_res = pd.DataFrame(results_table)
print(df_res.to_string(index=False))
df_res.to_csv(f"{OUTPUT_DIR}/metrics_comparison.csv", index=False)