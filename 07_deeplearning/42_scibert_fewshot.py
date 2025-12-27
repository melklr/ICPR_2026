import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os
import re

"""
Fine-tune a SciBERT model on a few-shot subset of the IEA dataset,
starting from a pre-trained SciBERT model (trained on Silver).
Evaluate the adapted model on the remaining IEA test set.
"""

# ==========================================
# 1. clean_iea_trl function
# ==========================================
def clean_iea_trl(val):
    """Nettoie les valeurs TRL (ex: '5', '5.0', '5-8' -> 5)"""
    try:
        s = str(val).strip()
        match = re.match(r"(\d+)", s)
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None

# ==========================================
# configurations
# ==========================================

MODEL_PATH = "./scibert_final_model/final" 
IEA_PATH = "C:\\Users\\Melusine\\.venv\\IEA_Clean_Guide_Final_with_Text.csv" 

# Few-Shot Parameters
FEW_SHOT_SIZE = 50   
EPOCHS = 10          
BATCH_SIZE = 8
LEARNING_RATE = 2e-5 # Soft rate 
if __name__ == "__main__":
    # ==========================================
    # 1. Data preparation (IEA)
    # ==========================================
    print("--- Loading IEA dataset ---")
    df_iea = pd.read_csv(IEA_PATH)

    df_iea['label_raw'] = df_iea['trl_final'].apply(clean_iea_trl)
    df_iea = df_iea.dropna(subset=['text', 'label_raw'])
    df_iea = df_iea[(df_iea['label_raw'] >= 1) & (df_iea['label_raw'] <= 9)]
    df_iea['label'] = df_iea['label_raw'] - 1 # 0 à 8
    df_iea = df_iea.rename(columns={'text': 'text'})

    # Filter out rare classes for stratify
    label_counts = df_iea['label'].value_counts()
    rare_labels = label_counts[label_counts < 2].index
    if len(rare_labels) > 0:
        print(f"Removing rare classes (<2 examples) to allow splitting: {list(rare_labels)}")
        df_iea = df_iea[~df_iea['label'].isin(rare_labels)]
    #

    # SPLIT FEW-SHOT
    train_df, test_df = train_test_split(
        df_iea, 
        train_size=FEW_SHOT_SIZE, 
        random_state=42, 
        stratify=df_iea['label'] 
    )

    print(f"Size Train : {len(train_df)}")
    print(f"Size Test  : {len(test_df)}")

    # ==========================================
    # 2. Loading model (Transfer Learning)
    # ==========================================
    print(f"Loading model from: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print(" Local model not found. Loading base SciBERT (Baseline).")
        MODEL_PATH = "allenai/scibert_scivocab_uncased"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=9)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    # ==========================================
    # 3. FINE-TUNING FEW-SHOT
    # ==========================================
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Relaxed accuracy (+/- 1)
        true_trl = labels + 1
        pred_trl = predictions + 1
        relaxed_acc = np.mean(np.abs(true_trl - pred_trl) <= 1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "relaxed_accuracy": relaxed_acc,
            "f1_macro": f1_score(labels, predictions, average='macro')
        }

    training_args = TrainingArguments(
        output_dir="./scibert_fewshot_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="relaxed_accuracy",
        save_total_limit=1,
        logging_steps=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("\n Starting Few-Shot")
    trainer.train()

    # ==========================================
    # 4. FINAL RESULT
    # ==========================================
    print("\n RESULTS :")
    metrics = trainer.evaluate()
    print(metrics)
