import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

"""
Train and evaluate BERT and SciBERT models on TRL classification using a silver dataset for training
and a few-shot adaptation on the IEA gold standard dataset.
Congregate all deep learning experiments (zero-shot and few-shot, BERT and SciBERT)."""
# ==========================================
# 0.Load datasets and models
# ==========================================
SILVER_PATH = "minimized_silver_dataset.csv"
IEA_PATH = "IEA_Clean_Guide_Final_with_Text.csv"
OUTPUT_DIR = "./43_deep_all_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    ("BERT (Base)", "bert-base-uncased"),
    ("SciBERT", "allenai/scibert_scivocab_uncased"),
]

MAX_LEN = 256
BATCH_SIZE_SILVER = 16
EPOCHS_SILVER = 3
LR_SILVER = 2e-5

BATCH_SIZE_FEWSHOT = 8
EPOCHS_FEWSHOT = 10
LR_FEWSHOT = 2e-5
FEW_SHOT_SIZE = 50

print("=" * 40)
if torch.cuda.is_available():
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Aucun GPU détecté, entraînement sur CPU (lent).")
print("=" * 40)


# ==========================================

def clean_iea_trl(val):
    try:
        s = str(val).strip()
        if "-" in s:
            return int(float(s.split("-")[0]))
        return int(float(s))
    except Exception:
        return None


def make_hf_dataset(df):
    tmp = df[["text", "label"]].copy()
    tmp["label"] = tmp["label"].astype(int) - 1  # 0-8 pour HF
    return Dataset.from_pandas(tmp[["text", "label"]])


def compute_trl_metrics_from_predictions(preds_output, true_trl):
    logits = preds_output.predictions
    preds_0_8 = np.argmax(logits, axis=1)
    pred_trl = preds_0_8 + 1
    true_trl = np.asarray(true_trl)

    diff = np.abs(true_trl - pred_trl)
    strict_acc = np.mean(true_trl == pred_trl)
    relaxed_acc = np.mean(diff <= 1)
    mae = diff.mean()
    cat_rate = np.mean(diff >= 3)
    f1 = f1_score(true_trl, pred_trl, average="macro")

    return strict_acc, relaxed_acc, mae, cat_rate, f1


def train_zero_and_fewshot(
    base_name,
    model_id,
    results,
    silver_train_ds,
    silver_test_ds,
    y_silver_test,
    iea_train_ds,
    iea_test_ds,
    y_iea_test,
):
    print(f"\n===== Model : {base_name} ({model_id}) =====")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=9)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    print("Tokenizing datasets")
    silver_train_tok = silver_train_ds.map(tokenize, batched=True)
    silver_test_tok = silver_test_ds.map(tokenize, batched=True)
    iea_train_tok = iea_train_ds.map(tokenize, batched=True)
    iea_test_tok = iea_test_ds.map(tokenize, batched=True)

    # ==========================
    # 1) Training on Silver (zero-shot)
    # ==========================
    print("\n--- Training on Silver (zero-shot) ---")
    args_zero = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"{base_name.replace(' ', '_')}_zero"),
        learning_rate=LR_SILVER,
        per_device_train_batch_size=BATCH_SIZE_SILVER,
        num_train_epochs=EPOCHS_SILVER,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=500,
    )

    trainer_zero = Trainer(
        model=model,
        args=args_zero,
        train_dataset=silver_train_tok,
    )

    t0 = time.time()
    trainer_zero.train()
    train_time_zero = time.time() - t0
    print(f"Training time on Silver (zero-shot) : {train_time_zero:.1f}s")

    # Evaluation zero-shot on Silver test
    print("Evaluation zero-shot on Silver test...")
    preds_silver_zero = trainer_zero.predict(silver_test_tok)
    acc_s, rel_s, mae_s, cat_s, f1_s = compute_trl_metrics_from_predictions(
        preds_silver_zero, y_silver_test
    )
    results.append(
        {
            "BaseModel": base_name,
            "Scenario": "zero-shot",
            "Dataset": "Silver test",
            "StrictAcc": acc_s,
            "RelaxedAcc": rel_s,
            "MAE": mae_s,
            "Cat>=3": cat_s,
            "F1Macro": f1_s,
            "TrainTime_s": train_time_zero,
        }
    )

    # Evaluation zero-shot on IEA test
    print("Evaluation zero-shot on IEA test...")
    preds_iea_zero = trainer_zero.predict(iea_test_tok)
    acc_g, rel_g, mae_g, cat_g, f1_g = compute_trl_metrics_from_predictions(
        preds_iea_zero, y_iea_test
    )
    results.append(
        {
            "BaseModel": base_name,
            "Scenario": "zero-shot",
            "Dataset": "Gold IEA test",
            "StrictAcc": acc_g,
            "RelaxedAcc": rel_g,
            "MAE": mae_g,
            "Cat>=3": cat_g,
            "F1Macro": f1_g,
            "TrainTime_s": train_time_zero,
        }
    )

    # ==========================
    # 2) Few-shot on IEA (from Silver model)
    # ==========================
    print("\n--- Few-shot adaptation (50 IEA) ---")
    args_fs = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"{base_name.replace(' ', '_')}_fewshot"),
        learning_rate=LR_FEWSHOT,
        per_device_train_batch_size=BATCH_SIZE_FEWSHOT,
        num_train_epochs=EPOCHS_FEWSHOT,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=20,
    )

    trainer_fs = Trainer(
        model=model,  
        args=args_fs,
        train_dataset=iea_train_tok,
    )

    t0 = time.time()
    trainer_fs.train()
    train_time_fs = time.time() - t0
    print(f"Training time on few-shot (IEA) : {train_time_fs:.1f}s")

    # Evaluation few-shot on Silver test
    print("Evaluation few-shot on Silver test...")
    preds_silver_fs = trainer_fs.predict(silver_test_tok)
    acc_s2, rel_s2, mae_s2, cat_s2, f1_s2 = compute_trl_metrics_from_predictions(
        preds_silver_fs, y_silver_test
    )
    results.append(
        {
            "BaseModel": base_name,
            "Scenario": "few-shot",
            "Dataset": "Silver test",
            "StrictAcc": acc_s2,
            "RelaxedAcc": rel_s2,
            "MAE": mae_s2,
            "Cat>=3": cat_s2,
            "F1Macro": f1_s2,
            "TrainTime_s": train_time_fs,
        }
    )

    # Evaluation few-shot on IEA test
    print("Evaluation few-shot on IEA test...")
    preds_iea_fs = trainer_fs.predict(iea_test_tok)
    acc_g2, rel_g2, mae_g2, cat_g2, f1_g2 = compute_trl_metrics_from_predictions(
        preds_iea_fs, y_iea_test
    )
    results.append(
        {
            "BaseModel": base_name,
            "Scenario": "few-shot",
            "Dataset": "Gold IEA test",
            "StrictAcc": acc_g2,
            "RelaxedAcc": rel_g2,
            "MAE": mae_g2,
            "Cat>=3": cat_g2,
            "F1Macro": f1_g2,
            "TrainTime_s": train_time_fs,
        }
    )


# ==========================================
# 2. LOADING DATA
# ==========================================
print("Loading Silver")
df_silver = pd.read_csv(SILVER_PATH).dropna(subset=["text", "label"])
df_silver["label"] = df_silver["label"].astype(int)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    df_silver["text"],
    df_silver["label"],
    test_size=0.2,
    random_state=42,
    stratify=df_silver["label"],
)

silver_train_df = pd.DataFrame({"text": X_train_s, "label": y_train_s})
silver_test_df = pd.DataFrame({"text": X_test_s, "label": y_test_s})

print(f"Silver train: {len(silver_train_df)} | Silver test: {len(silver_test_df)}")

print("\nLoading IEA Gold")
df_iea = pd.read_csv(IEA_PATH)
df_iea["label"] = df_iea["trl_final"].apply(clean_iea_trl)
df_iea = df_iea.dropna(subset=["text", "label"])
df_iea = df_iea[(df_iea["label"] >= 1) & (df_iea["label"] <= 9)]
df_iea["label"] = df_iea["label"].astype(int)

label_counts = df_iea["label"].value_counts()
rare_labels = label_counts[label_counts < 2].index
if len(rare_labels) > 0:
    print(f"Removing rare classes for few-shot split: {list(rare_labels)}")
    df_iea = df_iea[~df_iea["label"].isin(rare_labels)]

iea_train_df, iea_test_df = train_test_split(
    df_iea,
    train_size=FEW_SHOT_SIZE,
    random_state=42,
    stratify=df_iea["label"],
)

print(f"IEA few-shot train: {len(iea_train_df)} | IEA test: {len(iea_test_df)}")

y_silver_test_trl = silver_test_df["label"].values
y_iea_test_trl = iea_test_df["label"].values

silver_train_ds = make_hf_dataset(silver_train_df)
silver_test_ds = make_hf_dataset(silver_test_df)
iea_train_ds = make_hf_dataset(iea_train_df[["text", "label"]])
iea_test_ds = make_hf_dataset(iea_test_df[["text", "label"]])

# ==========================================
# 3. LOOP 
# ==========================================
all_results = []

for base_name, model_id in MODELS:
    train_zero_and_fewshot(
        base_name,
        model_id,
        all_results,
        silver_train_ds,
        silver_test_ds,
        y_silver_test_trl,
        iea_train_ds,
        iea_test_ds,
        y_iea_test_trl,
    )

# ==========================================
# 4. REesults Summary
# ==========================================
df_res = pd.DataFrame(all_results)
print("\n=== Final results (BERT, SciBERT, zero-shot et few-shot) ===")
print(df_res.to_string(index=False))
df_res.to_csv(os.path.join(OUTPUT_DIR, "metrics_deep_all_scenarios.csv"), index=False)
print(f"\nFinal results save to {os.path.join(OUTPUT_DIR, 'metrics_deep_all_scenarios.csv')}")