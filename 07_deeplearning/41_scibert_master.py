import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Train a SciBERT model on a silver dataset to classify TRL levels, and evaluate it on the gold standard (IEA).
Also generate visualizations for analysis.
"""

# ==========================================
# 1. DEFINE PATHS & PARAMETERS
# ==========================================
SILVER_PATH = "C:\\Users\\Melusine\\.venv\\silver_dataset_master.csv"
IEA_PATH = "C:\\Users\\Melusine\\.venv\\IEA_Clean_Guide_Final_with_Text.csv" # Pour le test final
MODEL_NAME = "allenai/scibert_scivocab_uncased"
OUTPUT_DIR = "./41_scibert_final_model"

# Parameters
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3 
# ==========================================
# 2. PREPARE DATA (Train)
# ==========================================
print("Loading Silver Dataset")
df = pd.read_csv(SILVER_PATH)
df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

print("Balancing classes...")
df_balanced = df.groupby('label').apply(lambda x: x.sample(min(len(x), 2000), random_state=42)).reset_index(drop=True)
print(df_balanced['label'].value_counts().sort_index())

# Mapping for BERT (0 to 8)
df_balanced['label_id'] = df_balanced['label'] - 1 

# Split Train/Val
train_df, val_df = train_test_split(df_balanced, test_size=0.1, random_state=42, stratify=df_balanced['label'])

# ==========================================
# 3. Prepare gold standard (Test IEA)
# ==========================================
print("\nLoading Gold Standard (IEA)...")
df_iea = pd.read_csv(IEA_PATH)

def clean_iea_trl(val):
    try:
        s = str(val).strip()
        if '-' in s: return int(float(s.split('-')[0]))
        return int(float(s))
    except: return None

df_iea['label'] = df_iea['trl_final'].apply(clean_iea_trl)
df_iea = df_iea.dropna(subset=['text', 'label'])
df_iea = df_iea[(df_iea['label'] >= 1) & (df_iea['label'] <= 9)]
df_iea['label_id'] = df_iea['label'] - 1
df_iea = df_iea.rename(columns={'text': 'text'})

print(f"Gold Standard IEA : {len(df_iea)} examples.")

# ==========================================
# 4. tokenization
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

train_ds = Dataset.from_pandas(train_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))
val_ds = Dataset.from_pandas(val_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))
test_ds = Dataset.from_pandas(df_iea[['text', 'label_id']].rename(columns={'label_id': 'label'}))

print("Tokenization")
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# ==========================================
# 5. Training
# ==========================================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=9)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Relaxed Accuracy (+/- 1)
    diff = np.abs((labels+1) - (preds+1)) 
    relaxed_acc = np.mean(diff <= 1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "relaxed_accuracy": relaxed_acc,
        "f1_macro": f1_score(labels, preds, average='macro')
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="relaxed_accuracy",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("\n Starting training...")
trainer.train()

# Save
model.save_pretrained(OUTPUT_DIR + "/final")
tokenizer.save_pretrained(OUTPUT_DIR + "/final")

# ==========================================
# 6. FINAL TEST ON IEA gold standard
# ==========================================
print("\n EVALUATION ON GOLD STANDARD (IEA) :")
metrics = trainer.evaluate(test_ds)
print(metrics)

# ==========================================
# 7. FIGURES
# ==========================================
print("\nGenerating figures")

# --- A. Extract training history ---
history = trainer.state.log_history
train_loss = []
eval_loss = []
eval_acc = []
epochs = []

for entry in history:
    if 'loss' in entry and 'epoch' in entry:
        train_loss.append({'epoch': entry['epoch'], 'loss': entry['loss']})
    elif 'eval_loss' in entry:
        eval_loss.append({'epoch': entry['epoch'], 'loss': entry['eval_loss']})
        eval_acc.append({'epoch': entry['epoch'], 'acc': entry['eval_relaxed_accuracy']})

# Conversion to DF for plotting
df_train = pd.DataFrame(train_loss)
df_eval = pd.DataFrame(eval_loss)
df_acc = pd.DataFrame(eval_acc)

# --- B. Figure 1 : Learning Curves (Training vs Validation) ---
plt.figure(figsize=(10, 6))
plt.plot(df_train['epoch'], df_train['loss'], label='Training Loss', color='blue')
plt.plot(df_eval['epoch'], df_eval['loss'], label='Validation Loss', color='red', linestyle='--')
plt.title('Learning Curve: Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("figure1_learning_curve.png", dpi=300)
print(" Figure 1 saved: figure1_learning_curve.png")

# --- C. Figure 2 : Evolution of Accuracy (Relaxed Accuracy) ---
plt.figure(figsize=(10, 6))
plt.plot(df_acc['epoch'], df_acc['acc'], label='Relaxed Accuracy (+/- 1)', color='green')
plt.title('Validation Metric Evolution')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("figure2_accuracy_evolution.png", dpi=300)
print(" Figure 2 saved: figure2_accuracy_evolution.png")

# --- D. Figure 3 : Confusion Matrix (On the IEA Test Set) ---
print("Calculating the confusion matrix")
preds_output = trainer.predict(test_ds)
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = preds_output.label_ids

# Reset the labels to 1-9 for display
labels_axis = [str(i) for i in range(1, 10)]

cm = confusion_matrix(y_true, y_pred, labels=range(9))
# Normalization by row (% of success per class)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=labels_axis, yticklabels=labels_axis)
plt.xlabel('Predicted TRL (SciBERT)')
plt.ylabel('True TRL (Expert IEA)')
plt.title('Confusion Matrix (Normalized)')
plt.savefig("figure3_confusion_matrix.png", dpi=300)
print(" Figure 3 saved: figure3_confusion_matrix.png")

# --- E. Figure 4 : 2D Representation of Documents (t-SNE) ---
print("Generating 2D representation (t-SNE)...")

with torch.no_grad():
    train_embeddings = model.base_model(**tokenizer(train_df['text'].tolist(), padding=True, truncation=True, return_tensors="pt")).last_hidden_state[:, 0, :]
    val_embeddings = model.base_model(**tokenizer(val_df['text'].tolist(), padding=True, truncation=True, return_tensors="pt")).last_hidden_state[:, 0, :]
    test_embeddings = model.base_model(**tokenizer(df_iea['text'].tolist(), padding=True, truncation=True, return_tensors="pt")).last_hidden_state[:, 0, :]

# combine all embeddings for t-SNE
all_embeddings = np.concatenate([train_embeddings, val_embeddings, test_embeddings])
all_labels = np.concatenate([train_df['label_id'], val_df['label_id'], df_iea['label_id']])
all_domains = ['Silver'] * len(train_embeddings) + ['Silver'] * len(val_embeddings) + ['IEA'] * len(test_embeddings)

# Application du t-SNE
X = all_embeddings.cpu().numpy()
Y_trl = all_labels
domain = all_domains
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)

df_vis = pd.DataFrame({
    "x": X_2d[:, 0],
    "y": X_2d[:, 1],
    "trl": Y_trl,
    "domain": domain
})

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_vis,
    x="x", y="y",
    hue="trl", style="domain",
    palette="Spectral", alpha=0.6
)
plt.title("Latent TRL space (SciBERT) – Silver vs IEA")
plt.tight_layout()
plt.savefig("fig_tsne_scibert_trl_silver_iea.png", dpi=300)
print(" Figure 4 saved: fig_tsne_scibert_trl_silver_iea.png")
