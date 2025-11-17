import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
from evaluate import load as load_metric

# === Load dataset ===
df = pd.read_csv("data/DATD_training.csv")
print("📄 Columns:", df.columns.tolist())
print(df.head())


if "text" in df.columns:
    df = df.rename(columns={"text": "sentence"})
elif "sentence" not in df.columns:
    raise ValueError("Dataset must contain a 'text' or 'sentence' column.")

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")


df["label"] = df["label"].astype(str).str.strip().str.upper()
print("\n🔍 Unique raw labels:", df["label"].unique())


label_map = {"MENTAL_HEALTH": 1, "OTHER": 0}
df["label"] = df["label"].map(label_map)


df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("\n✅ Label distribution:")
print(df["label"].value_counts())

if len(df) == 0:
    raise ValueError("❌ No valid samples found after mapping. Check label names in CSV.")


dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

dataset = dataset.map(preprocess, batched=True)

# === Model ===
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# === Metrics ===
acc_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./risk_detector",
    eval_strategy="epoch",                
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_total_limit=1,
    logging_dir="./logs",
    logging_strategy="epoch",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)


trainer.train()
print("🎉 Training completed! Model checkpoints in ./risk_detector")


log_df = pd.DataFrame(trainer.state.log_history)
log_df.to_csv("training_log.csv", index=False)
print("📝 Saved training log to training_log.csv")