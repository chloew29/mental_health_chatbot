import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import numpy as np
from evaluate import load as load_metric


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


# 2. Tokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

dataset = dataset.map(preprocess, batched=True)


# Model (with freezing + dropout) to avoid overfit
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

# increase dropout 
model.config.dropout = 0.3
model.config.attention_dropout = 0.3

# Freeze the DistilBERT encoder; train only classifier head
for param in model.distilbert.parameters():
    param.requires_grad = False

print("\n🧊 Frozen DistilBERT encoder; training only classifier head.")


# Metrics
acc_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# 5. Training arguments (tuned for small data)

training_args = TrainingArguments(
    output_dir="./risk_detector_optimized",
    eval_strategy="epoch",              
    save_strategy="epoch",             
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_total_limit=2,
    logging_dir="./logs_optimized",
    logging_strategy="epoch",

    # Early stopping / best-model settings:
    load_best_model_at_end=True,        # load best checkpoint when training finishes
    metric_for_best_model="eval_loss",  # monitor validation loss
    greater_is_better=False,            # lower loss = better
)


# Trainer with early stopping

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


trainer.train()
print("🎉 Training completed! Checkpoints in ./risk_detector_optimized")


log_df = pd.DataFrame(trainer.state.log_history)
log_df.to_csv("training_log.csv", index=False)
print("📝 Saved training log to training_log.csv")