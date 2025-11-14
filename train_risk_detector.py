import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# === Load dataset ===
df = pd.read_csv("DATD_training.csv")
print("📄 Columns:", df.columns.tolist())
print(df.head())

# Ensure proper column naming
if "text" in df.columns:
    df = df.rename(columns={"text": "sentence"})
elif "sentence" not in df.columns:
    raise ValueError("Dataset must contain a 'text' or 'sentence' column.")

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

# === Normalize labels ===
df["label"] = df["label"].astype(str).str.strip().str.upper()
print("\n🔍 Unique raw labels:", df["label"].unique())

# === Map labels ===
label_map = {"MENTAL_HEALTH": 1, "OTHER": 0}
df["label"] = df["label"].map(label_map)

# Drop invalid rows
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("\n✅ Label distribution:")
print(df["label"].value_counts())

if len(df) == 0:
    raise ValueError("❌ No valid samples found after mapping. Check label names in CSV.")

# === Convert to Hugging Face dataset ===
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# === Tokenization ===
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=True)

# === Model ===
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# === Training Arguments (new syntax for v4.57+) ===
training_args = TrainingArguments(
    output_dir="./risk_detector",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_total_limit=1,
    logging_dir="./logs",
    logging_strategy="epoch",
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# === Train ===
trainer.train()
print("🎉 Training completed! Model saved to ./risk_detector")
