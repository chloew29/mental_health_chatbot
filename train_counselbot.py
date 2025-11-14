from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)
import pandas as pd

# === Load CounselChat dataset ===
print("📥 Loading dataset...")
ds = load_dataset("nbertagnolli/counsel-chat")

# Remove rows missing question or answer
print("🧹 Cleaning dataset...")
ds = ds.filter(lambda x: x["questionText"] is not None and x["answerText"] is not None)

print("✅ Cleaned dataset sizes:", ds)

# === Tokenizer & model ===
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# === Preprocess ===
def preprocess(batch):
    questions = ["question: " + str(q) for q in batch["questionText"]]
    answers = [str(a) for a in batch["answerText"]]
    model_inputs = tokenizer(
        questions, truncation=True, padding="max_length", max_length=256
    )
    labels = tokenizer(
        answers, truncation=True, padding="max_length", max_length=256
    ).input_ids
    model_inputs["labels"] = labels
    return model_inputs

print("🔄 Tokenizing...")
ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir="./counselbot",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs_counsel",
    logging_strategy="epoch",
    save_total_limit=1,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"].shuffle(seed=42).select(range(2000)),  # safe subset
    eval_dataset=ds["train"].shuffle(seed=123).select(range(300)),   # small eval sample
)


print("🚀 Starting training...")
trainer.train()

print("🎉 Training completed! Model saved to ./counselbot")
