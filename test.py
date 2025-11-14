from transformers.training_args import TrainingArguments
a = TrainingArguments(output_dir="./tmp", evaluation_strategy="epoch")
print("✅ Works! Loaded:", a)
