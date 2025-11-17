import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Paths ===
model_path = "./risk_detector_optimized/checkpoint-450"
tokenizer_name = "distilbert-base-uncased"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# === Load test CSV ===
df = pd.read_csv("data/DATD_and_DATD+Rand_test.csv")

# Ensure text column
if "text" in df.columns:
    df = df.rename(columns={"text": "sentence"})
elif "sentence" not in df.columns:
    raise ValueError("Dataset must contain 'text' or 'sentence' column.")

# Ensure label column exists (for metrics)
if "label" not in df.columns:
    raise ValueError("Test CSV must contain a 'label' column for evaluation.")

# Normalize labels like in training
df["label"] = df["label"].astype(str).str.strip().str.upper()

label_map = {"OTHER": 0, "MENTAL_HEALTH": 1}
id2label = {v: k for k, v in label_map.items()}

df["true_id"] = df["label"].map(label_map)

if df["true_id"].isna().any():
    bad = df[df["true_id"].isna()]["label"].unique()
    raise ValueError(f"Unknown labels in test file: {bad}")

texts = df["sentence"].tolist()

# === Prediction function ===
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
    return pred_id, probs[0].tolist()

pred_ids = []
probabilities = []

for text in texts:
    pid, probs = predict(text)
    pred_ids.append(pid)
    probabilities.append(probs)

# === Add predictions to DataFrame ===
df["pred_id"] = pred_ids
df["predicted_label"] = [id2label[p] for p in pred_ids]
df["probabilities"] = probabilities          # list [p_other, p_mh]
df["mh_prob"] = df["probabilities"].apply(lambda x: x[1])  # probability of MENTAL_HEALTH

# === Compute metrics ===
y_true = df["true_id"]
y_pred = df["pred_id"]

print("\n✅ Evaluation on test set")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["OTHER", "MENTAL_HEALTH"]))
print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))

# === Save results to CSV ===
df.to_csv("DATA_test_with_predictions.csv", index=False)
print("\n🎉 Finished! Results saved to DATA_test_with_predictions.csv")