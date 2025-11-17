import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# Load test predictions
df = pd.read_csv("DATA_test_with_predictions.csv")

label_map = {"OTHER": 0, "MENTAL_HEALTH": 1}
df["true_label"] = df["label"].map(label_map) if "label" in df else None
df["pred_label"] = df["predicted_label"].map(label_map)


# Make Confusion Matrix
cm = confusion_matrix(df["true_label"], df["pred_label"])

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["OTHER", "MENTAL_HEALTH"],
    yticklabels=["OTHER", "MENTAL_HEALTH"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Label Distribution
plt.figure(figsize=(6,4))
df["true_label"].map({0: "OTHER", 1: "MENTAL_HEALTH"}).value_counts().plot(
    kind="bar", alpha=0.6, label="True"
)
df["predicted_label"].value_counts().plot(
    kind="bar", alpha=0.6, label="Predicted"
)
plt.legend()
plt.title("True vs Predicted Label Distribution")
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.show()

# Classification report

print("\n📊 Classification Report:\n")
print(
    classification_report(
        df["true_label"],
        df["pred_label"],
        target_names=["OTHER", "MENTAL_HEALTH"],
    )
)


# ROC curve
df["mh_prob"] = df["probabilities"].apply(
    lambda x: ast.literal_eval(x)[1] if isinstance(x, str) else x[1]
)

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(df["true_label"], df["mh_prob"])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# Training vs Test Loss Curve 
log_df = pd.read_csv("training_log.csv")

train_df = log_df[log_df["loss"].notna()]
train_epochs = train_df["epoch"].tolist()
train_loss = train_df["loss"].tolist()

eval_df = log_df[log_df["eval_loss"].notna()]
eval_epochs = eval_df["epoch"].tolist()
eval_loss = eval_df["eval_loss"].tolist()

plt.figure(figsize=(6,4))
plt.plot(train_epochs, train_loss, marker="o", label="Training")
plt.plot(eval_epochs,  eval_loss,  marker="o", label="Test")

plt.xlabel("Epoch")
plt.ylabel("Loss")  # like E_RMS in your reference plot
plt.title("Training vs Test Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_test_loss_curve.png")
plt.show()