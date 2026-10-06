import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    accuracy_score
)


# ---------------------------------------------------------
# 1. Create required directories
# ---------------------------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)


# ---------------------------------------------------------
# 2. Load Breast Cancer Wisconsin Dataset
# ---------------------------------------------------------

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

print("=" * 60)
print("BREAST CANCER WISCONSIN DATASET")
print("=" * 60)

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print("\nClass distribution:")
print(y.value_counts().sort_index())


# Save dataset used in the project
dataset = X.copy()
dataset["target"] = y
dataset["target_name"] = y.map(
    {0: data.target_names[0], 1: data.target_names[1]}
)

dataset.to_csv(
    "data/breast_cancer_wisconsin.csv",
    index=False
)

print("\nDataset saved to:")
print("data/breast_cancer_wisconsin.csv")


# ---------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ---------------------------------------------------------
# 4. Standardize Features
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures standardized using StandardScaler.")


# ---------------------------------------------------------
# 5. Train Logistic Regression Model
# ---------------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION MODEL")
print("=" * 60)

print("Model training completed successfully.")


# ---------------------------------------------------------
# 6. Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test_scaled)

# Probability of positive class
y_probability = model.predict_proba(X_test_scaled)[:, 1]


# ---------------------------------------------------------
# 7. Evaluation Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))


# ---------------------------------------------------------
# 8. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(
    "visualizations/confusion_matrix.png",
    dpi=300
)
plt.close()


# ---------------------------------------------------------
# 9. ROC Curve
# ---------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "visualizations/roc_curve.png",
    dpi=300
)
plt.close()


# ---------------------------------------------------------
# 10. Sigmoid Function
# ---------------------------------------------------------

z = np.linspace(-10, 10, 200)

sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 6))

plt.plot(z, sigmoid)

plt.axhline(
    0.5,
    linestyle="--",
    label="Probability = 0.5"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Linear Input (z)")
plt.ylabel("Sigmoid Probability")
plt.title("Sigmoid Function")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "visualizations/sigmoid_curve.png",
    dpi=300
)
plt.close()


# ---------------------------------------------------------
# 11. Threshold Tuning
# ---------------------------------------------------------

thresholds_to_test = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]

threshold_results = []

for threshold in thresholds_to_test:

    y_threshold_pred = (
        y_probability >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        y_threshold_pred,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        y_threshold_pred,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Precision": threshold_precision,
        "Recall": threshold_recall
    })


threshold_df = pd.DataFrame(threshold_results)

print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

print(threshold_df.to_string(index=False))


# Save threshold results
threshold_df.to_csv(
    "data/threshold_results.csv",
    index=False
)


# ---------------------------------------------------------
# 12. Threshold Visualization
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall at Different Thresholds")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "visualizations/threshold_analysis.png",
    dpi=300
)
plt.close()


# ---------------------------------------------------------
# 13. Final Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print("Files generated:")
print("1. data/breast_cancer_wisconsin.csv")
print("2. data/threshold_results.csv")
print("3. visualizations/confusion_matrix.png")
print("4. visualizations/roc_curve.png")
print("5. visualizations/sigmoid_curve.png")
print("6. visualizations/threshold_analysis.png")

print("\nLogistic Regression classification task completed successfully.")