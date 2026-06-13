from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# =====================================================
# ARGUMENT CHECK
# =====================================================

if len(sys.argv) != 2:
    print("Usage: python evaluate_model.py <glasses|hat|young>")
    sys.exit(1)

ATTRIBUTE = sys.argv[1].lower()

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / ATTRIBUTE / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / f"{ATTRIBUTE}_model.keras"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "report_assets"
    / "figures"
    / ATTRIBUTE
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "report_assets"
    / "results"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD TEST DATA
# =====================================================

print(f"\nEvaluating {ATTRIBUTE} model...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

# =====================================================
# LOAD MODEL
# =====================================================

model = tf.keras.models.load_model(MODEL_PATH)

# =====================================================
# PREDICTIONS
# =====================================================

y_true = []
y_pred = []
y_prob = []

for images, labels in test_ds:

    probs = model.predict(
        images,
        verbose=0
    )

    preds = (probs > 0.5).astype(int)

    y_true.extend(
        labels.numpy().flatten()
    )

    y_pred.extend(
        preds.flatten()
    )

    y_prob.extend(
        probs.flatten()
    )

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)
fpr_value = fp / (fp + tn)
fnr = fn / (fn + tp)

# =====================================================
# ROC / AUC
# =====================================================

roc_fpr, roc_tpr, _ = roc_curve(
    y_true,
    y_prob
)

roc_auc = auc(
    roc_fpr,
    roc_tpr
)

# =====================================================
# PRINT RESULTS
# =====================================================

print("\n====================================")
print(f"{ATTRIBUTE.upper()} MODEL RESULTS")
print("====================================")

print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Specificity   : {specificity:.4f}")
print(f"AUC Score     : {roc_auc:.4f}")

print("\nConfusion Matrix Values")

print(f"TP            : {tp}")
print(f"TN            : {tn}")
print(f"FP            : {fp}")
print(f"FN            : {fn}")

print(f"\nFalse Positive Rate : {fpr_value:.4f}")
print(f"False Negative Rate : {fnr:.4f}")

# =====================================================
# SAVE METRICS CSV
# =====================================================

metrics_df = pd.DataFrame([
    {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "auc": roc_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }
])

metrics_df.to_csv(
    RESULTS_DIR / f"{ATTRIBUTE}_metrics.csv",
    index=False
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    f"{ATTRIBUTE.title()} Confusion Matrix"
)

plt.savefig(
    FIGURE_DIR / "confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

# =====================================================
# ROC CURVE
# =====================================================

plt.figure(figsize=(8, 6))

plt.plot(
    roc_fpr,
    roc_tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    f"{ATTRIBUTE.title()} ROC Curve"
)

plt.legend()
plt.grid(True)

plt.savefig(
    FIGURE_DIR / "roc_curve.png",
    bbox_inches="tight"
)

plt.close()

print("\nSaved evaluation outputs successfully.")