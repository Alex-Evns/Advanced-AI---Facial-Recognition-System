from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "glasses"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "glasses_model.keras"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "report_assets"
    / "figures"
)

CONFUSION_MATRIX_PATH = (
    FIGURE_DIR
    / "glasses_confusion_matrix.png"
)

# =====================================================
# CHECKS
# =====================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

# =====================================================
# LOAD TEST DATA
# =====================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

# =====================================================
# GENERATE PREDICTIONS
# =====================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = (
        predictions > 0.5
    ).astype(int)

    y_true.extend(
        labels.numpy().flatten()
    )

    y_pred.extend(
        predictions.flatten()
    )

y_true = np.array(y_true)
y_pred = np.array(y_pred)

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

# =====================================================
# RESULTS
# =====================================================

print("\n==============================")
print("GLASSES MODEL EVALUATION")
print("==============================")

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

fpr = fp / (fp + tn) 

fnr = fn / (fn + tp)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Glasses",
        "Glasses"
    ]
)

display.plot()

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    CONFUSION_MATRIX_PATH,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nConfusion matrix saved to:\n"
    f"{CONFUSION_MATRIX_PATH}"
)


print("\n==============================")
print("CONFUSION MATRIX VALUES")
print("==============================")

print(f"True Positives : {tp}")
print(f"True Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Specificity     : {specificity:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"False Negative Rate: {fnr:.4f}")

print("\nEvaluation complete.")