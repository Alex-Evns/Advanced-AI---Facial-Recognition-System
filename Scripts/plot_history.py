from pathlib import Path
import sys
import pickle
import matplotlib.pyplot as plt

# =====================================================
# ARGUMENT CHECK
# =====================================================

if len(sys.argv) != 2:
    print("Usage: python plot_history.py <glasses|hat|young>")
    sys.exit(1)

ATTRIBUTE = sys.argv[1].lower()

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "training_history"
    / f"{ATTRIBUTE}_history.pkl"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "report_assets"
    / "figures"
    / ATTRIBUTE
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD HISTORY
# =====================================================

with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

# =====================================================
# ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])

plt.title(
    f"{ATTRIBUTE.title()} Model Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Training",
    "Validation"
])

plt.grid(True)

plt.savefig(
    FIGURE_DIR / "accuracy.png",
    bbox_inches="tight"
)

plt.close()

# =====================================================
# LOSS GRAPH
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(history["loss"])
plt.plot(history["val_loss"])

plt.title(
    f"{ATTRIBUTE.title()} Model Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    "Training",
    "Validation"
])

plt.grid(True)

plt.savefig(
    FIGURE_DIR / "loss.png",
    bbox_inches="tight"
)

plt.close()

print(f"\nSaved graphs for {ATTRIBUTE}")