from pathlib import Path
import pickle

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================================
# CONFIG
# =====================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

# Start small while testing
EPOCHS = 10

LEARNING_RATE = 0.001

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "glasses"
    / "train"
)

VAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "glasses"
    / "validation"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "glasses_model.keras"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "training_history"
    / "glasses_history.pkl"
)

# =====================================================
# CHECK PATHS
# =====================================================

print("\nChecking paths...")

print("Train directory:")
print(TRAIN_DIR)

print("\nValidation directory:")
print(VAL_DIR)

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found:\n{VAL_DIR}"
    )

# =====================================================
# LOAD DATASETS
# =====================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

# =====================================================
# PERFORMANCE OPTIMISATION
# =====================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

# =====================================================
# DATA AUGMENTATION
# =====================================================

print("\nCreating augmentation pipeline...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =====================================================
# LOAD MOBILENETV2
# =====================================================

print("\nLoading MobileNetV2...")


WEIGHTS_PATH = (
    PROJECT_ROOT
    / "models"
    / "pretrained"
    / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
)

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Could not find weights file:\n{WEIGHTS_PATH}"
    )

print("\nLoading MobileNetV2 weights from local file...")
print(WEIGHTS_PATH)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

base_model.load_weights(WEIGHTS_PATH)

print("Weights loaded successfully.")

base_model.trainable = False

# =====================================================
# BUILD MODEL
# =====================================================

print("\nBuilding model...")

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(
    inputs,
    outputs
)

# =====================================================
# COMPILE
# =====================================================

print("\nCompiling model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =====================================================
# CALLBACKS
# =====================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# =====================================================
# SUMMARY
# =====================================================

print("\nModel Summary:\n")

model.summary()

# =====================================================
# TRAIN
# =====================================================

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# =====================================================
# SAVE MODEL
# =====================================================

print("\nSaving model...")

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

model.save(MODEL_PATH)

print(
    f"\nModel saved successfully:\n{MODEL_PATH}"
)

# =====================================================
# SAVE HISTORY
# =====================================================

print("\nSaving training history...")

HISTORY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(
        history.history,
        f
    )

print(
    f"\nHistory saved successfully:\n{HISTORY_PATH}"
)

# =====================================================
# FINAL RESULTS
# =====================================================

final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print(
    f"Final Training Accuracy: "
    f"{final_train_acc:.4f}"
)

print(
    f"Final Validation Accuracy: "
    f"{final_val_acc:.4f}"
)