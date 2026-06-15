from pathlib import Path
import sys
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================================
# ARGUMENT
# =====================================================

if len(sys.argv) != 2:
    print("Usage: python scripts/train_model.py <glasses|hat|young>")
    sys.exit(1)

ATTRIBUTE = sys.argv[1].lower()

# =====================================================
# CONFIG
# =====================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / ATTRIBUTE / "train"
VAL_DIR = PROJECT_ROOT / "data" / ATTRIBUTE / "validation"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / f"{ATTRIBUTE}_model.keras"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "training_history"
    / f"{ATTRIBUTE}_history.pkl"
)

WEIGHTS_PATH = (
    PROJECT_ROOT
    / "models"
    / "pretrained"
    / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
)

# =====================================================
# LOAD DATA
# =====================================================

print(f"\nTraining model: {ATTRIBUTE}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

print("\nCLASS NAMES:")
print(train_ds.class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =====================================================
# AUGMENTATION
# =====================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# =====================================================
# BASE MODEL
# =====================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

base_model.load_weights(WEIGHTS_PATH)

base_model.trainable = False

# =====================================================
# BUILD MODEL
# =====================================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(inputs, outputs)

# =====================================================
# COMPILE
# =====================================================

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
# TRAIN
# =====================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# =====================================================
# SAVE
# =====================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

model.save(MODEL_PATH)

HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history.history, f)

print(f"\nSaved model: {MODEL_PATH}")
print(f"Saved history: {HISTORY_PATH}")