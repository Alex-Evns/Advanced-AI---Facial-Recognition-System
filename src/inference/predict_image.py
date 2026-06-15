from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
)

IMAGE_SIZE = (224, 224)

# =====================================================
# LOAD MODELS
# =====================================================

print("Loading models...")

glasses_model = tf.keras.models.load_model(
    MODEL_DIR / "glasses_model.keras"
)

hat_model = tf.keras.models.load_model(
    MODEL_DIR / "hat_model.keras"
)

young_model = tf.keras.models.load_model(
    MODEL_DIR / "young_model.keras"
)

print("Models loaded successfully.")

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image_path):

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = img_to_array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(
        image
    )

    return image

# =====================================================
# PREDICT
# =====================================================

def predict_attributes(image_path):

    image = preprocess_image(
        image_path
    )

    glasses_prob = float(
        glasses_model.predict(
            image,
            verbose=0
        )[0][0]
    )

    hat_prob = float(
        hat_model.predict(
            image,
            verbose=0
        )[0][0]
    )

    young_prob = float(
        young_model.predict(
            image,
            verbose=0
        )[0][0]
    )

    return {
        "glasses": 1 - glasses_prob,  # Invert: model outputs P(no_glasses), we want P(glasses)
        "hat": 1 - hat_prob,          # Invert: model outputs P(no_hat), we want P(hat)
        "young": young_prob           # Correct: model outputs P(young)
    }