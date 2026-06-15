from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"

IMAGE_SIZE = (224, 224)

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

    return image


def predict_attributes(image_path):
    image = preprocess_image(image_path)

    glasses_prob = float(
        glasses_model.predict(image, verbose=0)[0][0]
    )

    hat_prob = float(
        hat_model.predict(image, verbose=0)[0][0]
    )

    young_prob = float(
        young_model.predict(image, verbose=0)[0][0]
    )

    return {
        "glasses": glasses_prob,
        "hat": hat_prob,
        "young": young_prob
    }