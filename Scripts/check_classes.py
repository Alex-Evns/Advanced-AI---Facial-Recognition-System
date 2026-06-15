from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

for attribute in ["glasses", "hat", "young"]:

    train_dir = PROJECT_ROOT / "data" / attribute / "train"

    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode="binary",
        shuffle=False
    )

    print(f"\n{attribute.upper()}")
    print(ds.class_names)