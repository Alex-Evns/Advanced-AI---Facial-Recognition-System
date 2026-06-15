#!/usr/bin/env python
"""
Comprehensive diagnostic to check model training, class ordering, and predictions.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print("=" * 80)
print("COMPREHENSIVE MODEL DIAGNOSTIC")
print("=" * 80)

# Check directory structure
print("\n1. DIRECTORY STRUCTURE CHECK")
print("-" * 80)
for attr in ["glasses", "hat", "young"]:
    train_dir = PROJECT_ROOT / "Data" / attr / "train"
    if train_dir.exists():
        subdirs = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        print(f"{attr:10} train classes: {subdirs}")
    else:
        print(f"{attr:10} - DIRECTORY NOT FOUND: {train_dir}")

# Check models
print("\n2. MODEL FILES CHECK")
print("-" * 80)
for attr in ["glasses", "hat", "young"]:
    model_path = PROJECT_ROOT / "models" / "saved_models" / f"{attr}_model.keras"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"{attr:10} model exists: {size_mb:.1f} MB")
    else:
        print(f"{attr:10} - MODEL NOT FOUND: {model_path}")

# Test class ordering during data loading
print("\n3. CLASS ORDERING DURING DATA LOADING")
print("-" * 80)
for attr in ["glasses", "hat", "young"]:
    train_dir = PROJECT_ROOT / "Data" / attr / "train"
    if train_dir.exists():
        ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(224, 224),
            batch_size=32,
            label_mode="binary",
            shuffle=False
        )
        print(f"{attr:10} TF class order: {ds.class_names}")
        print(f"           Class 0={ds.class_names[0]}, Class 1={ds.class_names[1]}")
        print(f"           Model output represents: P({ds.class_names[1]})")

# Load models and test on actual data
print("\n4. LOADING MODELS AND CHECKING PREDICTIONS")
print("-" * 80)

try:
    glasses_model = tf.keras.models.load_model(
        PROJECT_ROOT / "models" / "saved_models" / "glasses_model.keras"
    )
    hat_model = tf.keras.models.load_model(
        PROJECT_ROOT / "models" / "saved_models" / "hat_model.keras"
    )
    young_model = tf.keras.models.load_model(
        PROJECT_ROOT / "models" / "saved_models" / "young_model.keras"
    )
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    exit(1)

# Test on test images if they exist
print("\n5. TEST PREDICTIONS ON AVAILABLE IMAGES")
print("-" * 80)

test_image_dir = PROJECT_ROOT / "test_images"
if test_image_dir.exists():
    test_images = sorted(list(test_image_dir.glob("*.jpg"))) + sorted(list(test_image_dir.glob("*.png")))
    print(f"Found {len(test_images)} test images\n")
    
    for img_path in test_images[:3]:  # Test first 3 images
        print(f"Testing: {img_path.name}")
        
        try:
            image = load_img(img_path, target_size=(224, 224))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)
            
            glasses_raw = float(glasses_model.predict(image_array, verbose=0)[0][0])
            hat_raw = float(hat_model.predict(image_array, verbose=0)[0][0])
            young_raw = float(young_model.predict(image_array, verbose=0)[0][0])
            
            print(f"  Raw model outputs:")
            print(f"    glasses: {glasses_raw:.4f}")
            print(f"    hat:     {hat_raw:.4f}")
            print(f"    young:   {young_raw:.4f}")
            
            print(f"  After inversion (if used):")
            print(f"    glasses: {1-glasses_raw:.4f} (inverted)")
            print(f"    hat:     {1-hat_raw:.4f} (inverted)")
            print(f"    young:   {young_raw:.4f} (NOT inverted)")
            print()
        except Exception as e:
            print(f"  Error processing image: {e}\n")
else:
    print(f"Test images directory not found: {test_image_dir}")

# Check on actual test dataset
print("\n6. PREDICTION DISTRIBUTION ON TEST SETS")
print("-" * 80)

for attr in ["glasses", "hat", "young"]:
    test_dir = PROJECT_ROOT / "Data" / attr / "test"
    if test_dir.exists():
        ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(224, 224),
            batch_size=32,
            label_mode="binary",
            shuffle=False
        )
        
        if attr == "glasses":
            model = glasses_model
        elif attr == "hat":
            model = hat_model
        else:
            model = young_model
        
        all_probs = []
        for images, labels in ds:
            probs = model.predict(images, verbose=0).flatten()
            all_probs.extend(probs)
        
        all_probs = np.array(all_probs)
        print(f"\n{attr.upper()}:")
        print(f"  Min prediction: {all_probs.min():.4f}")
        print(f"  Max prediction: {all_probs.max():.4f}")
        print(f"  Mean prediction: {all_probs.mean():.4f}")
        print(f"  Std prediction: {all_probs.std():.4f}")
        print(f"  % < 0.5: {(all_probs < 0.5).sum() / len(all_probs) * 100:.1f}%")
        print(f"  % >= 0.5: {(all_probs >= 0.5).sum() / len(all_probs) * 100:.1f}%")

print("\n" + "=" * 80)
print("END DIAGNOSTIC")
print("=" * 80)
