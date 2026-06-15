#!/usr/bin/env python
"""
Diagnostic script to verify class ordering in binary classification models.
This reveals how TensorFlow alphabetically orders classes and what model outputs mean.
"""

from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print("=" * 80)
print("CLASS ORDERING DIAGNOSIS")
print("=" * 80)

for attribute in ["glasses", "hat", "young"]:
    print(f"\n{attribute.upper()}")
    print("-" * 80)
    
    train_dir = PROJECT_ROOT / "Data" / attribute / "train"
    
    # Load dataset to see the class names in order
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode="binary",
        shuffle=False
    )
    
    class_names = ds.class_names
    print(f"Classes (alphabetically ordered by TensorFlow):")
    for idx, class_name in enumerate(class_names):
        print(f"  [{idx}] {class_name}")
    
    print(f"\nModel output interpretation:")
    print(f"  P = model output probability value")
    print(f"  P = Probability of class 1 ('{class_names[1]}')")
    print(f"  1-P = Probability of class 0 ('{class_names[0]}')")
    print(f"\nExample interpretations:")
    print(f"  If P=0.8 → 80% chance of '{class_names[1]}' and 20% chance of '{class_names[0]}'")
    print(f"  If P=0.3 → 30% chance of '{class_names[1]}' and 70% chance of '{class_names[0]}'")

print("\n" + "=" * 80)
print("KEY FINDING:")
print("=" * 80)
print("""
The model outputs a SINGLE probability value (sigmoid activation).
In binary classification, this value represents: P(class_1)

For correct interpretation:
- glasses: output = P(no_glasses), so 1-output = P(glasses) ❌ INVERTED
- hat:     output = P(no_hat), so 1-output = P(hat) ❌ INVERTED  
- young:   output = P(young), so output = P(young) ✓ CORRECT

SOLUTION: For glasses and hat, invert the probabilities!
""")
