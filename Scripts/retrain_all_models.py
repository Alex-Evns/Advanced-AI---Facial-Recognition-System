#!/usr/bin/env python
"""
Retrain all three models with corrected class ordering.

The models were previously trained with alphabetically-ordered classes, which 
caused the output to represent P(no_attribute) instead of P(attribute).

This script retrains all models with explicit class ordering so:
- Model output = P(has_attribute) for glasses, hat, young
- High output (>0.5) means "detected", low output (<0.5) means "not detected"

Usage: python Scripts/retrain_all_models.py
"""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print("=" * 80)
print("RETRAINING ALL MODELS WITH CORRECTED CLASS ORDERING")
print("=" * 80)

attributes = ["glasses", "hat", "young"]

for attribute in attributes:
    print(f"\n{'=' * 80}")
    print(f"Training {attribute.upper()} model...")
    print(f"{'=' * 80}")
    
    result = subprocess.run(
        [sys.executable, "Scripts/train_model.py", attribute],
        cwd=PROJECT_ROOT,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n✗ Failed to train {attribute} model")
        sys.exit(1)
    else:
        print(f"\n✓ Successfully trained {attribute} model")

print("\n" + "=" * 80)
print("ALL MODELS RETRAINED SUCCESSFULLY")
print("=" * 80)
print("\nThe models are now properly configured:")
print("- glasses: output = P(has_glasses)")
print("- hat:     output = P(has_hat)")
print("- young:   output = P(is_young)")
print("\nYou can now run: python src/test_agent.py test_images/your_image.jpg")
