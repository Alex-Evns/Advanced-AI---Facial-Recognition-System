# Comprehensive Analysis of Prediction Errors

## Problems Identified

### 1. **Class Ordering Mismatch (PRIMARY ISSUE)**

The original models were trained with **backwards semantics** due to TensorFlow's alphabetical directory ordering:

```
GLASSES:
  Training directories: ['glasses', 'no_glasses']
  TensorFlow assignment: class 0='glasses', class 1='no_glasses'
  Model output: P(no_glasses) [BACKWARDS]

HAT:
  Training directories: ['hat', 'no_hat']
  TensorFlow assignment: class 0='hat', class 1='no_hat'
  Model output: P(no_hat) [BACKWARDS]

YOUNG:
  Training directories: ['not_young', 'young']  
  TensorFlow assignment: class 0='not_young', class 1='young'
  Model output: P(young) [CORRECT by accident]
```

### 2. **Why Confidence Levels Were Way Off**

User reported: "I never seem to get an output that says glasses detected, or hat detected or not young"

**Root Cause**: Model outputs were being interpreted backwards:
- Raw output 0.60 for glasses = 60% confidence in NO glasses  
- But displayed as "Glasses detected (60%)" = 60% confidence in GLASSES
- This is semantically backwards

When inverted to fix: output 0.40 = 40% chance of glasses → "No Glasses detected (60%)"
- This is correct, but ONLY showed "no glasses", never showed "glasses detected"

### 3. **Training Script Issues**

The training and evaluation scripts had TWO problems:

1. **No explicit class order specification**:
   ```python
   # OLD (WRONG):
   train_ds = tf.keras.utils.image_dataset_from_directory(
       TRAIN_DIR,
       label_mode="binary"
       # No class_names - uses alphabetical ordering!
   )
   ```

2. **Wrong directory path** (evaluate_model.py):
   ```python
   # OLD (WRONG):
   TEST_DIR = PROJECT_ROOT / "data" / ATTRIBUTE / "test"  # lowercase
   
   # CORRECT:
   TEST_DIR = PROJECT_ROOT / "Data" / ATTRIBUTE / "test"  # uppercase
   ```

## Solutions Implemented

### 1. **Explicit Class Ordering in Training**

Fixed `Scripts/train_model.py`:
```python
# NEW (CORRECT):
if ATTRIBUTE == "glasses":
    class_names = ["no_glasses", "glasses"]  # negative first, positive second
elif ATTRIBUTE == "hat":
    class_names = ["no_hat", "hat"]
elif ATTRIBUTE == "young":
    class_names = ["not_young", "young"]

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    label_mode="binary",
    class_names=class_names  # Explicitly set order
)
```

This ensures:
- Label 0 assigned to negative class folder  
- Label 1 assigned to positive class folder
- Model learns: P(positive class)

### 2. **Fixed Evaluation Script**

Updated `Scripts/evaluate_model.py`:
- Fixed path: `"data"` → `"Data"` (case sensitivity)
- Added explicit class_names matching training

### 3. **Removed Manual Inversion**

Updated `src/inference/predict_image.py`:
```python
# OLD (tried to fix backwards models):
return {
    "glasses": 1 - glasses_prob,  # Manual inversion
    "hat": 1 - hat_prob,
    "young": young_prob
}

# NEW (models are now correct):
return {
    "glasses": glasses_prob,      # Directly P(glasses)
    "hat": hat_prob,              # Directly P(hat)
    "young": young_prob           # Directly P(young)
}
```

### 4. **Updated Agent Logic**

`src/agents/facial_analysis_agent.py` now works correctly with properly-ordered models:
- `glasses >= 0.5`: "Glasses detected (confidence%)"
- `glasses < 0.5`: "No Glasses detected ((1-glasses)%)"

## Model Retraining Results

All three models were retrained with corrected class ordering:

```
GLASSES MODEL:
  Val Accuracy: 91.25%
  Status: Good convergence

HAT MODEL:
  Val Accuracy: 96.75%
  Status: Excellent convergence

YOUNG MODEL:
  Val Accuracy: 76%
  Status: Lower accuracy - may indicate:
    - Difficult classification task
    - Imbalanced training data
    - Poor CelebA "young" attribute definition
```

## Testing Current Predictions

Example test image: `glasses_hat.jpg` (person wearing both glasses and hat)

**Current Model Outputs:**
```
Raw predictions: glasses=0.45, hat=0.26, young=0.24

Agent Analysis:
- No Glasses detected (55.3% confidence)
- No Hat detected (74.3% confidence)  
- Not young detected (76.4% confidence)
```

**Issue**: Still predicting negative when visual inspection shows positive.

This could indicate:
1. Test images aren't from the same distribution as training (CelebA)
2. Models still have issues (double-check needed)
3. Training data labels are incorrect
4. These specific test images don't match their filenames

## Next Steps Required

### CRITICAL - Verify Model Accuracy

Run the evaluation on the actual test set:
```bash
python Scripts/evaluate_model.py glasses
python Scripts/evaluate_model.py hat
python Scripts/evaluate_model.py young
```

Check the confusion matrices - if accuracy is 90%+, models are working and test images might just be mismatched.

### If Models Show >85% Accuracy:

The test images in `test_images/` might not be from the CelebA dataset or might be incorrectly named. Try finding actual CelebA test images that visually have the attributes.

### If Models Show <70% Accuracy:

There's still a semantic issue. Possible causes:
- Class labels are actually backwards in the training data
- Need to swap the `class_names` order
- Training data was mislabeled during extraction

### Young Classifier Investigation:

The young classifier has the lowest accuracy (76%). Options:
1. Try additional training with more epochs
2. Adjust learning rate
3. Check CelebA dataset - the "young" attribute may be poorly defined
4. Consider removing young classifier if accuracy is unacceptable

## Files Modified

- `Scripts/train_model.py` - Added explicit class_names
- `Scripts/evaluate_model.py` - Fixed path, added class_names
- `src/inference/predict_image.py` - Removed manual inversion
- `src/agents/facial_analysis_agent.py` - Confidence display fixed
- `Scripts/retrain_all_models.py` - New convenience script
- `src/comprehensive_diagnostic.py` - New diagnostic tool

## How to Verify Fix

```bash
# Run comprehensive diagnostic
python src/comprehensive_diagnostic.py

# Test on an image
python src/test_agent.py test_images/test_image_8.jpg

# Evaluate on test set
python Scripts/evaluate_model.py glasses
```

The "After inversion (if used)" section in diagnostic output shows what values WOULD be if we still had backwards semantics - these should NOT be needed anymore.
