from pathlib import Path
import pandas as pd
import shutil

# ----------------------------
# CONFIG
# ----------------------------

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 2000

# Resolve data directory relative to this script (matches workspace capitalization)
BASE_DIR = Path(__file__).resolve().parent.parent / "Data"

# Paths in the workspace use `Raw` and `Processed`
ATTR_FILE = BASE_DIR / "Raw" / "list_attr_celeba.csv"
IMAGE_DIR = BASE_DIR / "Raw" / "img_align_celeba"

OUTPUT_DIR = BASE_DIR / "Processed"

# ----------------------------
# LOAD DATA
# ----------------------------

if not ATTR_FILE.exists():
    raise FileNotFoundError(f"Attribute file not found: {ATTR_FILE}")
if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ATTR_FILE)

# ----------------------------
# DATASETS TO CREATE
# ----------------------------

DATASETS = {
    "glasses": {
        "attribute": "Eyeglasses",
        "positive": "glasses",
        "negative": "no_glasses"
    },
    "hat": {
        "attribute": "Wearing_Hat",
        "positive": "hat",
        "negative": "no_hat"
    },
    "young": {
        "attribute": "Young",
        "positive": "young",
        "negative": "not_young"
    }
}

# ----------------------------
# BUILD DATASETS
# ----------------------------

for dataset_name, config in DATASETS.items():

    attribute = config["attribute"]

    print(f"\nCreating {dataset_name} dataset...")

    positive_df = df[df[attribute] == 1]
    negative_df = df[df[attribute] == -1]

    positive_sample = positive_df.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_SEED
    )

    negative_sample = negative_df.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_SEED
    )

    pos_dir = OUTPUT_DIR / dataset_name / config["positive"]
    neg_dir = OUTPUT_DIR / dataset_name / config["negative"]

    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    for image_name in positive_sample["image_id"]:
        src = IMAGE_DIR / image_name
        dst = pos_dir / image_name
        shutil.copy(src, dst)

    for image_name in negative_sample["image_id"]:
        src = IMAGE_DIR / image_name
        dst = neg_dir / image_name
        shutil.copy(src, dst)

    print(
        f"{dataset_name}: "
        f"{len(positive_sample)} positive, "
        f"{len(negative_sample)} negative"
    )

print("\nDone.")