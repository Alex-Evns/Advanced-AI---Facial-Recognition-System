from pathlib import Path
import shutil
import random

RANDOM_SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data"

DATASETS = {
    "glasses": ["glasses", "no_glasses"],
    "hat": ["hat", "no_hat"],
    "young": ["young", "not_young"]
}

for dataset_name, classes in DATASETS.items():

    print(f"\nProcessing {dataset_name}")

    for class_name in classes:

        source_dir = (
            BASE_DIR /
            "processed" /
            dataset_name /
            class_name
        )

        images = list(source_dir.glob("*.jpg"))

        random.shuffle(images)

        total = len(images)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        splits = {
            "train": train_images,
            "validation": val_images,
            "test": test_images
        }

        for split_name, split_images in splits.items():

            destination = (
                BASE_DIR /
                dataset_name /
                split_name /
                class_name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            for image_path in split_images:

                shutil.copy(
                    image_path,
                    destination / image_path.name
                )

        print(
            f"{class_name}: "
            f"{len(train_images)} train, "
            f"{len(val_images)} validation, "
            f"{len(test_images)} test"
        )

print("\nDataset splitting complete.")