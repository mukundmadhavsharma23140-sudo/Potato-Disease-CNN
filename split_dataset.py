import os
import shutil
import random

# Paths
SOURCE_DIR = "dataset"
TARGET_DIR = "dataset_processed"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

classes = os.listdir(SOURCE_DIR)

for cls in classes:
    class_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for file in files:
            src = os.path.join(class_path, file)
            dst = os.path.join(split_dir, file)
            shutil.copy(src, dst)

print("✅ Dataset successfully split into train / val / test.")
