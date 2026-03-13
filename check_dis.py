import os

train_dir = "dataset_mendeley_processed/train"

print("Training samples per class:\n")

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        print(class_name, ":", len(os.listdir(class_path)))