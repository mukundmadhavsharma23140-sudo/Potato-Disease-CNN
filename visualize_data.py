import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

DATASET_DIR = "dataset_processed/train"
IMG_SIZE = (224, 224)

classes = os.listdir(DATASET_DIR)

plt.figure(figsize=(9, 6))

for i, cls in enumerate(classes):
    cls_path = os.path.join(DATASET_DIR, cls)
    img_name = random.choice(os.listdir(cls_path))
    img_path = os.path.join(cls_path, img_name)

    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0

    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()
plt.show()
