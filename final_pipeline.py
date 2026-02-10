import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from severity_estimation import estimate_severity
from cure_recommendation import recommend_cure

# Load trained CNN model
model = load_model("potato_disease_cnn.h5")

# Class labels (IMPORTANT: must match training order)
class_labels = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

def predict_pipeline(image_path):
    # --- Step 1: CNN Disease Classification ---
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]

    print(f"\nDisease Prediction: {predicted_class}")

    # --- Step 2: Severity Estimation ---
    if predicted_class != "Potato___healthy":
        infected_pct, severity = estimate_severity(image_path)

        print(f"Infected Area: {infected_pct:.2f}%")
        print(f"Estimated Severity: {severity}")

        # --- Step 3: Cure Recommendation ---
        cure = recommend_cure(predicted_class, severity)
        print(f"Recommended Action: {cure}")

    else:
        print("Leaf is healthy. No treatment required.")

if __name__ == "__main__":
    test_image = "dataset_processed/test/Potato___Early_blight/0c4f6f72-c7a2-42e1-9671-41ab3bf37fe7___RS_Early.B 6752.jpg"
    predict_pipeline(test_image)
