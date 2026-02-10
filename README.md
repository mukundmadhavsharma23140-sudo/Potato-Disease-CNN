# Image-Based Potato Leaf Disease Detection Using CNN

This project presents an end-to-end decision-support system for potato leaf disease detection using deep learning and image processing techniques. The system classifies potato leaf images into Early Blight, Late Blight, and Healthy categories using a Convolutional Neural Network (CNN). For diseased leaves, disease severity is estimated using infected leaf area percentage, and appropriate treatment recommendations are provided using rule-based logic.

---

## 📌 Features

- CNN-based classification of potato leaf diseases
- Support for Early Blight, Late Blight, and Healthy leaves
- Data augmentation and preprocessing for robust training
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Severity estimation based on infected leaf area percentage (image processing)
- Severity-aware treatment recommendation using predefined agronomic rules
- Fully modular and explainable pipeline

---

## 🧠 System Pipeline

1. **Input Image**
2. **CNN Disease Classification**
3. **Severity Estimation (for diseased leaves only)**
   - HSV-based segmentation
   - Infected leaf area percentage calculation
4. **Rule-Based Cure Recommendation**

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Image Processing:** OpenCV  
- **Data Handling:** NumPy  
- **Evaluation:** scikit-learn  
- **Visualization:** Matplotlib, Seaborn  

---

## 📂 Project Structure

Potato_Disease_CNN/
│
├── split_dataset.py # Train/val/test split
├── visualize_data.py # Sample visualization
├── train_cnn.py # CNN training script
├── evaluate_model.py # Test evaluation & confusion matrix
├── severity_estimation.py # Severity estimation module
├── cure_recommendation.py # Rule-based treatment logic
├── final_pipeline.py # End-to-end pipeline
├── requirements.txt
├── README.md
└── .gitignore


> **Note:** Datasets, trained model files, and virtual environments are excluded from this repository due to size and reproducibility constraints.

---

## 📊 Model Performance (Test Set)

- Overall Accuracy: **96%**
- Strong performance on Early Blight and Late Blight classes
- Minor confusion observed between Late Blight and Healthy leaves due to subtle early-stage symptoms

> Severity estimation is **not treated as a supervised learning task** and therefore no severity accuracy is reported.

---

## ⚠️ Important Notes

- Severity levels are **estimated**, not expert-labeled.
- The CNN is used **only for disease classification**, not for severity prediction.
- Cure recommendations are generated using **rule-based logic**, not learned by the model.

---

## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2 Train the model:
   python train_cnn.py

3 Evaluate on test set:
  python evaluate_model.py

4 Run full pipeline:
  python final_pipeline.py



