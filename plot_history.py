import matplotlib.pyplot as plt
import pickle

# Load training history saved manually
# If you didn't save it, we'll fix that in the next step
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.savefig("accuracy_curve.png")
plt.show()

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("loss_curve.png")
plt.show()
