import cv2
import numpy as np

def estimate_severity(image_path):
    """
    Estimates disease severity based on infected leaf area percentage.
    Returns infected percentage and severity label.
    """

    # Read image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for diseased (brown/necrotic) regions
    lower = np.array([10, 40, 40])
    upper = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    infected_pixels = cv2.countNonZero(mask)
    total_pixels = 224 * 224

    infected_percentage = (infected_pixels / total_pixels) * 100

    # Severity rules
    if infected_percentage < 10:
        severity = "Mild"
    elif infected_percentage < 30:
        severity = "Moderate"
    else:
        severity = "Severe"

    return infected_percentage, severity
if __name__ == "__main__":
    test_image = "dataset_processed/test/Potato___Late_blight/sample.jpg"

    percentage, severity = estimate_severity(test_image)

    print(f"Infected Area: {percentage:.2f}%")
    print(f"Estimated Severity: {severity}")

