def recommend_cure(disease, severity):
    if disease == "Potato___Early_blight":
        if severity == "Mild":
            return "Remove infected leaves and improve air circulation."
        elif severity == "Moderate":
            return "Apply recommended fungicide such as Mancozeb."
        else:
            return "Remove severely infected plants and apply fungicide."

    elif disease == "Potato___Late_blight":
        if severity == "Mild":
            return "Apply preventive fungicide and reduce moisture."
        elif severity == "Moderate":
            return "Apply systemic fungicide immediately."
        else:
            return "Destroy infected plants and sanitize field."

    else:
        return "No treatment required. Plant is healthy."
