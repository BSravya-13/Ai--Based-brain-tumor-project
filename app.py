import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------
# Flask setup
# -----------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
from flask import send_from_directory

from flask import send_from_directory

@app.route('/bg/<path:filename>')
def bg_image(filename):
    return send_from_directory(
        r"C:\Users\sravy\OneDrive\Desktop\ML Project",
        filename
    )



# -----------------------
# Load trained model
# -----------------------
model = tf.keras.models.load_model("brain_tumor_model.keras")

# MUST match training order
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -----------------------
# Image preprocessing
# -----------------------
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------
# Risk logic
# -----------------------
def get_risk_level(label, confidence):
    if label == "notumor":
        return "LOW"
    elif confidence >= 90:
        return "HIGH"
    else:
        return "MEDIUM"

# -----------------------
# AI decision logic
# -----------------------
def ai_decision(confidence):
    if confidence >= 90:
        return "AI CONFIDENT – Auto Decision"
    elif confidence >= 70:
        return "AI MODERATE – Doctor Review Suggested"
    else:
        return "AI NOT CONFIDENT – Manual Review Required"

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = risk = decision = image_url = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            img = preprocess_image(file_path)
            preds = model.predict(img)[0]

            idx = np.argmax(preds)
            result = CLASS_NAMES[idx]
            confidence = round(float(preds[idx]) * 100, 2)

            risk = get_risk_level(result, confidence)
            decision = ai_decision(confidence)

            image_url = file_path

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        risk=risk,
        decision=decision,
        file_path=image_url
    )

# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
