import os
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, request

# -----------------------
# Flask setup
# -----------------------
app = Flask(__name__)
app.config["DEBUG"] = False

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------
# Load TFLite model
# -----------------------
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MUST match training order
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -----------------------
# Image preprocessing
# -----------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
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

            # TFLite prediction
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

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

# For Gunicorn
application = app

# -----------------------
# Run locally
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)