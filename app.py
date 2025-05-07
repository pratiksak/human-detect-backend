from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

def detect_human(image_np):
    # Use a pre-trained Haar cascade for simplicity
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })

    return boxes, len(boxes) > 0

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        image_data = data.get("image").split(",")[1]  # Remove base64 header
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        boxes, human_detected = detect_human(image_np)

        return jsonify({
            "boxes": boxes,
            "human_detected": human_detected
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
