import os

import cv2
import numpy as np
from flask import Flask, jsonify, request
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model = keras.models.load_model(
    os.path.join(".", "model", "serena-emotion-detector.keras")
)
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
img_size = 224


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = image[y : y + h, x : x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        faces = faceCascade.detectMultiScale(roi_gray)
        if len(faces) == 0:
            raise ValueError("Face not detected")
        else:
            for ex, ey, ew, eh in faces:
                face_roi = roi_color[ey : ey + eh, ex : ex + ew]

    final_image = cv2.resize(face_roi, (img_size, img_size))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0

    return final_image


def predict(normalized_image):
    predictions = model.predict(normalized_image)

    result = {}
    for i, label in enumerate(classes):
        percentage = predictions[0][i] * 100
        result[label.lower()] = percentage

    return result


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            print("No file uploaded.")
            return jsonify({"message": "Missing image", "status": 400})

        try:
            filestr = request.files["file"].read()
            file_bytes = np.frombuffer(filestr, np.uint8)
            user_photo = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            processed_image = preprocess_image(user_photo)
            prediction = predict(processed_image)
            return prediction
        except Exception as e:
            print(e)
            return jsonify({"message": str(e), "status": 500})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
