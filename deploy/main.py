import io
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model = keras.models.load_model("./model/serena-emotion-detector.h5", compile=False)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def transform_image(image):
    # TODO: buat fungsi untuk transformasi gambar sesuai dengan yang dilakukan di training untuk bisa digunakan dalam model.predict
    # Turn to grayscale
    transformed_image = image.convert("L")

    # This one causes every class to be around 14% predicted
    # transformed_image = np.asarray(transformed_image)
    # transformed_image = transformed_image / 255
    # transformed_image = transformed_image[np.newaxis, ..., np.newaxis]
    # transformed_image = tf.image.resize(transformed_image, [48, 48])

    # Resize to 48x48
    transformed_image = transformed_image.resize((48, 48))
    # Reshape to np array
    transformed_image = np.reshape(transformed_image, (1, 48, 48, 1))
    return transformed_image


# Function to retrieve the label corresponding to an emotion
def get_emotion_label(emotion_code):
    # Emotion labels mapping
    emotion_labels = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    }
    return emotion_labels.get(emotion_code, "Invalid emotion")


def predict(image):
    # TODO: predict menggunakan model yang sudah diload di atas dan return hasil prediksi berupa desimal dari 7 emosi + labelnya dalam bentuk Dictionary
    # Predict
    prediction = model.predict(image)
    # Get labels
    result = {}
    for i, label in enumerate(labels):
        percentage = prediction[0][i] * 100
        result[label.lower()] = percentage

    return result


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"message": "Missing image", "status": 400})

        try:
            image_bytes = file.read()
            user_photo = Image.open(io.BytesIO(image_bytes))
            transformed_image = transform_image(user_photo)
            prediction = predict(transformed_image)
            return prediction
        except Exception as e:
            return jsonify({"message": str(e), "status": 500})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
