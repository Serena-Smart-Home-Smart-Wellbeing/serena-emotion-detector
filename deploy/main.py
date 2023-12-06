import io
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Yang ini lebih berat angry
# model = keras.models.load_model("./model/serena-emotion-detector.h5", compile=False)

# Yang ini lebih rata angry, fear, & neutral
model = keras.models.load_model(
    "./model/models_model_64accuray_angrycorrect.h5", compile=False
)
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
    # model.summary()
    app.run(debug=True)

# Model file models_model_64accuray_angrycorrect.h5
# {
#     "angry": 29.651018977165222,
#     "disgust": 10.591613501310349,
#     "fear": 28.091728687286377,
#     "happy": 2.5392208248376846,
#     "neutral": 23.156139254570007,
#     "sad": 0.6641537882387638,
#     "surprise": 5.3061265498399734,
# }

# Model file serena-emotion-detector.h5
# {
#     "angry": 66.56327843666077,
#     "disgust": 8.644931018352509,
#     "fear": 8.207805454730988,
#     "happy": 0.49696043133735657,
#     "neutral": 7.5864724814891815,
#     "sad": 7.924692332744598,
#     "surprise": 0.5758598912507296,
# }
