import os
import io
from tensorflow import keras
from PIL import Image
from flask import Flask, request, jsonify

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model = keras.models.load_model(
    "./model/models_model_CNN_facial_emotion_detector.h5", compile=False
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


def transform_image(image):
    # TODO: buat fungsi untuk transformasi gambar sesuai dengan yang dilakukan di training untuk bisa digunakan dalam model.predict
    pass


def predict(image):
    # TODO: predict menggunakan model yang sudah diload di atas dan return hasil prediksi berupa desimal dari 7 emosi + labelnya dalam bentuk Dictionary

    result = {
        # TODO Ubah nilai-nilai di bawah ini dengan hasil prediksi dari model
        "angry": 0,
        "fear": 0,
        "surprise": 0,
        "disgust": 0,
        "happy": 0,
        "neutral": 0,
        "sad": 0,
    }
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
            tensor = transform_image(user_photo)
            prediction = predict(tensor)
            return prediction
        except Exception as e:
            return jsonify({"message": str(e), "status": 500})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
