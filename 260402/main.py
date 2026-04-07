import os
import io
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from utils import preprocess_image, postprocess


app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.weights.h5")
print(f"Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
_ = model.predict(np.zeros((1,28,28,1), dtype=np.float32))



@app.route("/predict", methods=["POST"])
def predict():
    """
    multipart/form-data: file 필드로 이미지 업로드
    """
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    

    f = request.files["file"]
    arr = preprocess_image(f)
    pred = model.predict(arr, verbose=0)
    result = postprocess(pred[0])

    return jsonify(result)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)