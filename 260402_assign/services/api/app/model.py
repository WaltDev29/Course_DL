import tensorflow as tf
import numpy as np
import os

# 모델 로드
class Model:
    def __init__(self):
        MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.weights.h5")
        print(f"Loading model from {MODEL_PATH} ...")

        self.model = tf.keras.models.load_model(MODEL_PATH)
        _ = self.model.predict(np.zeros((1, 28, 28, 1), dtype=np.float32))

model = Model().model