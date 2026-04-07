'''
2026.04.02
Mnist 데이터 학습
'''

import tensorflow as tf
from tensorflow import keras
from keras import datasets
from keras import Input
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import random
import os


# 재현성을 위해 시드 고정
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



# ============ 모델 생성 함수 ============
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



# ============ History 그래프 표시 함수 ============
def plot_history(history, out_path="history.png"):
    plt.figure(figsize=(10,5))

    # Loss 
    plt.subplot(1,2,1)
    plt.title("Loss")
    plt.plot(history.history["loss"], c="blue", label="Train")
    plt.plot(history.history["val_loss"], c="red", label="Validation")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy 
    plt.subplot(1,2,2)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], c="blue", label="Train")
    plt.plot(history.history["val_accuracy"], c="red", label="Validation")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()


    # Save & Show 
    plt.savefig(out_path)
    print(f"Training history plot saved to {out_path}")

    plt.tight_layout()
    plt.show()



# ============ main 함수 ============
def main():
    # 데이터 로드 
    (train_data, train_label), (test_data, test_label) = datasets.mnist.load_data()

    # 데이터 전처리 
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # 모델 정의 
    model = build_model()

    # 모델 저장 경로 설정
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.weights.h5")


    # 모델 학습 콜백 함수 정의
    callbacks = [
        # 모델 저장 콜백
        keras.callbacks.ModelCheckpoint( 
            filepath=ckpt_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
            ),
        # 모델 학습 조기 종료 콜백
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=3,
            restore_best_weights=True 
        )
    ]


    # 모델 학습 
    history = model.fit(
        train_data,
        train_label,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=2
        )

    # 모델 학습 결과 시각화 
    plot_history(history)

    # 모델 추론 
    result = model.predict(test_data)


    # 모델 성능 비교 
    # 조기종료 함수에 restore_best_weights=True 옵션을 주었기 때문에 두 모델은 같은 가중치를 가지고 있음
    last_loss, last_acc = model.evaluate(test_data, test_label, verbose=0)
    print(f"[Last epoch model] Test accuracy: {last_acc:.4f}, loss: {last_loss:.4f}")

    model.load_weights(ckpt_path)

    best_loss, best_acc = model.evaluate(test_data, test_label, verbose=0)
    print(f"[Best checkpoint] Test accuracy: {best_acc:.4f}, loss: {best_loss:.4f}")
    
    os.makedirs("model", exist_ok=True)
    model.save("model/model.weights.h5") # SavedModel 형식으로 ./model 에 저장
    print("Best model (weights) saved to ./model (SavedModel format)")



if __name__ == "__main__":
    main()