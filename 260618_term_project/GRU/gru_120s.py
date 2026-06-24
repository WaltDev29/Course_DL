import os
import sys
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# utils 폴더 모듈을 import하기 위해 부모 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_and_preprocess_data, create_tf_dataset, build_datasets
from utils.visualization import plot_training_curve, evaluate_and_plot_predictions

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    # 1. 데이터 로드 (초기 테스트용으로 10개 에피소드만 사용)
    train_episodes, val_episodes, test_episodes, data_mean, data_std = load_and_preprocess_data(dataset_dir, num_episodes=300)
    
    # 2. 하이퍼파라미터 설정
    sampling_rate = 1      # 1초 간격으로 샘플링 (원본은 100ms 간격)
    sequence_length = 2000  # 과거 120초(1200 time steps)를 보고 예측
    prediction_steps = 1200
    batch_size = 256
    
    # 3. Keras Dataset 생성
    print("Creating tf.data.Datasets...")
    train_dataset, val_dataset, test_dataset, delay = build_datasets(
        train_episodes, val_episodes, test_episodes,
        sampling_rate, sequence_length, prediction_steps, batch_size
    )
    
    # 4. 모델 구조 정의 (Keras)
    # 센서 특성이 6개이므로 shape=(sequence_length, 6)
    inputs = keras.Input(shape=(sequence_length, 6))
    x = layers.GRU(64)(inputs)
    # 출력값 역시 6개 특성이어야 하므로 Dense(6)
    outputs = layers.Dense(6)(x)
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    
    model_save_path = os.path.join(os.path.dirname(__file__), "gru_120s.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    ]
    
    # 5. 모델 학습
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=30,  # 초기 테스트이므로 10번만 학습
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # 6. 모델 평가
    model = keras.models.load_model(model_save_path)
    test_loss, test_mae = model.evaluate(test_dataset)
    print(f"\n[Test Result]")
    print(f"테스트 MSE: {test_loss:.4f}")
    print(f"테스트 MAE: {test_mae:.4f}")
    
    # 7. 학습 곡선 시각화 및 저장
    plot_path = os.path.join(os.path.dirname(__file__), "gru_120s_loss.png")
    plot_training_curve(history, title="Training and validation MAE (GRU 120s)", save_path=plot_path)
    
    # 8. 예측 vs 실제 값 비교 시각화 (테스트 세트의 첫 번째 에피소드 전체 사용)
    print("\n[8. 첫 번째 테스트 에피소드 전체(약 300초) 예측 vs 실제 값 시각화 생성 중...]")
    save_prefix = os.path.join(os.path.dirname(__file__), "gru_120s_predictions")
    evaluate_and_plot_predictions(
        model=model,
        test_episodes=test_episodes,
        sequence_length=sequence_length,
        delay=delay,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        data_std=data_std,
        data_mean=data_mean,
        save_prefix=save_prefix
    )

if __name__ == "__main__":
    main()
