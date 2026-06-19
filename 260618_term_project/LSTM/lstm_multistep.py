import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# utils 폴더 모듈을 import하기 위해 부모 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_and_preprocess_data, create_tf_dataset_multi_step

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    # 1. 데이터 로드 (초기 테스트용으로 10개 에피소드만 사용)
    train_episodes, val_episodes, test_episodes, data_mean, data_std = load_and_preprocess_data(dataset_dir, num_episodes=10)
    
    # 2. 하이퍼파라미터 설정
    sampling_rate = 10     # 1s 마다 샘플링
    sequence_length = 60   # 과거 60초(60 time steps * 1s)를 보고 예측
    delays = [3, 9, 30, 60, 120]  # 3s, 9s, 30s, 60s, 120s 후 예측 (sampling_rate=10 기준)
    target_feature_idx = 4 # Gas Leak 피처의 인덱스
    batch_size = 256
    
    # 3. Keras Dataset 생성
    print("Creating Multi-step tf.data.Datasets...")
    train_dataset = create_tf_dataset_multi_step(train_episodes, sequence_length, delays, target_feature_idx, batch_size, shuffle=True, sampling_rate=sampling_rate)
    val_dataset = create_tf_dataset_multi_step(val_episodes, sequence_length, delays, target_feature_idx, batch_size, shuffle=False, sampling_rate=sampling_rate)
    test_dataset = create_tf_dataset_multi_step(test_episodes, sequence_length, delays, target_feature_idx, batch_size, shuffle=False, sampling_rate=sampling_rate)
    
    # 4. 모델 구조 정의 (Keras)
    # 센서 특성이 6개이므로 shape=(sequence_length, 6)
    inputs = keras.Input(shape=(sequence_length, 6))
    x = layers.LSTM(16)(inputs)
    # 출력값은 지정된 delays(5개 시점)의 Gas Leak 값 하나이므로 Dense(len(delays))
    outputs = layers.Dense(len(delays))(x)
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    
    model_save_path = os.path.join(os.path.dirname(__file__), "lstm_multistep.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    ]
    
    # 5. 모델 학습
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=10,  # 초기 테스트이므로 10번만 학습
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
    loss = history.history["mae"]
    val_loss = history.history["val_mae"]
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training MAE")
    plt.plot(epochs, val_loss, "b", label="Validation MAE")
    plt.title("Training and validation MAE (LSTM Multi-step)")
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(__file__), "lstm_multistep_loss.png")
    plt.savefig(plot_path)
    print(f"\n학습 곡선이 {plot_path}에 저장되었습니다.")
    
    # 8. 예측 vs 실제 값 비교 시각화 (테스트 세트의 첫 번째 에피소드 전체 사용)
    print("\n[8. 첫 번째 테스트 에피소드 전체 예측 vs 실제 값 시각화 생성 중...]")
    single_test_ep = [test_episodes[0]]
    single_test_dataset = create_tf_dataset_multi_step(single_test_ep, sequence_length, delays, target_feature_idx, batch_size, shuffle=False, sampling_rate=sampling_rate)
    
    # 예측 수행
    predictions = model.predict(single_test_dataset)
    
    # 실제 타겟 값 추출
    actual_targets = []
    for _, targets in single_test_dataset:
        actual_targets.append(targets.numpy())
    actual_targets = np.concatenate(actual_targets, axis=0)
    
    # 역정규화 (Inverse Transform) 적용하여 원래 스케일로 복원 (Gas Leak의 통계량만 사용)
    gas_mean = data_mean[target_feature_idx]
    gas_std = data_std[target_feature_idx]
    
    predictions_original = predictions * gas_std + gas_mean
    actual_targets_original = actual_targets * gas_std + gas_mean
    
    delay_labels = ['3s', '9s', '30s', '60s', '120s']
    time_axis = np.arange(len(actual_targets_original)) * 0.1
    
    plot_configs = [((15, 12), "_short"), ((15, 20), "_tall")]
    
    for figsize, suffix in plot_configs:
        plt.figure(figsize=figsize)
        for i in range(len(delays)):
            plt.subplot(len(delays), 1, i + 1)
            plt.plot(time_axis, actual_targets_original[:, i], label='Actual', color='b', alpha=0.7)
            plt.plot(time_axis, predictions_original[:, i], label='Predicted', color='r', linestyle='--', alpha=0.7)
            plt.title(f'Gas Leak Prediction: {delay_labels[i]} Ahead (Original Scale)')
            plt.ylabel("Gas Leak Value")
            if i == len(delays) - 1:
                plt.xlabel("Time (seconds)")
            plt.legend(loc='upper right')
            
        plt.suptitle('Gas Leak Multi-step Prediction vs Actual on Test Episode 1', y=1.02)
        plt.tight_layout()
        pred_plot_path = os.path.join(os.path.dirname(__file__), f"lstm_multistep_predictions{suffix}.png")
        plt.savefig(pred_plot_path)
        plt.close()
        print(f"예측 비교 그래프({suffix})가 {pred_plot_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()
