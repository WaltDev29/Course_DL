import os
import matplotlib.pyplot as plt

def plot_training_curve(history, title, save_path):
    """
    모델의 학습 과정(MAE)을 그래프로 시각화하여 파일로 저장합니다.
    """
    mae = history.history["mae"]
    val_mae = history.history["val_mae"]
    epochs = range(1, len(mae) + 1)
    
    plt.figure()
    plt.plot(epochs, mae, "bo", label="Training MAE")
    plt.plot(epochs, val_mae, "b", label="Validation MAE")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()
    print(f"\n학습 곡선이 {save_path}에 저장되었습니다.")

def evaluate_and_plot_predictions(model, test_episodes, sequence_length, delay, batch_size, sampling_rate, data_std, data_mean, save_prefix):
    """
    첫 번째 테스트 에피소드에 대한 예측을 수행하고 실제 값과 비교하는 그래프를 생성합니다.
    """
    from utils.data_loader import create_tf_dataset
    import numpy as np

    print("\n[8. 첫 번째 테스트 에피소드 전체(약 300초) 예측 vs 실제 값 시각화 생성 중...]")
    # 첫 번째 테스트 에피소드 전용 데이터셋 생성
    single_test_ep = [test_episodes[0]]
    single_test_dataset = create_tf_dataset(single_test_ep, sequence_length, delay, batch_size, shuffle=False, sampling_rate=sampling_rate)
    
    # 해당 에피소드 전체에 대한 예측 수행
    predictions = model.predict(single_test_dataset)
    
    # 실제 타겟 값 추출
    actual_targets = []
    for _, targets in single_test_dataset:
        actual_targets.append(targets.numpy())
    actual_targets = np.concatenate(actual_targets, axis=0)
    
    # 역정규화 (Inverse Transform) 적용하여 원래 스케일로 복원
    predictions_original = predictions * data_std + data_mean
    actual_targets_original = actual_targets * data_std + data_mean
    
    feature_names = ['Pressure_1', 'Pressure_2', 'Temperature_1', 'Temperature_2', 'Gas_Leak', 'Accelerometer']
    
    # x축을 초(Seconds) 단위로 변환 (1 step = 100ms = 0.1s)
    time_axis = np.arange(len(actual_targets_original)) * 0.1
    
    # short 버전 생략, (15, 24) 크기로 하나만 그리기
    plt.figure(figsize=(15, 24))
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(time_axis, actual_targets_original[:, i], label='Actual', color='#1f77b4', alpha=0.7)
        plt.plot(time_axis, predictions_original[:, i], label='Predicted', color='#ff7f0e', linestyle='--', alpha=0.7)
        plt.title(f'{feature_names[i]} (Original Scale)')
        plt.ylabel("Value")
        if i == 5:
            plt.xlabel("Time (seconds)")
        plt.legend(loc='upper right')
        
    plt.suptitle('Prediction vs Actual on Test Episode 1 (Full 300s)', y=1.02)
    plt.tight_layout()
    pred_plot_path = f"{save_prefix}.png"
    plt.savefig(pred_plot_path)
    plt.close()
    print(f"예측 비교 그래프가 {pred_plot_path}에 저장되었습니다.")

def evaluate_and_plot_seq2seq_predictions(model, test_episodes, sequence_length, prediction_steps, batch_size, sampling_rate, data_std, data_mean, save_prefix):
    """
    Seq2Seq 모델의 첫 번째 테스트 에피소드에 대한 시퀀스 예측 시각화.
    과거 데이터를 복사(Lag)하는지, 미래 추세를 잘 그리는지 확인하기 위해 
    일정 간격마다 예측된 시퀀스(선)를 원본 실제 데이터 위에 겹쳐서 그립니다.
    """
    from utils.data_loader import create_tf_seq2seq_dataset
    import numpy as np
    
    print("\n[8. 첫 번째 테스트 에피소드 전체 예측 vs 실제 값(Seq2Seq) 시각화 생성 중...]")
    single_test_ep = [test_episodes[0]]
    single_test_dataset = create_tf_seq2seq_dataset(single_test_ep, sequence_length, prediction_steps, batch_size, shuffle=False, sampling_rate=sampling_rate)
    
    predictions = model.predict(single_test_dataset)
    
    actual_targets = []
    for _, targets in single_test_dataset:
        actual_targets.append(targets.numpy())
    actual_targets = np.concatenate(actual_targets, axis=0)
    
    # 원본 시계열(Actual)을 하나의 긴 선으로 복원
    full_actual = np.concatenate([actual_targets[:, 0, :], actual_targets[-1, 1:, :]], axis=0)
    full_actual_original = full_actual * data_std + data_mean
    
    predictions_original = predictions * data_std + data_mean
    
    feature_names = ['Pressure_1', 'Pressure_2', 'Temperature_1', 'Temperature_2', 'Gas_Leak', 'Accelerometer']
    time_axis_full = np.arange(len(full_actual_original)) * 0.1
    
    plt.figure(figsize=(15, 24))
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        # 실제 전체 데이터
        plt.plot(time_axis_full, full_actual_original[:, i], label='Actual', color='#1f77b4', alpha=0.4, linewidth=1)
        
        # 예측된 시퀀스를 너무 빽빽하지 않게 간격을 두고(prediction_steps) 겹쳐 그립니다.
        for step in range(0, len(predictions_original), prediction_steps):
            pred_seq = predictions_original[step, :, i]
            time_seq = np.arange(step, step + prediction_steps) * 0.1
            label = 'Predicted Seq' if step == 0 else ""
            plt.plot(time_seq, pred_seq, color='#ff7f0e', alpha=0.9, linewidth=2, label=label)
            
        plt.title(f'{feature_names[i]} (Seq2Seq Original Scale)')
        plt.ylabel("Value")
        if i == 5:
            plt.xlabel("Time (seconds)")
        plt.legend(loc='upper right')
        
    plt.suptitle(f'Seq2Seq Prediction vs Actual on Test Episode 1 (Predicted 30 steps)', y=1.02)
    plt.tight_layout()
    pred_plot_path = f"{save_prefix}.png"
    plt.savefig(pred_plot_path)
    plt.close()
    print(f"Seq2Seq 예측 비교 그래프가 {pred_plot_path}에 저장되었습니다.")
