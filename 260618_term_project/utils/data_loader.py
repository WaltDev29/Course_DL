import os
import numpy as np
import pandas as pd
from tensorflow import keras

# 6개의 데이터 폴더 및 파일명 패턴 정의
FEATURES = [
    ('Pressure_1', 'Pressure_1_data_set_{id}.csv'),
    ('Pressure_2', 'Pressure_2_data_set_{id}.csv'),
    ('Temperature_1', 'Temperature_1_data_set_{id}.csv'),
    ('Temperature_2', 'Temperature_2_data_set_{id}.csv'),
    ('Gas_Leak', 'Gas_Leak_data_set_{id}.csv'),
    ('Accelerometer', 'Accelerometer_data_set_{id}.csv')
]

def load_and_preprocess_data(dataset_dir, num_episodes=10):
    """
    num_episodes 수만큼 에피소드를 로드하고, 학습/검증/테스트 데이터로 분리 후 정규화를 수행합니다.
    """
    print(f"Loading {num_episodes} episodes from {dataset_dir}...")
    raw_data_list = []
    
    for i in range(num_episodes):
        episode_id = f"{i:05d}"
        episode_data = []
        
        for folder, file_pattern in FEATURES:
            file_path = os.path.join(dataset_dir, folder, file_pattern.format(id=episode_id))
            # 1번 열의 실제 센서값 추출
            df = pd.read_csv(file_path, header=None)
            episode_data.append(df.iloc[:, 1].values)
            
        # (3000,) 형태의 6개 배열을 가로로 병합하여 (3000, 6) 형태로 만듦.
        episode_data = np.column_stack(episode_data)
        raw_data_list.append(episode_data)
        
    # 에피소드 단위로 분할 (Train 50%, Val 25%, Test 25%)
    num_train = int(0.5 * num_episodes)
    num_val = int(0.25 * num_episodes)
    
    train_episodes = raw_data_list[:num_train]
    val_episodes = raw_data_list[num_train:num_train+num_val]
    test_episodes = raw_data_list[num_train+num_val:]
    
    # 데이터 표준화 (데이터 누수 방지를 위해 Train 데이터의 mean, std만 사용)
    train_concat = np.concatenate(train_episodes, axis=0)
    mean = train_concat.mean(axis=0)
    std = train_concat.std(axis=0)
    std[std == 0] = 1.0 # 0으로 나누기 방지
    
    train_episodes = [(ep - mean) / std for ep in train_episodes]
    val_episodes = [(ep - mean) / std for ep in val_episodes]
    test_episodes = [(ep - mean) / std for ep in test_episodes]
    
    print(f"Data loading complete. Train: {len(train_episodes)}, Val: {len(val_episodes)}, Test: {len(test_episodes)}")
    return train_episodes, val_episodes, test_episodes, mean, std

def create_tf_dataset(episodes, sequence_length, delay, batch_size, shuffle=True, sampling_rate=1):
    """
    각 에피소드별로 시계열 데이터셋을 생성한 뒤, 하나로 병합합니다.
    """
    combined_dataset = None
    
    for ep in episodes:
        ds = keras.utils.timeseries_dataset_from_array(
            data=ep,
            targets=ep[delay:],
            sampling_rate=sampling_rate,
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        if combined_dataset is None:
            combined_dataset = ds
        else:
            combined_dataset = combined_dataset.concatenate(ds)
            
    return combined_dataset

def create_tf_dataset_multi_step(episodes, sequence_length, delays, feature_index, batch_size, shuffle=True):
    """
    하나의 모델이 지정된 여러 시점(delays)의 특정 피처(feature_index)를 동시에 예측할 수 있도록
    다중 출력(multi-step) 타겟 벡터를 생성하는 데이터셋 로더입니다.
    """
    combined_dataset = None
    max_delay = max(delays)
    
    for ep in episodes:
        # 최대 지연 시간만큼의 미래 데이터가 존재해야 마지막 시퀀스의 타겟 생성이 가능하므로 data 범위를 제한합니다.
        data = ep[:-max_delay]
        
        # 각 시점별(time step) 타겟 생성 (shape: (len(data), len(delays)))
        targets = []
        for j in range(len(data)):
            target_row = [ep[j + d, feature_index] for d in delays]
            targets.append(target_row)
        targets = np.array(targets)
        
        ds = keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=targets,
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        if combined_dataset is None:
            combined_dataset = ds
        else:
            combined_dataset = combined_dataset.concatenate(ds)
            
    return combined_dataset

def build_datasets(train_episodes, val_episodes, test_episodes, sampling_rate, sequence_length, prediction_steps, batch_size):
    """
    주어진 하이퍼파라미터를 사용하여 학습, 검증, 테스트 데이터셋을 한 번에 생성하고 
    계산된 delay 값과 함께 반환합니다.
    """
    delay = sampling_rate * (sequence_length + prediction_steps - 1)
    
    print("Creating tf.data.Datasets...")
    train_dataset = create_tf_dataset(train_episodes, sequence_length, delay, batch_size, shuffle=True, sampling_rate=sampling_rate)
    val_dataset = create_tf_dataset(val_episodes, sequence_length, delay, batch_size, shuffle=False, sampling_rate=sampling_rate)
    test_dataset = create_tf_dataset(test_episodes, sequence_length, delay, batch_size, shuffle=False, sampling_rate=sampling_rate)
    
    return train_dataset, val_dataset, test_dataset, delay

def create_tf_seq2seq_dataset(episodes, sequence_length, prediction_steps, batch_size, shuffle=True, sampling_rate=1):
    """
    Seq2Seq 모델을 위한 데이터셋을 생성합니다.
    입력: 과거 sequence_length 만큼의 데이터
    타겟: 미래 prediction_steps 만큼의 연속된 데이터
    """
    import tensorflow as tf
    combined_dataset = None
    
    for ep in episodes:
        # 입력과 타겟을 합친 전체 윈도우 길이를 구합니다.
        window_size = sequence_length + prediction_steps
        
        ds = keras.utils.timeseries_dataset_from_array(
            data=ep,
            targets=None,
            sequence_length=window_size,
            sampling_rate=sampling_rate,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        # 전체 윈도우를 (입력, 타겟)으로 분리하는 매핑 함수
        def split_window(window):
            inputs = window[:, :sequence_length, :]
            targets = window[:, sequence_length:, :]
            return inputs, targets
            
        ds = ds.map(split_window, num_parallel_calls=tf.data.AUTOTUNE)
        
        if combined_dataset is None:
            combined_dataset = ds
        else:
            combined_dataset = combined_dataset.concatenate(ds)
            
    return combined_dataset

def build_seq2seq_datasets(train_episodes, val_episodes, test_episodes, sequence_length, prediction_steps, batch_size, sampling_rate=1):
    """
    주어진 하이퍼파라미터를 사용하여 Seq2Seq 학습, 검증, 테스트 데이터셋을 한 번에 생성합니다.
    """
    print("Creating Seq2Seq tf.data.Datasets...")
    train_dataset = create_tf_seq2seq_dataset(train_episodes, sequence_length, prediction_steps, batch_size, shuffle=True, sampling_rate=sampling_rate)
    val_dataset = create_tf_seq2seq_dataset(val_episodes, sequence_length, prediction_steps, batch_size, shuffle=False, sampling_rate=sampling_rate)
    test_dataset = create_tf_seq2seq_dataset(test_episodes, sequence_length, prediction_steps, batch_size, shuffle=False, sampling_rate=sampling_rate)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    load_and_preprocess_data()