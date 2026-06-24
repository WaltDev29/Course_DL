import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# utils 폴더 모듈을 import하기 위해 부모 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_and_preprocess_data, build_seq2seq_datasets
from utils.visualization import plot_training_curve, evaluate_and_plot_seq2seq_predictions

def shape_loss(y_true, y_pred):
    """
    MSE와 Temporal Derivative Loss(모양/기울기 오차)를 결합한 커스텀 손실 함수.
    과거 값을 복사하는 지연(Lag) 예측 현상을 강력하게 방지합니다.
    """
    # 1. 값의 차이 (MSE)
    mse = K.mean(K.square(y_true - y_pred))
    
    # 2. 모양의 차이 (기울기/미분값 차이)
    diff_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    diff_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    shape = K.mean(K.square(diff_true - diff_pred))
    
    # Shape loss 비율은 조절 가능 (여기서는 1.0)
    return mse + 1.0 * shape

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    # 1. 데이터 로드
    train_episodes, val_episodes, test_episodes, data_mean, data_std = load_and_preprocess_data(dataset_dir, num_episodes=200)
    
    # 2. 하이퍼파라미터 설정
    sampling_rate = 1
    sequence_length = 300   # 과거 30초 분량
    prediction_steps = 30   # 미래 3초 분량을 전체 시퀀스로 예측
    batch_size = 256
    
    # 3. Keras Seq2Seq Dataset 생성
    train_dataset, val_dataset, test_dataset = build_seq2seq_datasets(
        train_episodes, val_episodes, test_episodes,
        sequence_length, prediction_steps, batch_size, sampling_rate
    )
    
    # 4. Seq2Seq 모델 구조 정의 (Encoder-Decoder)
    inputs = keras.Input(shape=(sequence_length, 6))
    
    # [Encoder] 과거 문맥 압축
    encoder = layers.GRU(128)
    encoder_outputs = encoder(inputs)
    
    # [RepeatVector] 예측할 스텝(30)만큼 압축된 벡터 복제
    x = layers.RepeatVector(prediction_steps)(encoder_outputs)
    
    # [Decoder] 미래 시퀀스 전개 (RepeatVector로 문맥을 매 스텝 전달하므로 initial_state 생략)
    decoder_gru = layers.GRU(128, return_sequences=True)
    x = decoder_gru(x)
    
    # [TimeDistributed Dense] 전개된 각 시점마다 6개 센서값 출력
    outputs = layers.TimeDistributed(layers.Dense(6))(x)
    
    model = keras.Model(inputs, outputs)
    
    # 커스텀 Shape Loss 적용
    model.compile(optimizer="rmsprop", loss=shape_loss, metrics=["mae"])
    
    model_save_path = os.path.join(os.path.dirname(__file__), "seq2seq_gru.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    ]
    
    # 5. 모델 학습
    print("Starting Seq2Seq training...")
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # 6. 모델 평가
    model = keras.models.load_model(model_save_path, custom_objects={'shape_loss': shape_loss})
    test_loss, test_mae = model.evaluate(test_dataset)
    print(f"\n[Test Result]")
    print(f"테스트 Custom Loss: {test_loss:.4f}")
    print(f"테스트 MAE: {test_mae:.4f}")
    
    # 7. 학습 곡선 시각화 및 저장
    plot_path = os.path.join(os.path.dirname(__file__), "seq2seq_loss.png")
    plot_training_curve(history, title="Training and validation MAE (Seq2Seq GRU)", save_path=plot_path)
    
    # 8. Seq2Seq 예측 시각화
    save_prefix = os.path.join(os.path.dirname(__file__), "seq2seq_predictions")
    evaluate_and_plot_seq2seq_predictions(
        model=model,
        test_episodes=test_episodes,
        sequence_length=sequence_length,
        prediction_steps=prediction_steps,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        data_std=data_std,
        data_mean=data_mean,
        save_prefix=save_prefix
    )

if __name__ == "__main__":
    main()
