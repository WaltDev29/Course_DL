import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 한글 폰트 설정 (Windows 환경)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    features = [
        ('Pressure_1', 'Pressure_1_data_set_{id}.csv'),
        ('Pressure_2', 'Pressure_2_data_set_{id}.csv'),
        ('Temperature_1', 'Temperature_1_data_set_{id}.csv'),
        ('Temperature_2', 'Temperature_2_data_set_{id}.csv'),
        ('Gas_Leak', 'Gas_Leak_data_set_{id}.csv'),
        ('Accelerometer', 'Accelerometer_data_set_{id}.csv')
    ]
    
    print("Loading 1000 episodes for EDA... (This may take around 30 seconds)")
    
    all_data = {f[0]: [] for f in features}
    
    # 1000개 에피소드 전체 로드
    num_episodes = 1000
    for i in range(num_episodes):
        episode_id = f"{i:05d}"
        for feature_name, file_pattern in features:
            file_path = os.path.join(dataset_dir, feature_name, file_pattern.format(id=episode_id))
            try:
                df = pd.read_csv(file_path, header=None)
                all_data[feature_name].append(df.iloc[:, 1].values)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    # 센서별로 1000개 에피소드 데이터 병합 (각 센서당 3,000,000 rows)
    merged_data = {}
    for feature_name in all_data:
        merged_data[feature_name] = np.concatenate(all_data[feature_name])
        
    df_all = pd.DataFrame(merged_data)
    
    # 1. 기초 통계량 (전체 1000개 에피소드 통합)
    print("\n" + "="*50)
    print("[1. 전체 데이터 기초 통계량 요약]")
    stats = df_all.describe()
    print(stats)
    print("="*50)
    
    # 2-1. 전체 데이터 시계열 추세 시각화 - 서브플롯 (단일 에피소드 00000 기준)
    print("\n[2. 시계열 트렌드 시각화 생성 중...]")
    # 주의: 300만 개의 포인트를 한 번에 선 그래프로 그리면 노이즈만 보이기 때문에,
    # 트렌드 파악은 대표 에피소드 1개(3000 스텝)만 샘플링하여 시각화합니다.
    df_ep0 = df_all.iloc[:3000].copy()
    
    colors = sns.color_palette('tab10', 6)
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df_ep0.columns):
        plt.subplot(6, 1, i + 1)
        plt.plot(df_ep0[col], label=col, color=colors[i])
        plt.legend(loc='upper right')
    plt.suptitle('단일 에피소드(00000) 시계열 트렌드 - 서브플롯 분리', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'eda_trend_subplots.png'))
    plt.close()
    print("-> 'eda_trend_subplots.png' 저장 완료.")

    # 2-2. 전체 데이터 시계열 추세 시각화 - 정규화 후 단일 피규어 중첩
    
    # 정규화 (표준화, Standardization) 적용
    # [이유] 
    # 1) LSTM 모델 학습용 전처리(data_loader.py)에서 평균 0, 분산 1로 맞추는 표준화를 사용하고 있으므로, 
    #    모델이 실제로 학습하게 될 형태의 데이터 스케일을 시각화하는 것이 타당합니다.
    # 2) 가스 누출은 1000단위, 가속도는 0.1단위로 스케일 차이가 매우 큰데, 
    #    표준화를 적용하면 모든 변수의 평균이 0에 맞춰져서 변동 폭과 흐름(Trend)의 일치 여부를 한 피규어 안에서 쉽게 비교할 수 있습니다.
    df_ep0_scaled = (df_ep0 - df_ep0.mean()) / df_ep0.std()
    
    plt.figure(figsize=(15, 6))
    for i, col in enumerate(df_ep0_scaled.columns):
        plt.plot(df_ep0_scaled[col], label=col, color=colors[i], linewidth=1.5)
    plt.title('단일 에피소드(00000) 정규화(Standardization) 시계열 트렌드 - 단일 피규어 중첩')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'eda_trend.png'))
    plt.close()
    print("-> 'eda_trend.png' 저장 완료.")
    
    # 3. 이상치 확인 및 분포 (전체 1000개 에피소드 통합)
    print("\n[3. 박스플롯 및 히스토그램 시각화 생성 중...]")
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(df_all.columns):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(y=df_all[col], color=colors[i])
        plt.title(col)
    plt.suptitle('전체 1000개 에피소드 센서별 박스플롯 (독립 스케일)', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'eda_boxplot.png'))
    plt.close()
    print("-> 'eda_boxplot.png' 저장 완료.")
    
    df_all.hist(bins=50, figsize=(15, 10), color='skyblue')
    plt.suptitle('전체 1000개 에피소드 분포 (Histogram)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'eda_histogram.png'))
    plt.close()
    print("-> 'eda_histogram.png' 저장 완료.")
    
    # 4. 속성 간 상관관계 분석 (전체 1000개 에피소드 통합)
    print("\n[4. 속성 간 상관관계 히트맵 생성 중...]")
    plt.figure(figsize=(8, 6))
    corr = df_all.corr()
    # 상관관계의 이론적 범위인 -1 ~ 1로 컬러바(Color bar) 범위를 고정합니다.
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('전체 센서 속성 간 상관관계 (Correlation Heatmap)')
    plt.savefig(os.path.join(base_dir, 'eda_correlation.png'))
    plt.close()
    print("-> 'eda_correlation.png' 저장 완료.")
    
    print("\n[종료] 모든 EDA 분석이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()
