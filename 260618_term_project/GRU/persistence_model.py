import os
import sys
import numpy as np

# utils 폴더 모듈을 import하기 위해 부모 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_and_preprocess_data

def evaluate_persistence(episodes, step):
    total_mae = 0
    total_count = 0
    for ep in episodes:
        if len(ep) <= step:
            continue
        # 퍼시스턴스 예측: 현재 시점의 값을 그대로 step 만큼 미래의 예측값으로 사용
        pred = ep[:-step]
        actual = ep[step:]
        mae = np.mean(np.abs(pred - actual))
        
        total_mae += mae * len(pred)
        total_count += len(pred)
        
    if total_count > 0:
        return total_mae / total_count
    return None

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    # 100개 에피소드 로드 (GRU 모델들과 동일한 정규화 적용)
    train_eps, val_eps, test_eps, _, _ = load_and_preprocess_data(dataset_dir, num_episodes=100)
    
    all_eps = train_eps + val_eps + test_eps
    
    # 100ms 단위 데이터이므로 1초 = 10 스텝
    steps_dict = {
        "3s": 30,
        "9s": 90,
        "30s": 300,
        "60s": 600,
        "120s": 1200
    }
    
    print("\n==================================================")
    print(" Persistence Model Baseline (100 Episodes)")
    print("==================================================")
    for name, step in steps_dict.items():
        mae = evaluate_persistence(all_eps, step)
        mae_str = f"{mae:.4f}" if mae is not None else "N/A"
        
        print(f"[{name:>4} prediction] MAE: {mae_str}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
