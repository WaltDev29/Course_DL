import os
import pandas as pd

def check_missing_values(dataset_dir):
    """
    지정된 디렉토리 내의 CSV 파일들을 순회하며 결측치(missing values) 여부를 확인합니다.
    PNG 파일은 무시합니다.
    """
    print(f"[{dataset_dir}] 경로의 데이터 결측치 검사를 시작합니다...\n")
    total_files = 0
    files_with_missing = 0
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.png'):
                continue
            
            if file.endswith('.csv'):
                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, header=None)
                    missing_sum = df.isnull().sum().sum()
                    
                    if missing_sum > 0:
                        files_with_missing += 1
                        print(f"[Warning] 결측치 발견: {file_path} (총 {missing_sum}개의 결측치)")
                except Exception as e:
                    print(f"[Error] 파일을 읽는 중 오류 발생 ({file_path}): {e}")

    print("\n" + "-" * 40)
    print(f"총 검사한 CSV 파일 수: {total_files}")
    print(f"결측치가 포함된 파일 수: {files_with_missing}")
    print("-" * 40)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '../dataset')
    
    if os.path.exists(dataset_path):
        check_missing_values(dataset_path)
    else:
        print(f"[Error] 데이터셋 폴더를 찾을 수 없습니다: {dataset_path}")
