# 입력과 출력 처리를 위한 함수들

import numpy as np
from PIL import Image, ImageOps

# 입력된 이미지를 전처리
def preprocess_image(file_obj):
    img = Image.open(file_obj).convert("L") # 그레이스케일로 변환
    img = ImageOps.invert(img)  # 이미지의 색상을 반전 (흰 바탕에 검은 글씨로)
    
    img = ImageOps.pad( # 패딩 함수
        img,
        (28, 28),   # 28*28 사이즈로 Reshape
        method=Image.BILINEAR,  # 이미지 보간 방식 설정
        color=255,  # 흰색으로 패딩
        centering=(0.5, 0.5)    # 0.5, 0.5 (정가운데) 위치를 중심으로 28*28 사이즈로 보간
    )

    arr = np.array(img).astype("float32") / 255.0   # 정규화
    arr = arr.reshape(1, 28, 28, 1)     # 입력 형식에 맞게 Reshape

    return arr



# 모델 출력을 후처리
def postprocess(pred):
    prob = float(pred.max())    # 일치 확률
    cls = int(pred.argmax())    # 숫자
    return {"digit": cls, "prob": prob}