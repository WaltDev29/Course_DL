# 베이스 이미지
FROM python:3.12-slim

# APT 패키지 설치
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# ComfyUI 클론
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# 작업 디렉토리 이동
WORKDIR /ComfyUI

# 파이썬 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 포트 오픈
EXPOSE 8188

# 실행
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]