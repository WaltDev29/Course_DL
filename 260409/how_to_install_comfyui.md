## Docker 환경 구축
```
docker run -it -d --gpus all -p 8188:8188 -v D:\course\DL:/app --name comfyui python:3.12-slim bash
```

## 의존성 패키지 설치
```
pip install --no-cache-dir -r requirements.txt
```

## APT 패키지 설치
```
apt update && apt install git -y && apt install wget -y
```

## GIT Clone
```
git clone https://github.com/comfyanonymous/ComfyUI.git
```

## 의존성 패키지 실행
```
cd ComfyUI
pip install -r requirements.txt
```

## ComfyUI 실행
```
python main.py --listen 0.0.0.0 --port 8188
```

## Web에서 확인
```
http://localhost:8188
```