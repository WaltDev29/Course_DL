FROM python:3.12-slim

WORKDIR /app

# 의존성 파일만 먼저 복사 (레이어 캐싱)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
