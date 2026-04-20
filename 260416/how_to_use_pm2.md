## 패키지 설치
```
apt-get install -y curl ca-certificates gnupg
```

## nodejs 설치 파일 받기
```
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
```

## nodejs 설치
```
apt-get install -y nodejs
```

## pm2 설치
```
npm i -g pm2
```

## pm2로 comfyui 실행
```
pm2 start main.py --name myapp --interpreter python3 -- --listen 0.0.0.0 --port 8188
```


## 로그 확인
```
pm2 logs
```

## 프로세스 목록 확인
```
pm2 list
```

## 프로세스 저장
```
pm2 save
```

## 저장한 프로세스 시작
```
pm2 resurrect
```

## 프로세스 시작
```
pm2 start [name or ID]
```

## 프로세스 종료
```
pm2 stop [name or ID]
```