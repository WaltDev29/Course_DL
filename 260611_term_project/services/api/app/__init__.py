# app/__init__.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import generate

def create_app() -> FastAPI:
    app = FastAPI(title="ComfyUI Studio API")

    # 프론트 서버에서 요청을 받을 수 있도록 CORS 허용
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- 라우터 등록 ----------
    app.include_router(generate.router)

    return app