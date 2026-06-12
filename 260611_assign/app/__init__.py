from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import chat_router

def create_app() -> FastAPI:
    app = FastAPI(title="Chatbot with Session Memory")

    # CORS 미들웨어 설정 (필요시)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(chat_router.router)

    return app
