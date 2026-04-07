from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_app():
    app = FastAPI()

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    from .route import router
    app.include_router(router)

    return app