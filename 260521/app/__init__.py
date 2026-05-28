from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

def create_app():
    app = FastAPI()

    from .routes import router
    app.include_router(router)

    @app.get("/")
    def index():
        template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

    return app