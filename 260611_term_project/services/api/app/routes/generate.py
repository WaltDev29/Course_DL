import io
import requests
import time
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.generate import GenerateRequest, GenerateResponse
from app.services.comfyui import (
    translate_prompt,
    load_workflow,
    update_prompts,
    submit_prompt,
    poll_history,
    extract_first_image_url
)
from app.core.config import config

router = APIRouter()

@router.get("/")
def health():
    return {"status": "ok"}

@router.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest):
    """프롬프트를 받아 Ollama로 번역 후 ComfyUI로 이미지를 생성하고 URL을 반환"""
    try:
        # 번역 로직 (Service Layer)
        pos_final, neg_final = translate_prompt(body.pos, body.neg)

        # 워크플로 로드 및 실행 (Service Layer)
        graph = load_workflow(config.WORKFLOW_PATH)
        graph = update_prompts(graph, pos_final, neg_final)
        prompt_id = submit_prompt(graph)
        history_block = poll_history(prompt_id)
        img_url = extract_first_image_url(history_block)
        
        if not img_url:
            raise HTTPException(status_code=504, detail="Image generation timed out")
        return GenerateResponse(img_url=img_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download")
def download(url: str):
    """ComfyUI 이미지를 서버 사이드에서 받아 클라이언트에 전달 (CORS 우회)."""
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    try:
        parsed = urlparse(url)
        path = parsed.path

        if "/comfy/" in path:
            comfy_idx = path.index("/comfy/")
            internal_path = path[comfy_idx + len("/comfy"):]
            internal_url = config.COMFYUI_URL + internal_path
            if parsed.query:
                internal_url += f"?{parsed.query}"
        else:
            internal_url = url

        r = requests.get(internal_url, timeout=30)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "image/png")
        ext = "png" if "png" in content_type else "webp" if "webp" in content_type else "jpg"
        filename = f"comfyui_studio_{int(time.time())}.{ext}"

        return StreamingResponse(
            io.BytesIO(r.content),
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")
