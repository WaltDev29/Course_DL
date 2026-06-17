import io
import requests
import time
import json
import asyncio
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, Cookie
from fastapi.responses import StreamingResponse

from app.schemas.generate import GenerateRequest, GenerateResponse
from app.services.comfyui import (
    load_workflow,
    update_prompts,
    submit_prompt,
    poll_history,
    extract_first_image_url
)
from app.core.config import config
from app.services.chat_service import chat_chain, translate_to_english_json, get_history

router = APIRouter()

@router.get("/")
def health():
    return {"status": "ok"}

@router.post("/generate")
async def generate(session_id: str | None = Cookie(None)):
    """채팅 기록을 기반으로 프롬프트를 추출하고 ComfyUI로 이미지를 생성하여 SSE로 스트리밍 반환"""
    if not session_id:
        raise HTTPException(status_code=400, detail="No session found")
        
    history = get_history(session_id).messages
    if not history:
        raise HTTPException(status_code=400, detail="No chat history to generate image from")

    async def event_generator():
        try:
            # 1. 한국어 프롬프트 생성 (요약 전용 체인 사용, JSON 파서 적용)
            from app.services.chat_service import summary_chain
            
            summary_result = await summary_chain.ainvoke(
                {
                    "history": history,
                    "input": "지금까지의 대화를 바탕으로 생성할 이미지를 요약하시오."
                }
            )
            
            # 요약된 JSON을 문자열로 변환
            kor_prompt = json.dumps(summary_result, ensure_ascii=False, indent=2)
            
            # 프론트엔드에 한 번에 출력
            yield f'data: {json.dumps({"status": "streaming_kor", "content": kor_prompt})}\n\n'
            
            # 2. 이미지 생성 중 표시
            yield f'data: {json.dumps({"status": "generating_image"})}\n\n'
            
            # 3. 영어 번역 (JSON 반환만 처리)
            eng_prompts = await translate_to_english_json(kor_prompt)
            pos_final = eng_prompts.get("pos", kor_prompt)
            neg_final = eng_prompts.get("neg", "")
            print(f"pos: {pos_final}\nneg: {neg_final}")
            
            # ComfyUI 실행 (동기 함수들을 비동기에서 실행)
            loop = asyncio.get_running_loop()
            
            def run_comfyui():
                graph = load_workflow(config.WORKFLOW_PATH)
                updated_graph = update_prompts(graph, pos_final, neg_final)
                prompt_id = submit_prompt(updated_graph)
                history_block = poll_history(prompt_id)
                return extract_first_image_url(history_block)
            
            img_url = await loop.run_in_executor(None, run_comfyui)
            
            if not img_url:
                yield f'data: {json.dumps({"status": "error", "error": "Image generation timed out"})}\n\n'
                return
                
            # 이미지 세션에 저장
            from app.services.chat_service import add_image_to_session, get_history
            add_image_to_session(session_id, img_url)

            # 채팅 내역에도 이미지를 남겨 새로고침 시 대화창에 복구되도록 함
            history_obj = get_history(session_id)
            img_html = f'<img src="{img_url}" class="chat-img" alt="Generated Image" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); cursor: zoom-in;" />'
            history_obj.add_ai_message(img_html)

            # 4. 완료 및 이미지 전송
            yield f'data: {json.dumps({"status": "complete", "img_url": img_url})}\n\n'
            yield f'data: [DONE]\n\n'
            
        except Exception as e:
            yield f'data: {json.dumps({"status": "error", "error": str(e)})}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
