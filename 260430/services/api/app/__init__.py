# app/__init__.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
from urllib.parse import urlparse
import json, time, os, io, random

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://host.docker.internal:8188")
WORKFLOW_PATH = os.getenv("WORKFLOW_PATH", "app/workflow.json")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")


# ---------- 요청/응답 스키마 ----------
class GenerateRequest(BaseModel):
    pos: str = "a beautiful landscape with galaxy in a bottle"
    neg: str = "text, watermark"



class GenerateResponse(BaseModel):
    img_url: str


def create_app() -> FastAPI:
    app = FastAPI(title="ComfyUI Studio API")

    # 프론트 서버에서 요청을 받을 수 있도록 CORS 허용
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- 기능별 함수 ----------
    def load_workflow(path: str) -> dict:
        """JSON 워크플로를 파일에서 읽기"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Workflow not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_prompts(
        graph: dict,
        pos: str,
        neg: str,
        pos_node_id: str = "6",
        neg_node_id: str = "7",
    ) -> dict:
        graph[pos_node_id]["inputs"]["text"] = pos
        graph[neg_node_id]["inputs"]["text"] = neg
        graph["3"]["inputs"]["seed"] = random.randint(0, 999_999_999_999_999)
        return graph

    def submit_prompt(graph: dict) -> str:
        """ComfyUI /prompt 에 워크플로 제출 후 prompt_id 반환"""
        r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": graph})
        r.raise_for_status()
        return r.json()["prompt_id"]

    def poll_history(
        prompt_id: str, timeout_sec: int = 60, interval: float = 1.0
    ) -> dict | None:
        """간단 폴링: /history/{prompt_id} 에서 완료 결과를 받을 때까지 대기"""
        end = time.time() + timeout_sec
        while time.time() < end:
            h = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
            if h.status_code == 200:
                data = h.json()
                if data:
                    return list(data.values())[0]
            time.sleep(interval)
        return None

    def extract_first_image_url(history_block: dict) -> str:
        """히스토리 블록에서 SaveImage 출력의 첫 번째 이미지를 view URL로 구성.
        브라우저가 접근하는 상대경로를 반환하여 원격 접속 시에도 동작하도록 함.
        실제 이미지 전달은 Nginx의 /comfy/view 프록시가 담당.
        """
        if not history_block:
            return ""
        outputs = history_block.get("outputs", {})
        for _node_id, node_out in outputs.items():
            if "images" in node_out and node_out["images"]:
                img = node_out["images"][0]
                fn = img.get("filename")
                sub = img.get("subfolder", "")
                t = img.get("type", "output")
                if fn:
                    # 내부 주소 대신 Nginx 프록시 경로(상대경로) 반환
                    return f"/comfy/view?filename={fn}&subfolder={sub}&type={t}"
        return ""

    # ---------- 라우트 ----------
    @app.get("/")
    def health():
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(body: GenerateRequest):
        """프롬프트를 받아 Ollama로 번역 후 ComfyUI로 이미지를 생성하고 URL을 반환"""

        # Ollama를 이용한 번역 및 프롬프트 정제
        system_msg = (
            "You are a professional prompt engineer and translator. "
            "Translate the given Korean prompts into descriptive English for AI image generation. "
            "Return ONLY a JSON object with 'pos' and 'neg' keys."
        )
        prompt_content = f"Positive Prompt: {body.pos}\nNegative Prompt: {body.neg}"

        res = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "gemma4:e2b",
                "system": system_msg,
                "prompt": prompt_content,
                "format": "json",
                "stream": False
            },
            timeout=600,
        )

        res.raise_for_status()
        ollama_res = res.json()
        
        # response 문자열을 JSON 객체로 변환
        try:
            translated = json.loads(ollama_res["response"])
            pos_final = translated.get("pos", body.pos)
            neg_final = translated.get("neg", body.neg)
        except (json.JSONDecodeError, KeyError):
            # 파싱 실패 시 원본 사용
            pos_final = body.pos
            neg_final = body.neg

        try:
            graph = load_workflow(WORKFLOW_PATH)
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

    @app.get("/download")
    def download(url: str):
        """ComfyUI 이미지를 서버 사이드에서 받아 클라이언트에 전달 (CORS 우회).
        브라우저는 img.src를 항상 절대경로(http://host/comfy/view?...)로 반환하므로,
        urlparse로 path/query만 추출해 내부 COMFYUI_URL로 재조합한다.
        """
        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")
        try:
            parsed = urlparse(url)
            path = parsed.path  # e.g. "/comfy/view"

            if "/comfy/" in path:
                # /comfy/view → /view
                comfy_idx = path.index("/comfy/")
                internal_path = path[comfy_idx + len("/comfy"):]
                internal_url = COMFYUI_URL + internal_path
                if parsed.query:
                    internal_url += f"?{parsed.query}"
            else:
                # 알 수 없는 경로면 그대로 시도
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

    return app