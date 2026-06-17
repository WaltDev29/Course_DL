import json
import os
import random
import requests
import time
from app.core.config import config

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
    r = requests.post(f"{config.COMFYUI_URL}/prompt", json={"prompt": graph})
    r.raise_for_status()
    return r.json()["prompt_id"]

def poll_history(
    prompt_id: str, timeout_sec: int = 60, interval: float = 1.0
) -> dict | None:
    """간단 폴링: /history/{prompt_id} 에서 완료 결과를 받을 때까지 대기"""
    end = time.time() + timeout_sec
    while time.time() < end:
        h = requests.get(f"{config.COMFYUI_URL}/history/{prompt_id}")
        if h.status_code == 200:
            data = h.json()
            if data:
                return list(data.values())[0]
        time.sleep(interval)
    return None

def extract_first_image_url(history_block: dict) -> str:
    """히스토리 블록에서 SaveImage 출력의 첫 번째 이미지를 view URL로 구성."""
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
                return f"/comfy/view?filename={fn}&subfolder={sub}&type={t}"
    return ""
