# app2.py
from flask import Flask, request, render_template, Response, send_file
import requests, json, time, os, io
import random

COMFY_URL = "http://host.docker.internal:8188" # GPU Server URL
WORKFLOW_PATH = "./static/test.json" # JSON파일 경로


app = Flask(__name__)

images = []


# ---------- 기능별 함수 ----------
def load_workflow(path: str) -> dict:
    """JSON 워크플로를 파일에서 읽기"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Workflow not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_prompts(graph: dict, pos: str, neg: str, pos_node_id: str = "6", neg_node_id: str = "7") -> dict:
    graph[pos_node_id]["inputs"]["text"] = pos
    graph[neg_node_id]["inputs"]["text"] = neg
    graph["3"]["inputs"]["seed"] = random.randint(0, 999999999999999)
    return graph


def submit_prompt(graph: dict) -> str:
    #ComfyUI /prompt 에 워크플로 제출 후 prompt_id 반환
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph})
    r.raise_for_status()
    return r.json()["prompt_id"]


def poll_history(prompt_id: str, timeout_sec: int = 60, interval: float = 1.0) -> dict | None:
    """
    간단 폴링: /history/{prompt_id} 에서 완료 결과를 받을 때까지 대기
    """
    end = time.time() + timeout_sec

    while time.time() < end:
        h = requests.get(f"{COMFY_URL}/history/{prompt_id}")
        if h.status_code == 200:
            data = h.json()
            if data:  # 결과가 비어있지 않을 때만 반환
                return list(data.values())[0]  # 해당 prompt의 히스토리 블록
        time.sleep(interval)

    return None


def extract_first_image_url(history_block: dict) -> str:
    """
    히스토리 블록에서 SaveImage 출력의 첫 번째 이미지를 view URL로 구성
    """
    if not history_block:
        return ""
        
    outputs = history_block.get("outputs", {})

    for node_id, node_out in outputs.items():
        if "images" in node_out and node_out["images"]:
            img = node_out["images"][0]
            fn = img.get("filename")
            sub = img.get("subfolder", "")
            t = img.get("type", "output")
            if fn:
                return f"{COMFY_URL}/view?filename={fn}&subfolder={sub}&type={t}"
    return ""



# ---------- Flask 라우트 ----------
@app.route("/download")
def download():
    """ComfyUI 이미지를 서버 사이드에서 받아 브라우저에 파일로 전달 (CORS 우회)"""
    img_url = request.args.get("url", "")
    if not img_url:
        return "No URL provided", 400

    try:
        r = requests.get(img_url, timeout=30)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "image/png")
        ext = "png" if "png" in content_type else "webp" if "webp" in content_type else "jpg"
        filename = f"comfyui_studio_{int(time.time())}.{ext}"

        return send_file(
            io.BytesIO(r.content),
            mimetype=content_type,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return f"Download failed: {e}", 500


@app.route("/", methods=["GET", "POST"])
def index():
    pos = request.form.get("pos", "a beautiful landscape with galaxy in a bottle")
    neg = request.form.get("neg", "text, watermark")
    img_url = ""

    if request.method == "POST":
        graph = load_workflow(WORKFLOW_PATH)
        graph = update_prompts(graph, pos, neg) # 필요 시 노드 ID 인자 조정 가능
        prompt_id = submit_prompt(graph)
        history_block = poll_history(prompt_id)
        img_url = extract_first_image_url(history_block)
        if img_url:  # 빈 URL은 갤러리에 추가하지 않음
            images.append(img_url)  # template의 img.url 접근에 맞게 dict로 저장
    return render_template("image.html", pos=pos, neg=neg, img_url=img_url, images=reversed(images))



if __name__ == "__main__":
    # pip install flask requests
    app.run(host="0.0.0.0", port=5000, debug=True)