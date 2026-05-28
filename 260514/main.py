from flask import Flask, request, jsonify, Response, stream_with_context, render_template
import requests
 
BASEURL = "http://192.168.24.202:11434"
OLLAMA_URL = BASEURL + "/api/generate"
OLLAMA_CHAT_URL = BASEURL + "/api/chat"
 
app = Flask(__name__)
@app.get("/")
def index():
    return render_template("ex8.html")
 
 
@app.post("/api/chat")
def chat_stream():
    body = request.get_json(force=True, silent=True) or {}
    model = body.get("model", "gemma4:e2b")
    messages = body.get("messages", [])
    options = body.get("options") # 온도, 최대토큰 등 필요시 사용
    upstream = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            **({"options": options} if options else {}),
        },
        stream=True,
        timeout=600,
    )
    upstream.raise_for_status()
    def generate():
            for line in upstream.iter_lines():
                if not line:
                    continue
                yield line + b"\n"
    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")
 
@app.post("/api/generate")
def generate_stream():
    body = request.get_json(force=True, silent=True) or {}
    model = body.get("model", "gemma4:e2b")
    prompt = body.get("prompt", "")
    stream = True # 이 엔드포인트는 강제 스트리밍
    upstream = requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": stream},
                stream=True,
                timeout=10000,
    )
    def gen():
        for line in upstream.iter_lines():
            if not line:
                continue
            yield line + b"\n"
    return Response(stream_with_context(gen()), mimetype="application/x-ndjson")
 
 
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
 
 