from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

app = Flask(__name__)

# ---- 체인 구성 (서버 시작 시 1회 초기화) ----
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "다음 텍스트를 핵심 bullet 5개로 한국어 요약해줘."),
    ("human", "텍스트:\n{content}") 
])

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.2,
    base_url="http://host.docker.internal:11434",
)

chain = PROMPT | llm | StrOutputParser()


def _sse_format(data: str, event: str | None = None) -> str:
    lines = []
    if event:
        lines.append(f"\nevent: {event}\n")
    else:
        lines.append(f"{data}")
    return "".join(lines)


@app.route("/summarize/stream", methods=["POST"])
def summarize_stream():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "text 필드에 요약할 문자열을 넣어주세요."}), 400

    @stream_with_context
    def generate():
        # 연결 성사 이벤트(옵션)
        yield _sse_format("started", event="open")

        # LangChain 체인 스트림
        try:
            for chunk in chain.stream({"content": text}):
                # chunk는 문자열 파서 결과의 조각이므로 그대로 전송
                yield _sse_format(chunk)
        except Exception as e:
            # 에러 이벤트
            yield _sse_format(f"error: {str(e)}", event="error")
        else:
            # 완료 이벤트(옵션)
            yield _sse_format("end", event="end")

    # SSE 응답 헤더
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Nginx 등 리버스 프록시 사용 시 버퍼링 방지
        "X-Accel-Buffering": "no",
    }

    return Response(generate(), mimetype="text/event-stream", headers=headers)


@app.route("/", methods=["GET"])
def index():
    return render_template("appsse1.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)