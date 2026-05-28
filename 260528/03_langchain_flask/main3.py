from flask import Flask, request, render_template
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

app = Flask(__name__)

# ---- 체인 및 파서 설정 ----
_json_llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.2,
    base_url="http://host.docker.internal:11434",
)
_json_parser = JsonOutputParser()
_json_format = _json_parser.get_format_instructions()

_json_prompt = ChatPromptTemplate.from_template(
    """다음 텍스트를 JSON으로 요약해.
- "summary": 한국어 핵심 요약(3~5문장)
- "keywords": 핵심 키워드 5~8개 리스트
- "oneline": 한 줄 요약(20자 내외)
반드시 JSON만 출력.
{format_instructions}
텍스트:
{content}
"""
)
_json_chain = _json_prompt | _json_llm | _json_parser


@app.route("/", methods=["GET"])
def index():
    return render_template("appsse2.html")


# ---- 라우트 설정 ----
@app.route("/summarize/structured", methods=["POST"])
def summarize_structured():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return {"error": "text 필드가 필요합니다."}, 400

    import json
    from flask import Response
    
    result = _json_chain.invoke(
        {"content": text, "format_instructions": _json_format}
    )

    # 한글 깨짐 방지 및 보기 좋게 들여쓰기(indent=2) 적용
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    return Response(json_str, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)