from flask import Flask, jsonify, request
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

app = Flask(__name__)

PROMPT = ChatPromptTemplate.from_template(
    "다음 텍스트를 핵심 bullet 5개로 한국어 요약해줘.\n\n텍스트:\n{content}"
)

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.2,
    base_url="http://host.docker.internal:11434",
)

chain = PROMPT | llm | StrOutputParser()


@app.route("/summarize", methods=["POST"])
def summarize():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "text 필드에 요약할 문자열을 넣어주세요."}), 400

    summary = chain.invoke({"content": text})
    return {"summary": summary}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)