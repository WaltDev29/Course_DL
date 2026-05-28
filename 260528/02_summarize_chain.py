import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 1) 프롬프트 (한국어 요약, 5줄 이내)
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "다음 텍스트를 핵심 bullet 5개로 한국어 요약해줘."),
        ("human", "텍스트:{content}") 
    ]
)

# 2) LLM 준비: gemma3:4b (온도 낮게: 요약 일관성↑)
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.2,
    base_url="http://host.docker.internal:11434",
)

# 3) 체인: 프롬프트 → LLM → 문자열 파싱
chain = PROMPT | llm | StrOutputParser()


def summarize_text(text: str) -> str:
    return chain.invoke({"content": text})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python summarize_chain.py <텍스트파일 경로>")
        raise SystemExit(1)

    text_path = Path(sys.argv[1])
    text = text_path.read_text(encoding="utf-8")

    # 일반 출력 (주석 처리)
    # print(summarize_text(text))

    # Stream 출력
    for chunk in chain.stream({"content": text}):
        print(chunk, end="", flush=True)