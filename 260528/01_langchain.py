from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 1. LLM 설정
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.2,  # 요약은 보통 낮은 온도 권장
    base_url="http://host.docker.internal:11434",
)

# 2. 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant. Reply in Korean."),
        ("human", "한 줄로 자기소개해줘."),
    ]
)

# 3. 체인 구성
chain = prompt | llm | StrOutputParser()

# 4. 실행
if __name__ == "__main__":
    print(chain.invoke({}))