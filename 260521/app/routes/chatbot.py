# import ollama
from fastapi import APIRouter
from pydantic import BaseModel
import requests
import os
from langchain_openai import ChatOpenAI

OLLAMA_CHAT_URL = "http://host.docker.internal:11434/api/chat"
model = "gemma4:e2b"

llm = ChatOpenAI(
    base_url="http://kimi.aikopo.net/v1",
    model="MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf",
    api_key="dummy_key",  # 빈 문자열("")이면 Pydantic 에러가 발생하므로 임의의 값 입력
    default_headers={
        "User-Agent": "Mozilla/5.0"
    }
)

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

memory = []

def response(messages: list):
    global memory

    res = llm.invoke(messages)
    # res = requests.post(
    #     OLLAMA_CHAT_URL,
    #     json={
    #         "model": model,
    #         "messages": messages,
    #         "stream": False,  # stream=False로 변경하여 전체 응답을 한 번에 받음
    #     },
    #     timeout=600,
    # )
    # res.raise_for_status()
    print(res.content)
    
    # requests.Response 객체에서 JSON 데이터를 파싱
    data = { "role": "assistant", "content": res.content }
    memory.append(data)

    # 대화 기록 슬라이싱
    if len(memory) > 11:
        # memory[0]은 시스템 프롬프트이므로 유지
        # 최근 3턴만 남기고 과거 기록 삭제
        memory = [memory[0]] + memory[-6:]
    
    return {"message": data}


@router.get("/init")
def init_chat():
    global memory

    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = "당신은 식인종 선교사 게임의 진행자입니다. 본 게임은 텍스트로 진행합니다."

    memory = [{"role": "system", "content": system_prompt.strip()}]
    return {"status": "initialized"}



class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(chat_request: ChatRequest):
    global memory

    memory.append({"role": "user", "content": chat_request.message})

    return response(memory)