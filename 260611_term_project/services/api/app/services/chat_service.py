from collections import defaultdict
from typing import Dict, AsyncGenerator
import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# LLM 설정
llm = ChatOllama(
    model="gemma4:e2b",
    temperature=0.7,
    base_url="http://host.docker.internal:11434",
)

to_str = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 사용자가 원하는 이미지를 구체화하도록 돕는 전문 AI 어시스턴트입니다. "
                   "직접 이미지를 생성할 수는 없으며, 오직 이미지 생성을 위한 아이디어를 구체화하는 대화만 나눕니다. "
                   "사용자의 아이디어에 대해 질문을 던져 스타일, 분위기, 색상, 피사체 등을 상세히 설정하도록 유도하세요. "
                   "사용자가 이미지 생성을 원하거나 프롬프트가 충분히 구체화되었다고 판단되면, "
                   "화면의 '그림 생성 버튼(이미지 아이콘)'을 클릭해 달라고 안내하십시오. "
                   "사용자가 해당 버튼을 누르면 그동안의 대화 내용을 바탕으로 프롬프트가 완성되어 실제 이미지가 생성됩니다. "
                   "절대로 임의의 URL을 제공하거나 임의의 이미지를 생성하지 마십시오."
                   ),
        MessagesPlaceholder("history"),  # 과거 대화 주입
        ("human", "{input}"),
    ]
)

base_chain = prompt | llm | to_str

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 사용자의 대화 내역을 바탕으로 이미지 생성을 위한 프롬프트를 추출하는 AI 어시스턴트입니다. "
                   "대화 내역을 분석하여 사용자가 생성하고자 하는 이미지의 구체적인 묘사를 'pos'에, 피해야 할 요소를 'neg'에 한글로 요약하세요. "
                   "주의: 'pos'와 'neg'의 내용(value)은 반드시 한국어(한글)로만 작성해야 합니다. 영어를 절대 사용하지 마세요. "
                   "neg에 입력할 요소가 없으면 빈 문자열을 반환하세요. "
                   "반드시 'pos'와 'neg' 키를 가진 JSON 형식으로만 반환하고, 다른 발화는 절대 하지 마세요. "
                   "\n\n출력 예시:\n"
                   "{{\n"
                   "  \"pos\": \"따뜻한 햇살이 비치는 숲 속에서 평화롭게 잠자는 귀여운 하얀 고양이, 수채화 스타일, 파스텔 톤\",\n"
                   "  \"neg\": \"텍스트, 사람, 기괴한 모습, 무서운 분위기\"\n"
                   "}}"
                   ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

summary_chain = summary_prompt | llm | JsonOutputParser()

# 세션 저장소 (실무에선 Redis/File 등으로 교체)
_STORE: Dict[str, InMemoryChatMessageHistory] = defaultdict(
    InMemoryChatMessageHistory
)
_IMAGE_STORE: Dict[str, list] = defaultdict(list)

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """세션 ID에 해당하는 대화 내역을 반환합니다."""
    return _STORE[session_id]

def add_image_to_session(session_id: str, img_url: str):
    """지정된 세션에 생성된 이미지 URL을 추가합니다."""
    _IMAGE_STORE[session_id].append(img_url)

def get_session_images(session_id: str) -> list:
    """지정된 세션에서 생성된 모든 이미지 URL 목록을 반환합니다."""
    return _IMAGE_STORE[session_id]

# 히스토리 지원 체인 생성
chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",     
    history_messages_key="history", 
)

async def stream_chat_response(session_id: str, user_message: str) -> AsyncGenerator[str, None]:
    """사용자 메시지에 대해 스트리밍 응답을 생성하는 비동기 제너레이터입니다."""
    try:
        # astream()을 사용하여 실시간 청크 반환
        async for chunk in chat_chain.astream(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}},
        ):
            # Server-Sent Events (SSE) 포맷에 맞게 데이터 전송
            # JSON 형태로 감싸서 프론트엔드에서 파싱하기 쉽게 만듭니다.
            data = json.dumps({"content": chunk})
            yield f"data: {data}\n\n"
        
        # 스트리밍 종료 신호 전송
        yield f"data: [DONE]\n\n"
    except Exception as e:
        # 에러 핸들링
        error_msg = json.dumps({"error": str(e)})
        yield f"data: {error_msg}\n\n"

translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI image prompt translator. Translate the following Korean text into English for ComfyUI. "
               "Include all descriptive details in the positive prompt, and things to avoid in the negative prompt. "
               "Return ONLY a JSON object with 'pos' and 'neg' keys."),
    ("human", "{korean_prompt}")
])

eng_translate_chain = translate_prompt | llm | JsonOutputParser()

async def translate_to_english_json(korean_prompt: str) -> dict:
    """한국어 프롬프트를 영어 pos, neg 형태의 JSON으로 번역합니다."""
    try:
        return await eng_translate_chain.ainvoke({"korean_prompt": korean_prompt})
    except Exception as e:
        # 파싱 실패 시 원본 문자열을 그대로 pos에 반환
        return {"pos": korean_prompt, "neg": ""}

def clear_session_history(session_id: str) -> bool:
    """지정된 세션의 대화 기록 및 생성된 이미지를 삭제합니다."""
    cleared = False
    if session_id in _STORE:
        _STORE[session_id].clear() # InMemoryChatMessageHistory 의 clear() 사용
        cleared = True
    if session_id in _IMAGE_STORE:
        _IMAGE_STORE[session_id].clear()
        cleared = True
    return cleared
