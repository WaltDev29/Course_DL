from collections import defaultdict
from typing import Dict, AsyncGenerator
import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
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
        ("system", "너는 친절한 한국어 비서야. 대화 맥락을 잘 이어가."),
        MessagesPlaceholder("history"),  # 과거 대화 주입
        ("human", "{input}"),
    ]
)

base_chain = prompt | llm | to_str

# 세션 저장소 (실무에선 Redis/File 등으로 교체)
_STORE: Dict[str, InMemoryChatMessageHistory] = defaultdict(
    InMemoryChatMessageHistory
)

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """세션 ID에 해당하는 대화 내역을 반환합니다."""
    return _STORE[session_id]

# 히스토리 지원 체인 생성
chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",      # 최신 사용자 입력 변수명
    history_messages_key="history",  # MessagesPlaceholder 이름과 일치
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

def clear_session_history(session_id: str) -> bool:
    """지정된 세션의 대화 기록을 삭제합니다."""
    if session_id in _STORE:
        _STORE[session_id].clear() # InMemoryChatMessageHistory 의 clear() 사용
        return True
    return False
