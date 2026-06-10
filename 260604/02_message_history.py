# memory_chat_lcel.py

from collections import defaultdict
from typing import Dict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOllama(
    model="gemma4:e2b",
    temperature=0.7,
    base_url="http://host.docker.internal:11434",
)

to_str = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 친절한 한국어 비서야. 대화 맥락을 잘 이어가."),
        MessagesPlaceholder("history"),  # 과거 대화가 주입될 자리
        ("human", "{input}"),
    ]
)

base_chain = prompt | llm | to_str

# 세션 저장소 (실무에선 Redis/File 등으로 교체)
_STORE: Dict[str, InMemoryChatMessageHistory] = defaultdict(
    InMemoryChatMessageHistory
)


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    return _STORE[session_id]


chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",      # 최신 사용자 입력 변수명
    history_messages_key="history",  # MessagesPlaceholder 이름과 일치
)

if __name__ == "__main__":
    sid = "user-1"

    print(
        chat_chain.invoke(
            {"input": "안녕? 오늘 할 일 정리해줘."},
            config={"configurable": {"session_id": sid}},
        )
    )

    print(
        chat_chain.invoke(
            {"input": "내가 방금 뭐라했지?"},
            config={"configurable": {"session_id": sid}},
        )
    )