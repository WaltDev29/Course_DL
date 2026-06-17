import uuid
from typing import Optional
from fastapi import APIRouter, Response, Cookie
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.chat_service import stream_chat_response, clear_session_history, get_history

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.get("/chat/init")
async def init_session(response: Response, session_id: Optional[str] = Cookie(None)):
    """프론트엔드 로드 시 세션 초기화를 위한 API (Nginx에서 정적 파일 서빙 시 호출 필요)"""
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, max_age=2592000, httponly=True)
        return {"status": "ok", "session_id": session_id, "is_new": True}
        
    return {"status": "ok", "session_id": session_id, "is_new": False}

@router.get("/chat/history")
async def get_session_history(session_id: Optional[str] = Cookie(None)):
    """현재 세션의 대화 내역을 반환"""
    if not session_id:
        return {"status": "error", "message": "No session found", "history": []}
    
    history = get_history(session_id).messages
    formatted_history = []
    for msg in history:
        # system 프롬프트는 제외하고 사용자(human)와 AI(ai) 대화만 포함
        if msg.type in ["human", "ai"]:
            formatted_history.append({"type": msg.type, "content": msg.content})
            
    return {"status": "success", "history": formatted_history}

@router.get("/chat/images")
async def get_images(session_id: Optional[str] = Cookie(None)):
    """현재 세션에서 생성된 이미지 URL 목록을 반환"""
    if not session_id:
        return {"status": "error", "message": "No session found", "images": []}
    
    from app.services.chat_service import get_session_images
    images = get_session_images(session_id)
    return {"status": "success", "images": images}

@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, session_id: Optional[str] = Cookie(None)):
    """사용자 메시지를 받아 LLM으로부터 스트리밍 응답을 생성"""
    if not session_id:
        # 일반적으로 첫 접근시 /를 통해 쿠키를 받으나 직접 호출할 수도 있으므로 예외처리
        session_id = str(uuid.uuid4())
    
    # StreamingResponse를 이용해 SSE 전송
    return StreamingResponse(
        stream_chat_response(session_id, body.message),
        media_type="text/event-stream"
    )

@router.delete("/chat/history")
async def clear_history(session_id: Optional[str] = Cookie(None)):
    """현재 세션의 대화 내역을 초기화"""
    if not session_id:
         return {"status": "error", "message": "No session found"}
    
    if clear_session_history(session_id):
        return {"status": "success", "message": "History cleared"}
    else:
        return {"status": "not_found", "message": "History not found for this session"}
