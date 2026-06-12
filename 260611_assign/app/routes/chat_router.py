import uuid
from typing import Optional
from fastapi import APIRouter, Request, Response, Cookie
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.chat_service import stream_chat_response, clear_session_history

router = APIRouter()
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request, session_id: Optional[str] = Cookie(None)):
    """채팅 메인 페이지 렌더링 및 세션 초기화"""
    is_new_session = False
    # 세션 쿠키가 없으면 새로 발급
    if not session_id:
        session_id = str(uuid.uuid4())
        is_new_session = True
    
    response = templates.TemplateResponse(request=request, name="index.html")
    if is_new_session:
        # 반환할 Response 객체에 쿠키를 세팅해야 정상 작동함
        response.set_cookie(key="session_id", value=session_id, max_age=86400, httponly=True)
        
    return response


@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest, session_id: Optional[str] = Cookie(None)):
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
