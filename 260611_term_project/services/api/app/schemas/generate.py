from pydantic import BaseModel

class GenerateRequest(BaseModel):
    pos: str = "a beautiful landscape with galaxy in a bottle"
    neg: str = "text, watermark"

class GenerateResponse(BaseModel):
    img_url: str
