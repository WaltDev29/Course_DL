import io
from fastapi import APIRouter, File, UploadFile, HTTPException
from .utils import preprocess_image, postprocess
from .model import model

router = APIRouter(tags=["prediction"])


# 상태 확인 엔드포인트
@router.get("/")
def index():
    return {"status": "ok", "message": "MNIST prediction API"}



# 예측 엔드포인트
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file")

    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="uploaded file must be an image")

    contents = await file.read()
    try:
        arr = preprocess_image(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image preprocessing failed: {e}")

    pred = model.predict(arr, verbose=0)
    result = postprocess(pred[0])
    return result
