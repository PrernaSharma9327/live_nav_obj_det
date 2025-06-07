import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from yolov8_utils import detect_objects
import numpy as np
from PIL import Image
import io
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)[:, :, ::-1]
    result = detect_objects(frame)
    return JSONResponse(content=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
