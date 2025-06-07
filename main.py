from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
from yolov8_utils import detect_objects
import io
from PIL import Image

app = FastAPI()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV

    result = detect_objects(frame)

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
