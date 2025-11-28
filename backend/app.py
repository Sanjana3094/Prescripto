# backend/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import run_ocr_on_image_bytes

app = FastAPI(title="Prescripto OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to "http://localhost:5500" etc if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = run_ocr_on_image_bytes(img_bytes)
    return result
