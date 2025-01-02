from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Load the pre-trained YOLOv5 model (small version 'yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Object detection service is running"}

@app.post("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Perform inference
    results = model(img)
    
    # Process results
    labels = results.names
    predicted_classes = results.xywh[0][:, -1].tolist()
    predicted_objects = [labels[int(class_id)] for class_id in predicted_classes]
    unique_predicted_objects = list(set(predicted_objects))
    
    return {"detected_objects": unique_predicted_objects}

