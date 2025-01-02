from fastapi import FastAPI
from pydantic import BaseModel
import torch
from PIL import Image
import io
import base64

app = FastAPI()

# Load the pre-trained YOLOv5 model (small version 'yolov5s')
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',trust_repo=True)

# Pydantic model to accept base64 image data
class ImagePayload(BaseModel):
    image: str  # base64 encoded image

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Object detection service is running"}

@app.post("/detect-objects")
async def detect_objects(payload: ImagePayload):
    try:
        # Decode the base64 image
        base64_str = payload.image.split(",")[1]
     
        image_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_data))

        # Perform inference with the YOLO model
        results = model(img)

        # Process results
        labels = results.names
        predicted_classes = results.xywh[0][:, -1].tolist()
        predicted_objects = [labels[int(class_id)] for class_id in predicted_classes]
        unique_predicted_objects = list(set(predicted_objects))

        return {"detected_objects": unique_predicted_objects}
    
    except Exception as e:
        return {"error": str(e)}
