from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import io
import base64

app = FastAPI(title="Image Caption Generator")
# Pydantic model to accept base64 image data
class ImagePayload(BaseModel):
    image: str  # base64 encoded image

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Object detection service is running"}

# Initialize the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.post("/generate_caption")
async def generate_caption(request: ImageRequest):
    try:
        # Get the base64 image from the request
        base64_image = request.image
        
        # Remove the base64 prefix if it exists
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        max_new_tokens = 50
        
        # Generate caption
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        return {"caption": caption}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
