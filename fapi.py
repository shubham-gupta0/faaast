import os
import tempfile
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

app = FastAPI()

# Load your YOLO model (adjust the path as needed)
model_path = "best.pt"
model = YOLO(model_path, task="detect")

def draw_boxes(image: Image.Image, boxes):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image

@app.get("/")
async def hello_world():
    return {"message": "Hello, World!"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        file.file.seek(0)
        image = Image.open(file.file)

        # Check image format
        if image.format not in ["JPEG", "PNG"]:
            raise HTTPException(status_code=400, detail="Unsupported image format.")

        # Check image size
        # if image.size[0] > 2000 or image.size[1] > 2000:
        #     raise HTTPException(status_code=400, detail="Image is too large.")

        # Check color channels
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Perform inference
        results = model(image_np, conf=0.5, imgsz=640, iou=0.25)
        
        # Process results
        boxes = results[0].boxes.xyxy  # Bounding boxes

        # Check if there are no bounding boxes
        if len(boxes) == 0:
            raise HTTPException(status_code=500, detail="No objects detected in the image.")

        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image, boxes)

        # Save the image with bounding boxes to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_with_boxes.save(temp_file, format="JPEG")
        temp_file.close()

        # Serve the file
        return FileResponse(temp_file.name, media_type="image/jpeg", filename="image_with_boxes.jpg")

    except HTTPException as e:
        print(f"HTTPException: {e.detail}")
        traceback.print_exc()
        return JSONResponse(status_code=e.status_code, content={"message": str(e.detail)})
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=e.status_code, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)