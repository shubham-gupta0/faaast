import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
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
def upload_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = Image.open(file.file)

        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Perform inference
        results = model(image_np, conf=0.5, imgsz=640, iou=0.25)
        
        # Process results
        boxes = results[0].boxes.xyxy  # Bounding boxes

        # Check if there are no bounding boxes
        if len(boxes) == 0:
            raise HTTPException(status_code=400, detail="No objects detected in the image.")

        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image, boxes)

        # Save the image with bounding boxes to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_with_boxes.save(temp_file, format="JPEG")
        temp_file.close()

        # Serve the file
        return FileResponse(temp_file.name, media_type="image/jpeg", filename="image_with_boxes.jpg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file after response is sent
        @app.on_event("shutdown")
        def remove_temp_file():
            try:
                os.remove(temp_file.name)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="172.16.61.108", port=3000)
