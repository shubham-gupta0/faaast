import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO

app = FastAPI()

# Load your YOLO model (adjust the path as needed)
model_path = "best.pt"
model = YOLO(model_path, task="detect")

# Load your MobileNet model (adjust the path as needed)
mobilenet_model_path = "muzzle_model.h5"
mobilenet_model = load_model(mobilenet_model_path)

def draw_boxes(image: Image.Image, boxes):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image

def preprocess_image(image: Image.Image, target_size=(128, 128)):
    try:
        image = image.convert('RGB')  # Convert to RGB
        image = image.resize(target_size, Image.LANCZOS)
        image = img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def compare_images(model, image1: Image.Image, image2: Image.Image):
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)
    
    if image1 is None or image2 is None:
        print("Error in preprocessing images.")
        return None

    image1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    image2 = np.expand_dims(image2, axis=0)  # Add batch dimension

    similarity = model.predict([image1, image2])

    # Assuming the model outputs a similarity score where higher values indicate greater similarity
    similarity_score = float(similarity[0][0])  # Convert numpy.float32 to Python float

    # Determine if images are of the same muzzle or not based on a threshold
    threshold = 0.5  # You may need to adjust this threshold based on your model's performance
    if similarity_score > threshold:
        result = "Same muzzle"
    else:
        result = "Different muzzles"

    return result, similarity_score

@app.get("/")
async def hello_world():
    return {"message": "Hello, World!"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
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

@app.post("/compare")
async def compare_images_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # Read the image files
        image1 = Image.open(file1.file)
        image2 = Image.open(file2.file)

        # Compare the images using the MobileNet model
        result, similarity_score = compare_images(mobilenet_model, image1, image2)

        # Check if there was an error during comparison
        if result is None:
            raise HTTPException(status_code=400, detail="Error in processing images for comparison.")

        return {
            "result": result,
            "similarity_score": similarity_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
