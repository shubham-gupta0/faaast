import os
import tempfile
from flask import Flask, request, send_file, abort
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import io

app = Flask(__name__)

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

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        # Read the image file
        file = request.files['file']
        image = Image.open(file.stream)

        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Perform inference
        results = model(image_np, conf=0.5, imgsz=640, iou=0.25)
        
        # Process results
        boxes = results[0].boxes.xyxy  # Bounding boxes

        # Check if there are no bounding boxes
        if len(boxes) == 0:
            abort(400, description="No objects detected in the image.")

        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image, boxes)

        # Save the image with bounding boxes to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_with_boxes.save(temp_file, format="JPEG")
        temp_file.close()

        # Serve the file
        response = send_file(temp_file.name, mimetype="image/jpeg")

        # Clean up the temporary file
        @response.call_on_close
        def remove_file():
            os.remove(temp_file.name)

        return response

    except Exception as e:
        abort(500, description=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
