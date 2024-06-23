import tempfile
from flask import Flask, jsonify, render_template, request, redirect, session, flash, send_file
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
model_path = "best.pt"
model = YOLO(model_path, task="detect")
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
# Load your YOLO model (adjust the path as needed)
def draw_boxes(image: Image.Image, boxes):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        from ultralytics import YOLO
        # Read the image file
        image = Image.open(request.files["file"])
        # Convert PIL image to numpy array
        image_np = np.array(image)
        # Perform inference
        results = model(image_np, conf=0.5, imgsz=640, iou=0.25)
        # Process results
        boxes = results[0].boxes.xyxy  # Bounding boxes
        # Check if there are no bounding boxes
        if len(boxes) == 0:
            return jsonify({"error": "No objects detected in the image."}), 400
        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image, boxes)
        # Save the image with bounding boxes to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_with_boxes.save(temp_file, format="JPEG")
        temp_file.close()
        # Serve the file
        return send_file(temp_file.name, as_attachment=True, attachment_filename="image_with_boxes.jpg", mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True,port=5000,threaded=True)