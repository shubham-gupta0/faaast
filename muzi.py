import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

# Load the model once at startup
model = None

def preprocess_image(image_file, target_size=(128, 128)):
    try:
        image = Image.open(BytesIO(image_file))
        image = image.convert('RGB')  # Convert to RGB
        image = image.resize(target_size, Image.LANCZOS)
        image = img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def load_mobilenet_model(model_path):
    global model
    model = load_model(model_path)

def compare_images(image1_file, image2_file):
    image1 = preprocess_image(image1_file)
    image2 = preprocess_image(image2_file)
    
    if image1 is None or image2 is None:
        print("Error in preprocessing images.")
        return None, None

    image1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    image2 = np.expand_dims(image2, axis=0)  # Add batch dimension

    prediction = model.predict([image1, image2])

    # Assuming the model outputs a distance metric where smaller values indicate greater similarity
    distance = prediction[0][0]

    # Determine if images are of the same muzzle or not based on a threshold
    threshold = 0.5  # You may need to adjust this threshold based on your model's performance
    if distance > threshold:
        result = "Same muzzle"
    else:
        result = "Different muzzles"

    return result, distance

@app.on_event("startup")
def load_model_on_startup():
    model_path = "muzzle_model.h5"
    load_mobilenet_model(model_path)

@app.post("/compare-muzzles")
async def compare_muzzles(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        image1_file = await file1.read()
        image2_file = await file2.read()
        result, distance = compare_images(image1_file, image2_file)
        if result is None:
            raise HTTPException(status_code=400, detail="Error processing images")
        return {"result": result, "distance": distance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
