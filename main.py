from fastapi import FastAPI
import io
from PIL import Image
from fastapi.responses import FileResponse
from model import predict
import numpy as np
import struct
import random
import base64

app = FastAPI()

@app.get("/")
def home():
    return FileResponse("index.html")

with open("t10k-images.idx3-ubyte", "rb") as f:
    magic_number, num_images, rows, cols = struct.unpack(
        ">IIII", f.read(16)
    )

    mnist_images = f.read()

@app.get("/random-images")
def random_images():

    images = []

    for _ in range(10):

        index = random.randrange(num_images)

        start = index * rows * cols
        end = start + rows * cols

        pixels = mnist_images[start:end]

        image = Image.frombytes(
            "L",
            (cols, rows),
            pixels
        )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        images.append(
            f"data:image/png;base64,{image_base64}"
        )

    return {"images": images}

@app.post("/predict")
def predict_digit(data: dict) :
    image_data = data ["image"]

    image_bytes = base64.b64decode(image_data.split(",")[1])

    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image_array = np.array(image)

    coords = np.argwhere(image_array > 20)


    if len(coords) == 0:
        return {
            "prediction": -1,
            "confidence": 0
        }

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1)) 

    width, height = cropped.size

    scale = 20 / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    cropped = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    final_image = Image.new("L", (28, 28), 0)

    left = (28 - new_width) // 2
    top = (28 - new_height) // 2

    final_image.paste(cropped, (left, top))
   
    final_image.save("debug.png")

    x = np.array(image, dtype=np.float32) / 255.0
    x = x.reshape(1, 784)
    final_image.save("debug.png")


    x = np.array (image , dtype = np.float32 )/255 

    x= x.reshape (1,784)

    prediction , probabilities = predict (x)

    digit = int (prediction[0])

    confidence = float(probabilities[0][digit])

    return {
        "prediction" : digit,
        "confidence" : confidence
    }