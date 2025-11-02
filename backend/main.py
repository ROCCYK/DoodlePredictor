import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from PIL import Image
import json
import cv2
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Categories list (same as in Streamlit app)
cats = ['airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

# Global model variable
model = None
NCATS = 340
SIZE = 64

def top_3_accuracy(y_true, y_pred):
    from tensorflow.keras.metrics import top_k_categorical_accuracy
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def load_model():
    """Load the model once at startup"""
    global model
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model = MobileNet(input_shape=(SIZE, SIZE, 1), alpha=1., weights=None, classes=NCATS)
    model.load_weights('../model.h5')
    model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss='categorical_crossentropy',
        metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy]
    )
    print("Model loaded successfully")

# EXACT image processing functions from Streamlit app
def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    """Exactly as in Streamlit app - draws strokes on a canvas"""
    BASE_SIZE = 256
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def df_to_image_array_xd(df, size, lw=6, time_color=True):
    """Exactly as in Streamlit app - converts dataframe to image array"""
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x

def image_to_strokes(image_data):
    """Exactly as in Streamlit app - converts image to strokes"""
    # Convert image data to PIL Image
    img = Image.fromarray(image_data).convert('L')

    # Resize to 256x256 (same as Streamlit)
    img = img.resize((256, 256))

    # Convert to numpy array
    img_array = np.array(img)

    # Binarize the image
    _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract strokes
    strokes = []
    for contour in contours:
        x_points = contour[:, 0, 0].tolist()
        y_points = contour[:, 0, 1].tolist()
        strokes.append([x_points, y_points])

    return strokes

def preds2catids(predictions):
    """Get top 3 predictions"""
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image from canvas

@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    load_model()

@app.get("/")
async def root():
    return {"message": "Doodle Classifier API"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predict the drawing using the EXACT same pipeline as Streamlit app
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image.split(',')[1] if ',' in request.image else request.image)

        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_data))

        # Convert to numpy array (RGBA from canvas)
        img_array = np.array(img)

        # CRITICAL: This is the exact same processing as Streamlit
        # Convert image to strokes (same as image_to_strokes in Streamlit)
        strokes = image_to_strokes(img_array)

        # Create drawing data structure (same as save_to_ndjson in Streamlit)
        drawing_data = {'drawing': strokes}

        # Create dataframe (same as pd.read_json in Streamlit)
        df = pd.DataFrame([drawing_data])

        # Convert to image array with preprocessing (EXACT same function)
        x_test = df_to_image_array_xd(df, SIZE)

        # Predict
        prediction = model.predict(x_test)

        # Get top 3 predictions
        top3 = preds2catids(prediction)
        id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
        top3cats = top3.replace(id2cat)

        # Get the predictions
        first_row_list = top3cats.iloc[0].tolist()

        return {
            "predictions": first_row_list,
            "success": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Return all available categories"""
    return {"categories": cats}
