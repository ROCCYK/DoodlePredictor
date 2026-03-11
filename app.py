import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.applications.mobilenet import preprocess_input
import json
import cv2



cats = ['airplane',
 'alarm clock',
 'ambulance',
 'angel',
 'animal migration',
 'ant',
 'anvil',
 'apple',
 'arm',
 'asparagus',
 'axe',
 'backpack',
 'banana',
 'bandage',
 'barn',
 'baseball',
 'baseball bat',
 'basket',
 'basketball',
 'bat',
 'bathtub',
 'beach',
 'bear',
 'beard',
 'bed',
 'bee',
 'belt',
 'bench',
 'bicycle',
 'binoculars',
 'bird',
 'birthday cake',
 'blackberry',
 'blueberry',
 'book',
 'boomerang',
 'bottlecap',
 'bowtie',
 'bracelet',
 'brain',
 'bread',
 'bridge',
 'broccoli',
 'broom',
 'bucket',
 'bulldozer',
 'bus',
 'bush',
 'butterfly',
 'cactus',
 'cake',
 'calculator',
 'calendar',
 'camel',
 'camera',
 'camouflage',
 'campfire',
 'candle',
 'cannon',
 'canoe',
 'car',
 'carrot',
 'castle',
 'cat',
 'ceiling fan',
 'cell phone',
 'cello',
 'chair',
 'chandelier',
 'church',
 'circle',
 'clarinet',
 'clock',
 'cloud',
 'coffee cup',
 'compass',
 'computer',
 'cookie',
 'cooler',
 'couch',
 'cow',
 'crab',
 'crayon',
 'crocodile',
 'crown',
 'cruise ship',
 'cup',
 'diamond',
 'dishwasher',
 'diving board',
 'dog',
 'dolphin',
 'donut',
 'door',
 'dragon',
 'dresser',
 'drill',
 'drums',
 'duck',
 'dumbbell',
 'ear',
 'elbow',
 'elephant',
 'envelope',
 'eraser',
 'eye',
 'eyeglasses',
 'face',
 'fan',
 'feather',
 'fence',
 'finger',
 'fire hydrant',
 'fireplace',
 'firetruck',
 'fish',
 'flamingo',
 'flashlight',
 'flip flops',
 'floor lamp',
 'flower',
 'flying saucer',
 'foot',
 'fork',
 'frog',
 'frying pan',
 'garden',
 'garden hose',
 'giraffe',
 'goatee',
 'golf club',
 'grapes',
 'grass',
 'guitar',
 'hamburger',
 'hammer',
 'hand',
 'harp',
 'hat',
 'headphones',
 'hedgehog',
 'helicopter',
 'helmet',
 'hexagon',
 'hockey puck',
 'hockey stick',
 'horse',
 'hospital',
 'hot air balloon',
 'hot dog',
 'hot tub',
 'hourglass',
 'house',
 'house plant',
 'hurricane',
 'ice cream',
 'jacket',
 'jail',
 'kangaroo',
 'key',
 'keyboard',
 'knee',
 'ladder',
 'lantern',
 'laptop',
 'leaf',
 'leg',
 'light bulb',
 'lighthouse',
 'lightning',
 'line',
 'lion',
 'lipstick',
 'lobster',
 'lollipop',
 'mailbox',
 'map',
 'marker',
 'matches',
 'megaphone',
 'mermaid',
 'microphone',
 'microwave',
 'monkey',
 'moon',
 'mosquito',
 'motorbike',
 'mountain',
 'mouse',
 'moustache',
 'mouth',
 'mug',
 'mushroom',
 'nail',
 'necklace',
 'nose',
 'ocean',
 'octagon',
 'octopus',
 'onion',
 'oven',
 'owl',
 'paint can',
 'paintbrush',
 'palm tree',
 'panda',
 'pants',
 'paper clip',
 'parachute',
 'parrot',
 'passport',
 'peanut',
 'pear',
 'peas',
 'pencil',
 'penguin',
 'piano',
 'pickup truck',
 'picture frame',
 'pig',
 'pillow',
 'pineapple',
 'pizza',
 'pliers',
 'police car',
 'pond',
 'pool',
 'popsicle',
 'postcard',
 'potato',
 'power outlet',
 'purse',
 'rabbit',
 'raccoon',
 'radio',
 'rain',
 'rainbow',
 'rake',
 'remote control',
 'rhinoceros',
 'river',
 'roller coaster',
 'rollerskates',
 'sailboat',
 'sandwich',
 'saw',
 'saxophone',
 'school bus',
 'scissors',
 'scorpion',
 'screwdriver',
 'sea turtle',
 'see saw',
 'shark',
 'sheep',
 'shoe',
 'shorts',
 'shovel',
 'sink',
 'skateboard',
 'skull',
 'skyscraper',
 'sleeping bag',
 'smiley face',
 'snail',
 'snake',
 'snorkel',
 'snowflake',
 'snowman',
 'soccer ball',
 'sock',
 'speedboat',
 'spider',
 'spoon',
 'spreadsheet',
 'square',
 'squiggle',
 'squirrel',
 'stairs',
 'star',
 'steak',
 'stereo',
 'stethoscope',
 'stitches',
 'stop sign',
 'stove',
 'strawberry',
 'streetlight',
 'string bean',
 'submarine',
 'suitcase',
 'sun',
 'swan',
 'sweater',
 'swing set',
 'sword',
 't-shirt',
 'table',
 'teapot',
 'teddy-bear',
 'telephone',
 'television',
 'tennis racquet',
 'tent',
 'The Eiffel Tower',
 'The Great Wall of China',
 'The Mona Lisa',
 'tiger',
 'toaster',
 'toe',
 'toilet',
 'tooth',
 'toothbrush',
 'toothpaste',
 'tornado',
 'tractor',
 'traffic light',
 'train',
 'tree',
 'triangle',
 'trombone',
 'truck',
 'trumpet',
 'umbrella',
 'underwear',
 'van',
 'vase',
 'violin',
 'washing machine',
 'watermelon',
 'waterslide',
 'whale',
 'wheel',
 'windmill',
 'wine bottle',
 'wine glass',
 'wristwatch',
 'yoga',
 'zebra',
 'zigzag']
def print_title():
    st.markdown(
        """
        <div style='display:flex;flex-direction:column;gap:6px'>
            <h1 style='margin-bottom:0'>✏️ Doodle Classifier</h1>
            <p style='margin-top:0;color:#4b5563'>
                Draw a doodle and get instant Top-3 predictions from a MobileNetV1 model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "The architecture employed for this **Convolutional Neural Network (CNN)** doodle classifier is based on the **MobileNetV1** model. The classifier is trained using Google’s **Quick, Draw!** dataset. To enhance model performance and ensure robustness against variations in real-world doodling, the doodles are randomly augmented through rotations, shifts, shearing, zooming, and pixelation. Created by Rhichard Koh")


def print_credits():
    st.write(
        "Dataset from: Google's Quick, Draw! Dataset, with 50,000,000 doodles in total. The Dataset consists of 340 different classes. Created by Rhichard Koh")


def show_canvas_opts():
    drawing_mode = st.selectbox(
        "🖌️ Drawing mode",
        ("freedraw", "transform"),
        index=0,  # default: freedraw
        key="drawing_mode",
    )
    stroke_width = st.slider(":straight_ruler: Stroke width", 1, 25, 14)
    return drawing_mode, stroke_width


def draw_canvas(drawing_mode, stroke_width):
    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        background_color="#FFFFFF",  # white
        update_streamlit=True,
        height=320,
        width=320,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    return canvas_result

def get_info():
    st.markdown("## 📖 About This Model")
    st.write("This model uses a **Convolutional Neural Network** with a **MobileNetV1** architecture to categorize hand drawn doodles.")
    st.markdown(f"- **Version:** MobileNetV1-style model fine-tuned on Quick, Draw classes")
    st.markdown(f"- **Categories:** {len(cats)}")

    with st.expander("Supported categories", expanded=False):
        st.write(" , ".join(cats))
        st.image("dataset.png", use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Architecture", "Training", "Notes"])
    with tab1:
        st.image("MobileNet-V1-architecture.png", use_container_width=True)
    with tab2:
        st.image("accuracy_loss_chart.png", use_container_width=True)
    with tab3:
        st.caption("Trained with rotation, shifts, shearing, zooming, and pixel-level augmentation.")

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def apply_page_style():
    st.markdown(
        """
        <style>
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }
        [data-testid="stSidebar"] {
            background: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_top_predictions(prediction, top_k=3):
    probs = prediction[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    return pd.DataFrame(
        [{"label": cats[idx], "confidence": float(probs[idx] * 100)} for idx in top_idx]
    )

@st.cache_resource
def load_model():
    model = MobileNet(input_shape=(64, 64, 1), alpha=1.0, weights=None, classes=len(cats))
    model.load_weights('model.h5')
    model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss='categorical_crossentropy',
        metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy]
    )
    return model

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
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
    # df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x


def image_to_strokes(image_path):
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Resize the image to a manageable size (optional, can use the original size)
    img = img.resize((256, 256))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Binarize the image (convert to black and white)
    _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract strokes
    strokes = []
    for contour in contours:
        x_points = contour[:, 0, 0].tolist()
        y_points = contour[:, 0, 1].tolist()
        strokes.append([x_points, y_points])

    return strokes


def save_to_ndjson(strokes, output_path):
    # Create the structure to match the Quick, Draw! dataset
    drawing_data = {
        'drawing': strokes
    }

    # Write to NDJSON file
    with open(output_path, 'w') as f:
        f.write(json.dumps(drawing_data) + '\n')

def main():
    st.set_page_config(layout="wide", page_title="Doodle Classifier", page_icon="✏️")
    apply_page_style()

    print_title()
    st.markdown("""---""")
    with st.expander("How to use", expanded=False):
        st.write("Draw a doodle on the canvas and click **Predict Drawing** to view your Top-3 classes.")

    with st.sidebar:
        st.markdown("## Controls")
        drawing_mode, stroke_width = show_canvas_opts()

    left_col, right_col = st.columns([1.05, 1.55], gap="large")
    with left_col:
        st.markdown("### 🎨 Canvas")
        canvas_result = draw_canvas(drawing_mode, stroke_width)
        predict_btn = st.button(":mag: Predict Drawing", use_container_width=True, type="primary")

    with right_col:
        st.markdown("### 📊 Top 3 Predictions")
        result_box = st.container()

    if predict_btn:
        if canvas_result.image_data is not None:
            with result_box:
                with st.spinner("Running inference..."):
                    img = Image.fromarray(canvas_result.image_data)
                    img.save("drawn_image.png")
                    strokes = image_to_strokes("drawn_image.png")
                    save_to_ndjson(strokes, "output.ndjson")

                    df = pd.read_json("output.ndjson", lines=True)
                    x_test = df_to_image_array_xd(df, size=64)
                    prediction = load_model().predict(x_test)
                    top3_df = get_top_predictions(prediction, top_k=3)

                    st.dataframe(
                        top3_df.assign(confidence=top3_df["confidence"].round(2).map(lambda x: f"{x:.2f}%")),
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.markdown("### Confidence")
                    for _, row in top3_df.iterrows():
                        st.progress(int(row["confidence"]), text=f"{row['label']} ({row['confidence']:.2f}%)")
        else:
            with result_box:
                st.warning("Please draw something before submitting.")

    st.markdown("""---""")
    get_info()
    st.markdown("""---""")
    print_credits()


if __name__ == "__main__":
    main()
