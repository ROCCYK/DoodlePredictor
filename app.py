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
    st.title('✏️ Doodle Classifier')
    st.markdown(
        "The architecture employed for this **Convolutional Neural Network (CNN)** doodle classifier is based on the **MobileNetV1** model. The classifier is trained using Google’s **Quick, Draw!** dataset. To enhance model performance and ensure robustness against variations in real-world doodling, the doodles are randomly augmented through rotations, shifts, shearing, zooming, and pixelation. Created by Rhichard Koh")


def print_credits():
    st.write(
        "Dataset from: Google's Quick, Draw! Dataset, with 50,000,000 doodles in total. The Dataset consists of 340 different classes. Created by Rhichard Koh")


def show_canvas_opts():
    expander1 = st.expander(label=':gear: Canvas Options')
    with expander1:
        col1, col2 = st.columns(2)
        with col1:
            drawing_mode = st.selectbox(
                ":lower_left_paintbrush: Drawing Options:",
                ("freedraw", "transform"),
                index=0,  # Set the default index to 0 (freedraw)
                key="drawing_mode",  # Provide a unique key to the widget to ensure it updates correctly
            )
        with col2:
            stroke_width = st.slider(":straight_ruler: Stroke width: ", 1, 25, 14)
        return drawing_mode, stroke_width


def draw_canvas(drawing_mode, stroke_width):
    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        background_color="#FFFFFF",  # white
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    return canvas_result

def get_info():
    st.markdown("## :book: About This Model")
    st.write("This model uses a **Convolutional Neural Network** with a **MobileNetV1** architecture to categorize hand drawn doodles.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Below are categories trained on.")
        st.write("'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag'")
        st.image("dataset.png")
    with col2:
        st.markdown("#### Below is the MobileNetV1's architecture.")
        st.image("MobileNet-V1-architecture.png")
        st.image("accuracy_loss_chart.png")

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

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
    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide", page_title="️Doodle Classifier", page_icon="✏️")

    print_title()
    st.markdown("""---""")

    drawing_mode, stroke_width = show_canvas_opts()

    # Specify canvas parameters in application
    col1, col2 = st.columns([1, 2.5])
    with col1:
        canvas_result = draw_canvas(drawing_mode, stroke_width)

    # # Add a button to submit the drawing
    if st.button(":mag: Predict Drawing"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data)
            # Save the image to a file
            img.save("drawn_image.png")

            strokes = image_to_strokes("drawn_image.png")
            save_to_ndjson(strokes, "output.ndjson")

            NCATS = 340
            size = 64
            model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
            model.load_weights('model.h5')
            model.compile(optimizer=Adam(learning_rate=0.002), loss='categorical_crossentropy',
                          metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

            df = pd.read_json('output.ndjson', lines=True)
            x_test = df_to_image_array_xd(df, size)
            prediction = model.predict(x_test)
            top3 = preds2catids(prediction)
            id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
            top3cats = top3.replace(id2cat)


            with col2:
                first_row_list = top3cats.iloc[0].tolist()
                first_row_string = ' '.join(map(str, first_row_list))
                st.markdown(f"### :clipboard: Top 3 Predictions: **`{first_row_string}`**")


        else:
            st.warning("Please draw something before submitting.")
    st.markdown("""---""")
    get_info()
    st.markdown("""---""")
    print_credits()


if __name__ == "__main__":
    main()