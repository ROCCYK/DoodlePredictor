# Doodle Classifier - FastAPI Backend + React Frontend

This is a conversion of the Streamlit Doodle Classifier app to a modern FastAPI backend with a React frontend.

## Key Features

- **Exact Image Processing Pipeline**: The backend uses the EXACT same image processing pipeline as the original Streamlit app to ensure predictions work correctly.
- **FastAPI Backend**: High-performance async backend with model loading at startup.
- **React Frontend**: Modern, responsive UI with canvas drawing.
- **MobileNetV1 Model**: Trained on Google's Quick, Draw! dataset with 340 categories.

## Project Structure

```
DoodlePredictor/
├── backend/
│   ├── main.py              # FastAPI server with exact image processing
│   └── requirements.txt     # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js          # React app with canvas
│   │   └── App.css         # Styling
│   └── package.json        # Frontend dependencies
├── model.h5                # Pre-trained model weights
└── app.py                  # Original Streamlit app
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the FastAPI server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will start at `http://localhost:3000`

## How It Works

### Image Processing Pipeline (Critical for Correct Predictions)

The key to making predictions work correctly is using the EXACT same image processing pipeline as the Streamlit app:

1. **Canvas to Image**: React canvas outputs as PNG data URL
2. **Image to Strokes**: Convert image to contours using OpenCV (same as Streamlit)
3. **Strokes to Drawing**: Create the same data structure as Quick, Draw! dataset
4. **Drawing to Array**: Use `df_to_image_array_xd()` function (exact copy from Streamlit)
5. **Preprocessing**: Apply MobileNet preprocessing (exact same function)
6. **Prediction**: Feed to the model

### Backend Endpoints

- `GET /` - Health check
- `POST /predict` - Predict drawing from base64 encoded image
- `GET /categories` - Get list of all 340 categories

### Frontend Features

- Canvas drawing with adjustable stroke width
- Clear canvas button
- Predict button that sends image to backend
- Display top 3 predictions
- Responsive design

## Why This Works (vs. Your Previous Attempt)

The previous implementation likely failed because the image wasn't processed in the exact same way as the Streamlit app. This implementation:

1. **Exact Functions**: Copies the `draw_cv2()`, `df_to_image_array_xd()`, and `image_to_strokes()` functions exactly from the Streamlit app.
2. **Same Data Flow**: Follows the exact same steps: Canvas → Image → Strokes → DataFrame → Array → Preprocessing → Prediction.
3. **Correct Canvas Format**: Uses a canvas that outputs RGBA image data, just like Streamlit's drawable canvas.
4. **Proper Preprocessing**: Applies MobileNet's `preprocess_input()` in the exact same way.

## Testing

1. Start both backend and frontend servers
2. Open `http://localhost:3000` in your browser
3. Draw a simple doodle (e.g., a circle, a house, a cat)
4. Click "Predict Drawing"
5. See the top 3 predictions

## Troubleshooting

### Model Not Loading
- Ensure `model.h5` is in the parent directory of `backend/`
- Check the path in `backend/main.py` (line 86): `model.load_weights('../model.h5')`

### CORS Errors
- The backend is configured to allow CORS from `http://localhost:3000`
- If using a different port, update the CORS settings in `backend/main.py`

### Poor Predictions
- Make sure you're drawing clearly
- Try drawing simpler objects first
- Ensure the backend loaded successfully (check console logs)

## Model Information

- **Architecture**: MobileNetV1
- **Input Size**: 64x64 grayscale images
- **Output**: 340 classes
- **Dataset**: Google's Quick, Draw! (50M drawings)
- **Training**: Augmented with rotations, shifts, shearing, zooming, and pixelation

## Credits

- Original Streamlit app created by Rhichard Koh
- Dataset from Google's Quick, Draw!
- Converted to FastAPI + React with exact image processing pipeline
