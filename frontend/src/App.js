import React, { useRef, useState } from 'react';
import SignatureCanvas from 'react-signature-canvas';
import './App.css';

function App() {
  const sigCanvas = useRef(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const clearCanvas = () => {
    sigCanvas.current.clear();
    setPredictions([]);
    setError(null);
  };

  const predictDrawing = async () => {
    if (sigCanvas.current.isEmpty()) {
      setError('Please draw something first!');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Get the canvas as a data URL
      // CRITICAL: This is exactly how the Streamlit canvas provides the image
      const canvas = sigCanvas.current.getCanvas();
      const imageData = canvas.toDataURL('image/png');

      // Send to backend
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData
        })
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictions(data.predictions);
    } catch (err) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>✏️ Doodle Classifier</h1>
        <p>Draw a doodle and let the AI predict what it is!</p>
      </header>

      <div className="main-content">
        <div className="canvas-container">
          <h2>Draw Here</h2>
          <div className="canvas-wrapper">
            <SignatureCanvas
              ref={sigCanvas}
              canvasProps={{
                width: 300,
                height: 300,
                className: 'signature-canvas'
              }}
              backgroundColor="#FFFFFF"
              penColor="black"
              minWidth={2}
              maxWidth={4}
            />
          </div>

          <div className="button-group">
            <button onClick={predictDrawing} disabled={loading} className="predict-button">
              {loading ? 'Predicting...' : '🔍 Predict Drawing'}
            </button>
            <button onClick={clearCanvas} className="clear-button">
              🗑️ Clear Canvas
            </button>
          </div>

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </div>

        {predictions.length > 0 && (
          <div className="predictions-container">
            <h2>Top 3 Predictions</h2>
            <div className="predictions-list">
              {predictions.map((prediction, index) => (
                <div key={index} className="prediction-item">
                  <span className="prediction-rank">#{index + 1}</span>
                  <span className="prediction-label">{prediction}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <footer className="App-footer">
        <p>
          The architecture employed for this Convolutional Neural Network (CNN) doodle classifier
          is based on the MobileNetV1 model. The classifier is trained using Google's Quick, Draw! dataset.
        </p>
      </footer>
    </div>
  );
}

export default App;
