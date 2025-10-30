# Doodle Classifier

This project uses a **MobileNet** architecture for classifying grayscale images. The model has been optimized to handle greyscale images efficiently, leveraging the depth and speed of MobileNet, which is typically used for lightweight image classification tasks. The project includes a data pipeline, model training, evaluation, and visualization of results, including a accuracy_loss graph to assess model performance.

## Model Overview

- **Architecture**: MobileNet, a streamlined and efficient convolutional neural network designed for mobile and embedded applications.
- **Input**: Grayscale images.
- **Output**: Multi-class classification.
- **Evaluation**: The model's performance is evaluated using metrics like accuracy and loss.

## How to Run the App Locally (Apple Silicon / macOS)

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ROCCYK/DoodlePredictor
    cd DoodlePredictor
    ```

2. **Prerequisites**:
    - macOS on Apple Silicon (M1/M2/M3)
    - Python 3.11 (required for the pinned TensorFlow wheels)
      - Install via Homebrew if needed:
        ```bash
        brew install python@3.11
        ```

3. **Create a virtual environment and install dependencies**:
    ```bash
    /opt/homebrew/bin/python3.11 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    python -m pip install -U pip wheel setuptools
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:
    After installing the dependencies, you can run the Streamlit app locally using the following command:
    ```bash
    streamlit run app.py
    ```

5. **Access the app**:
    Open your web browser and go to `http://localhost:8501` to interact with the app.

### Notes
- This project uses `tensorflow-macos==2.15.0` and `tensorflow-metal==1.1.0`, which provide prebuilt wheels for Python 3.11 on Apple Silicon. Using Python 3.12/3.13 may fail to install TensorFlow.
- If you recreate the environment in the future, repeat the steps above. To make reinstalls reproducible, you can lock versions:
  ```bash
  pip freeze > requirements-lock.txt
  # Next time:
  pip install -r requirements-lock.txt
  ```

### Troubleshooting
- If you previously installed `tensorflow` (CPU-only) in this repo, remove it before reinstalling:
  ```bash
  pip uninstall -y tensorflow
  ```
- If you encounter excessive TensorFlow logging, you can run with:
  ```bash
  TF_CPP_MIN_LOG_LEVEL=2 streamlit run app.py
  ```

## Link to Deployed Streamlit App

The app has been deployed and can be accessed via the following link:
[Deployed App Link](https://doodleclassifier.streamlit.app/)
