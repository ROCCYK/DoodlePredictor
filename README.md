# Doodle Classifier

This project uses a **MobileNet** architecture for classifying grayscale images. The model has been optimized to handle greyscale images efficiently, leveraging the depth and speed of MobileNet, which is typically used for lightweight image classification tasks. The project includes a data pipeline, model training, evaluation, and visualization of results, including a accuracy_loss graph to assess model performance.

## Model Overview

- **Architecture**: MobileNet, a streamlined and efficient convolutional neural network designed for mobile and embedded applications.
- **Input**: Grayscale images.
- **Output**: Multi-class classification.
- **Evaluation**: The model's performance is evaluated using metrics like accuracy and a loss.

## How to Run the App Locally

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ROCCYK/DoodlePredictor
    cd DoodlePredictor
    ```

2. **Install required dependencies**:
    Make sure you have Python 3.x installed. Then, create a virtual environment and install the dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    After installing the dependencies, you can run the Streamlit app locally using the following command:
    ```bash
    streamlit run app.py
    ```

4. **Access the app**:
    Open your web browser and go to `http://localhost:8501` to interact with the app.

## Link to Deployed Streamlit App

The app has been deployed and can be accessed via the following link:
[Deployed App Link](https://doodleclassifier.streamlit.app/)
