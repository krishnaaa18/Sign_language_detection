# Sign Language Detection 🖐️🔤  

This project is an AI-based **Sign Language Translator** that detects hand gestures (finger shapes) using a live camera feed and translates them into text in real time. The detected letters are saved in a file.

## Features 🚀
- Real-time detection of hand gestures.
- Translates sign language into text dynamically.
- Saves recognized letters to a file.
- Uses deep learning models for accurate predictions.

## Installation 🔧
1. Clone the repository:
   ```bash
   git clone git@github.com:krishnaaa18/Sign_language_detection.git
   cd Sign_language_detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the detection script:
   ```bash
   python hand_tracking.py
   ```

## Dependencies 📦
This project requires the following libraries:
- OpenCV
- TensorFlow
- Keras
- Mediapipe
- NumPy

Ensure they are installed using `pip install -r requirements.txt`.

## Model 🧠
The model is trained to recognize the ASL alphabet (A-Z) using a dataset of hand gestures. It processes live video input and predicts the corresponding letter.

## Usage 🎯
1. Run the script, and it will access your webcam.
2. Make a sign with your fingers.
3. The detected letter will be displayed on the screen and saved in a text file.

## Contributing 🤝
Feel free to contribute by opening issues or making pull requests to improve the model or add new features.

## License 📜
This project is open-source and free to use.

---
