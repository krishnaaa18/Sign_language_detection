import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("sign_language_model.keras")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Space", "Delete", "Nothing"] 

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    """Extracts landmark positions from hand tracking data."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

# File to save detected letters
output_file = "detected_text.txt"
with open(output_file, "w") as f:
    f.write("")  # Clear file contents

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip and process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract features and reshape
            landmarks = extract_landmarks(hand_landmarks)
            input_data = np.expand_dims(landmarks, axis=0)
            
            # Predict sign
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            detected_letter = labels[predicted_class]
            
            # Display detected letter
            cv2.putText(frame, detected_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save letter to file
            with open(output_file, "a") as f:
                f.write(detected_letter)
    
    cv2.imshow("Sign Language Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
