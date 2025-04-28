import cv2
import numpy as np
import tensorflow as tf
import joblib
import os
import string
import sys
import time
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.hand_detector import HandDetector  # Adjust if your path is different

def load_static_model_and_scaler():
    """Load the trained model and scaler."""
    model = tf.keras.models.load_model('models/letter_model.h5')
    scaler = joblib.load('models/feature_scaler.pkl')
    return model, scaler

def predict_letter(model, scaler, landmarks_array):
    """Predict the ASL letter given the landmark features."""
    if landmarks_array is None:
        return None, 0

    input_data = scaler.transform(landmarks_array.reshape(1, -1))
    prediction = model.predict(input_data, verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)

    # Map labels 0–23 to letters A–Y (skipping J and Z)
    letters = [letter for letter in string.ascii_lowercase if letter not in ['j', 'z']]
    if predicted_label < len(letters):
        return letters[predicted_label], confidence
    else:
        return None, 0

def main():
    print("Starting live static ASL translator...")

    model, scaler = load_static_model_and_scaler()
    detector = HandDetector(max_hands=1, detection_confidence=0.7)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    typed_text = ""

    # Typing parameters
    prediction_history = deque(maxlen=20)  # How many frames to average over
    required_stable_frames = 15             # How many frames same prediction needed
    last_added_letter = None
    cooldown_frames = 15                    # Prevent double-adding after typing
    cooldown_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)

        frame, results = detector.find_hands(frame)
        landmarks_array = detector.get_landmark_array(frame)

        current_letter = None
        if landmarks_array is not None:
            current_letter, confidence = predict_letter(model, scaler, landmarks_array)

            if current_letter and confidence > 0.7:  # Only if confident enough
                prediction_history.append(current_letter)
            else:
                prediction_history.append(None)
        else:
            prediction_history.append(None)

        # Check if prediction history is stable
        if len(prediction_history) == prediction_history.maxlen:
            most_common = None
            count = 0
            for letter in set(prediction_history):
                if letter is not None:
                    occurrences = prediction_history.count(letter)
                    if occurrences > count:
                        count = occurrences
                        most_common = letter

            # If a letter has been consistently predicted
            if most_common is not None and count >= required_stable_frames:
                if cooldown_counter == 0 and most_common != last_added_letter:
                    typed_text += most_common
                    print(f"Added letter: {most_common}")
                    last_added_letter = most_common
                    cooldown_counter = cooldown_frames

        # Cooldown logic
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # Draw prediction and text
        overlay_text = "No Hand Detected"
        if current_letter:
            overlay_text = f"Detected: {current_letter.upper()}"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 70), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, overlay_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Draw typed text
        cv2.rectangle(frame, (0, frame.shape[0]-70), (frame.shape[1], frame.shape[0]), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"Text: {typed_text}", (10, frame.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("ASL Static Translator", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Quitting translator...")
            break
        elif key == ord('c'):
            typed_text = ""
            last_added_letter = None
            print("Cleared typed text.")
        elif key == ord('s'):
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"typed_text_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write(typed_text)
            print(f"Saved typed text to {filename}")
        elif key == 32:  # Space key
            typed_text += " "
            print("Added space.")
        elif key == 8:   # Backspace key
            typed_text = typed_text[:-1]
            print("Deleted last character.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
