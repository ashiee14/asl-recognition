
"""
infer.py

Runs the real-time ASL recognition pipeline:
  webcam -> hand landmarks -> letter prediction -> sentence builder -> on-screen text

Recognizes STATIC ASL letters using a trained machine learning model.

HOW TO USE:
  Run after training the model (train.py):
    python infer.py

Controls:
  - Hold a letter shape steady to have it accepted into the current word.
  - Pull your hand out of frame briefly to end the current word.
  - Press 'c' to clear the sentence.
  - Press 'q' to quit.
"""

import time

import cv2
import joblib
import mediapipe as mp

from landmarks import create_detector, extract_hand_points, normalize_landmarks
from sentence_builder import SentenceBuilder

MODEL_PATH = "models/asl_letter_model.pkl"


# --- Setup ---------------------------------------------------------------

model = joblib.load(MODEL_PATH)

detector = create_detector()

builder = SentenceBuilder(
    stability_frames=8,
    pause_frames=20
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# --- Main Recognition Loop ------------------------------------------------

while True:

    ok, frame = cap.read()

    if not ok:
        break

    # Convert frame to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect hand landmarks
    result = detector.detect_for_video(
        mp_image,
        int(time.time() * 1000)
    )

    points = extract_hand_points(result)

    letter = None

    # Predict static ASL letter
    if points:

        vec = normalize_landmarks(points)

        letter = model.predict([vec])[0]

    # Update sentence builder
    builder.update(letter)

    # Display predicted letter
    if letter:

        cv2.putText(
            frame,
            str(letter),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3
        )

    # Display constructed sentence
    cv2.putText(
        frame,
        builder.get_display_text(),
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Show webcam window
    cv2.imshow("ASL", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        builder.reset()


# --- Cleanup --------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()
