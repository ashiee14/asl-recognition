"""
collect_data.py

Run this once per letter to record training examples from your webcam.

HOW TO USE:
  1. Change LETTER below to the letter you're recording (e.g. "A").
  2. Run: python collect_data.py
  3. Hold your hand in the ASL shape for that letter in front of the camera.
     Move it slightly (angle, distance, position) so the model sees variety.
     The first ~1 second of frames is skipped automatically to avoid
     recording your hand still settling into the shape.
  4. Aim for ~150-300 recorded frames per letter (watch the terminal/window,
     each accepted frame appends one row to data/data.csv).
  5. Press 'q' to stop.
  6. Repeat for every letter A-Y (skip J and Z — they require motion,
     not a static handshape, and this MVP only handles static shapes).

NOTE: this writes RAW (un-normalized) landmarks to data/data.csv. That
file is the single source of truth for all your recordings. Whenever the
feature engineering in landmarks.py changes, run migrate_data.py to
regenerate data/data_normalized.csv (what train.py actually uses) from
this raw file -- nothing you've recorded is ever lost that way.
"""

import os
import time
import csv
import cv2
import mediapipe as mp
from landmarks import create_detector, extract_landmarks_raw

LETTER = "R"                       # <-- change this before each run
OUT_FILE = "data/data.csv"
SETTLE_FRAMES = 30                 # ~1 second at 30fps: let your hand settle first

VALID_LETTERS = list("ABCDEFGHI KLMNOPQRSTUVWXY".replace(" ", ""))

if LETTER not in VALID_LETTERS:
    raise ValueError(
        f"Invalid letter: {LETTER}. "
        "Only static ASL letters A-I and K-Y (excluding J) are allowed."
    )

os.makedirs("data", exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector = create_detector()

print(f"Recording letter '{LETTER}'. Get your hand into position -- "
      f"recording starts in ~1 second. Press 'q' to stop.")

frame_count = 0
recorded_count = 0

with open(OUT_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from webcam.")
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, int(time.time() * 1000))

        vec = extract_landmarks_raw(result)
        settled = frame_count > SETTLE_FRAMES

        if vec and settled:
            writer.writerow(vec + [LETTER])
            recorded_count += 1

        status = f"Recording: {LETTER} ({recorded_count})" if settled else "Get ready..."
        color = (0, 255, 0) if settled else (0, 165, 255)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Collect", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Done recording '{LETTER}'. {recorded_count} samples saved to {OUT_FILE}.")