import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PATH = "hand_landmarker.task"
DATA_FILE = "asl_data.csv"
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
]

def init_csv():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ["label"]
            for i in range(21):
                headers.extend([f"x{i}", f"y{i}", f"z{i}"])
            writer.writerow(headers)

def handtrack():
    init_csv()
    
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(PATH), num_hands=2)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            mp_results = landmarker.detect(mp_image)
            key = cv2.waitKey(1) & 0xFF

            # Draw landmarks AND connections
            if mp_results.hand_landmarks:
                height, width, _ = frame.shape
                for hand_landmarks in mp_results.hand_landmarks:
                    # Draw lines between connected landmarks
                    for connection in HAND_CONNECTIONS:
                        landmark1 = hand_landmarks[connection[0]]
                        landmark2 = hand_landmarks[connection[1]]
                        cx1, cy1 = int(landmark1.x * width), int(landmark1.y * height)
                        cx2, cy2 = int(landmark2.x * width), int(landmark2.y * height)
                        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                    
                    # Draw dots on landmarks
                    for landmark in hand_landmarks:
                        cx, cy = int(landmark.x * width), int(landmark.y * height)
                        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    if 97 <= key <= 122: # If 'a' through 'z' is pressed
                        letter = chr(key).upper()
                        
                        # 1. Normalize (Center the hand at 0,0 relative to wrist)
                        wrist = hand_landmarks[0]
                        row = [letter]
                        for lm in hand_landmarks:
                            row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                        
                        # 2. Save to CSV
                        with open(DATA_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
                        
                        print(f"Saved {letter}")
                        cv2.putText(frame, f"Saved: {letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    handtrack()
    
