import cv2
import time
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model import ASLLandmarkMLP, ASLLandmarkNet, ASL_CLASSES, get_model

HAND_LANDMARKER_PATH = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]


class HoldToConfirm:
    def __init__(self, hold_time: float = 1.0, confidence_threshold: float = 0.7):
        self.hold_time = hold_time
        self.confidence_threshold = confidence_threshold
        self.current_letter = None
        self.hold_start = None
        self.confirmed_text = ""

    def update(self, predicted_letter: str, confidence: float) -> tuple[str, float, bool]:
        now = time.time()

        if confidence < self.confidence_threshold:
            self.current_letter = None
            self.hold_start = None
            return predicted_letter, 0.0, False

        if predicted_letter != self.current_letter:
            self.current_letter = predicted_letter
            self.hold_start = now
            return predicted_letter, 0.0, False

        hold_progress = (now - self.hold_start) / self.hold_time

        if hold_progress >= 1.0:
            if predicted_letter == 'del':
                self.confirmed_text = self.confirmed_text[:-1]
            elif predicted_letter == 'space':
                self.confirmed_text += ' '
            elif predicted_letter != 'nothing':
                self.confirmed_text += predicted_letter

            self.current_letter = None
            self.hold_start = None
            return predicted_letter, 1.0, True

        return predicted_letter, hold_progress, False


class ASLRecognizer:
    def __init__(self, model_path: str = None, model_type: str = "mlp", hold_time: float = 1.0):
        self.options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(self.options)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model = get_model(checkpoint.get('model_type', model_type),
                                   checkpoint.get('num_classes', 29))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.idx_to_label = checkpoint.get('idx_to_label', {i: c for i, c in enumerate(ASL_CLASSES)})
        else:
            self.model = get_model(model_type)
            self.idx_to_label = {i: c for i, c in enumerate(ASL_CLASSES)}

        self.model = self.model.to(self.device)
        self.model.eval()

        self.hold_tracker = HoldToConfirm(hold_time=hold_time)
        self.last_landmarks = None

    def extract_landmarks(self, frame) -> tuple[np.ndarray | None, list | None]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return None, None

        hand_landmarks = results.hand_landmarks[0]

        wrist = hand_landmarks[0]
        middle_mcp = hand_landmarks[9]

        scale_factor = ((wrist.x - middle_mcp.x)**2 +
                        (wrist.y - middle_mcp.y)**2 +
                        (wrist.z - middle_mcp.z)**2)**0.5

        if scale_factor == 0:
            scale_factor = 1

        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([
                (lm.x - wrist.x) / scale_factor,
                (lm.y - wrist.y) / scale_factor,
                (lm.z - wrist.z) / scale_factor
            ])

        return np.array(landmarks, dtype=np.float32), hand_landmarks

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)

        letter = self.idx_to_label[predicted.item()]
        return letter, confidence.item()

    def draw_hand(self, frame, hand_landmarks):
        h, w = frame.shape[:2]

        for connection in HAND_CONNECTIONS:
            lm1 = hand_landmarks[connection[0]]
            lm2 = hand_landmarks[connection[1]]
            x1, y1 = int(lm1.x * w), int(lm1.y * h)
            x2, y2 = int(lm2.x * w), int(lm2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    def draw_ui(self, frame, letter: str, confidence: float, hold_progress: float, confirmed: bool):
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (10, 10), (200, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (200, 80), (255, 255, 255), 2)

        color = (0, 255, 0) if confirmed else (255, 255, 255)
        cv2.putText(frame, letter, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"{confidence*100:.0f}%", (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        bar_width = 180
        bar_height = 20
        bar_x = 10
        bar_y = 90

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        progress_width = int(bar_width * hold_progress)
        if progress_width > 0:
            bar_color = (0, 255, 0) if hold_progress >= 1.0 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

        text_y = h - 30
        cv2.rectangle(frame, (0, text_y - 40), (w, h), (0, 0, 0), -1)
        display_text = self.hold_tracker.confirmed_text[-50:] if len(self.hold_tracker.confirmed_text) > 50 else self.hold_tracker.confirmed_text
        cv2.putText(frame, display_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, "Hold pose to confirm | 'c' clear | 'q' quit", (10, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def run(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("ASL Recognizer started. Press 'q' to quit, 'c' to clear text.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks, hand_landmarks = self.extract_landmarks(frame)

            if landmarks is not None:
                letter, confidence = self.predict(landmarks)
                letter, hold_progress, confirmed = self.hold_tracker.update(letter, confidence)
                self.draw_hand(frame, hand_landmarks)
            else:
                letter, confidence, hold_progress, confirmed = "---", 0.0, 0.0, False

            frame = self.draw_ui(frame, letter, confidence, hold_progress, confirmed)
            cv2.imshow("ASL Recognizer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.hold_tracker.confirmed_text = ""

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

        print(f"\nFinal text: {self.hold_tracker.confirmed_text}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASL Real-time Recognition")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (.pth)")
    parser.add_argument("--model-type", type=str, default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--hold-time", type=float, default=1.0, help="Seconds to hold pose")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")

    args = parser.parse_args()

    recognizer = ASLRecognizer(
        model_path=args.model,
        model_type=args.model_type,
        hold_time=args.hold_time
    )
    recognizer.run(camera_id=args.camera)
