import cv2
import mediapipe as mp
import time
import torch
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- IMPORTS ---
import user_manager
from model import get_model, normalize_landmarks, ASL_CLASSES

# --- CONFIGURATION ---
MODEL_PATH = "hand_landmarker.task"
TRAINED_MODEL_PATH = "asl_model.pth"
TARGET_HOLD_TIME = 2.0

# --- AI BRIDGE (Your existing class) ---
class ASLInferenceBridge:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading AI Model on: {self.device}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            num_classes = checkpoint.get('num_classes', len(ASL_CLASSES))
            self.model = get_model(model_type="mlp", num_classes=num_classes)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            self.idx_to_label = checkpoint.get('idx_to_label', {i: c for i, c in enumerate(ASL_CLASSES)})
            print("✅ AI Model loaded successfully!")
        except FileNotFoundError:
            print(f"❌ ERROR: Could not find '{model_path}'.")
            self.model = None

    def predict(self, landmarks_list):
        if not self.model: return "?"
        landmarks_np = np.array(landmarks_list, dtype=np.float32)
        norm_landmarks = normalize_landmarks(landmarks_np)
        input_tensor = torch.tensor(norm_landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        if confidence.item() > 0.6: 
            idx = predicted_idx.item()
            return self.idx_to_label.get(idx, "?") if isinstance(self.idx_to_label, dict) else ASL_CLASSES[idx]
        return "?"

# --- THE GAME LOOP (Moved to a function) ---
def practice_mode(user, ai_brain):
    # Setup Camera & MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    # Initial Lesson State
    current_lesson = user_manager.get_next_lesson(user)
    if not current_lesson:
        print("All lessons complete! Reset stats to play again.")
        return

    stats = user_manager.get_lesson_status(user, current_lesson)
    target_letter = min(stats, key=stats.get)
    hold_start_time = None
    
    print("\n--- STARTING PRACTICE (Press 'q' to quit) ---")

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)
        
        predicted_char = "?"
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            raw_landmarks = []
            height, width, _ = frame.shape
            for lm in hand_landmarks:
                raw_landmarks.extend([lm.x, lm.y, lm.z])
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            predicted_char = ai_brain.predict(raw_landmarks)

        # Game Logic
        color = (0, 0, 255)
        status_msg = f"Target: {target_letter} | You: {predicted_char}"

        if predicted_char == target_letter:
            color = (0, 255, 0)
            if hold_start_time is None: hold_start_time = time.time()
            elapsed = time.time() - hold_start_time
            progress = min(1.0, elapsed / TARGET_HOLD_TIME)
            cv2.rectangle(frame, (50, 200), (50 + int(200 * progress), 220), (0, 255, 0), -1)
            
            if elapsed >= TARGET_HOLD_TIME:
                user_manager.record_attempt(user.username, target_letter, True)
                current_lesson = user_manager.get_next_lesson(user)
                if current_lesson:
                    stats = user_manager.get_lesson_status(user, current_lesson)
                    target_letter = min(stats, key=stats.get)
                    hold_start_time = None
                else:
                    status_msg = "LESSON COMPLETE!"
        else:
            hold_start_time = None

        # UI
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"XP: {user.total_xp} | Streak: {user.current_streak}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("ASL Trainer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- THE MAIN MENU LOOP ---
def main():
    print("--- LAUNCHING ASL TRAINER ---")
    current_user = user_manager.login()
    if not current_user: return

    # Load AI once at startup
    ai_brain = ASLInferenceBridge(TRAINED_MODEL_PATH)

    while True:
        print(f"\n=== MAIN MENU ({current_user.username}) ===")
        print("1. Practice Mode (Camera)")
        print("2. Check Stats")
        print("3. Skip Current Level (Cheat)")
        print("4. Delete Account")
        print("5. Quit")
        
        choice = input("Select an option: ")
        
        if choice == '1':
            practice_mode(current_user, ai_brain)
        elif choice == '2':
            user_manager.print_user_stats(current_user)
        elif choice == '3':
            user_manager.skip_current_lesson(current_user)
        elif choice == '4':
            if user_manager.delete_user(current_user.username):
                return # Exit if user is deleted
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()