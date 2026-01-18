import cv2
import mediapipe as mp
import time
import torch
import numpy as np
import subprocess
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- IMPORTS ---
import user_manager
from model import get_model, normalize_landmarks, ASL_CLASSES
from inference import HoldToConfirm

# --- CONFIGURATION ---
MODEL_PATH = "hand_landmarker.task"
TRAINED_MODEL_PATH = "asl_model.pth"

# --- VISUAL STYLE ---
COLOR_DOT = (0, 255, 255)   # Yellow
COLOR_LINE = (0, 0, 255)    # Red

HAND_CONNECTIONS = [
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

# --- AI BRIDGE ---
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
        if not self.model: return "?", 0.0
        
        landmarks_np = np.array(landmarks_list, dtype=np.float32)
        norm_landmarks = normalize_landmarks(landmarks_np)
        input_tensor = torch.tensor(norm_landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        idx = predicted_idx.item()
        label = self.idx_to_label.get(idx, "?") if isinstance(self.idx_to_label, dict) else ASL_CLASSES[idx]
        return label, confidence.item()

# --- DRAWING HELPER ---
def draw_skeleton(frame, hand_landmarks):
    height, width, _ = frame.shape
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        p1 = (int(start.x * width), int(start.y * height))
        p2 = (int(end.x * width), int(end.y * height))
        cv2.line(frame, p1, p2, COLOR_LINE, 2)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (cx, cy), 5, COLOR_DOT, -1)

# --- GAME MODE: PRACTICE ---
def practice_mode(user, ai_brain, specific_lesson=None):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    # Setup Lesson
    if specific_lesson:
        current_lesson = specific_lesson
        print(f"\n--- REVIEW MODE: {current_lesson.title} ---")
    else:
        current_lesson = user_manager.get_next_lesson(user)
        if not current_lesson:
            print("All lessons complete! Use 'Select Level' to review.")
            return
        print(f"\n--- CAMPAIGN MODE: {current_lesson.title} ---")

    stats = user_manager.get_lesson_status(user, current_lesson)
    target_letter = min(stats, key=stats.get)
    
    hold_tracker = HoldToConfirm(hold_time=2.0, confidence_threshold=0.6)
    print(f"TASK: Sign '{target_letter}' (Press 'q' to quit)")

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)
        
        predicted_char = "?"
        confidence = 0.0
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            draw_skeleton(frame, hand_landmarks)
            
            raw_landmarks = []
            for lm in hand_landmarks:
                raw_landmarks.extend([lm.x, lm.y, lm.z])
            
            predicted_char, confidence = ai_brain.predict(raw_landmarks)

        # GAME LOGIC
        current_sign, hold_progress, is_confirmed = hold_tracker.update(predicted_char, confidence)

        color = (0, 0, 255)
        status_msg = f"Target: {target_letter} | You: {current_sign}"

        if current_sign == target_letter:
            color = (0, 255, 0)
            cv2.rectangle(frame, (50, 200), (50 + int(200 * hold_progress), 220), (0, 255, 0), -1)
            
            if is_confirmed:
                user_manager.record_attempt(user.username, target_letter, True)
                hold_tracker.hold_start = None 
                
                if specific_lesson:
                    stats = user_manager.get_lesson_status(user, current_lesson)
                    target_letter = min(stats, key=stats.get)
                    status_msg = f"CORRECT! Next: {target_letter}"
                else:
                    current_lesson = user_manager.get_next_lesson(user)
                    if current_lesson:
                        stats = user_manager.get_lesson_status(user, current_lesson)
                        target_letter = min(stats, key=stats.get)
                        status_msg = f"CORRECT! Next: {target_letter}"
                    else:
                        status_msg = "LESSON COMPLETE!"

        # UI
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"XP: {user.total_xp} | Streak: {user.current_streak}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("ASL Trainer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# --- HELPER: LAUNCH SUBPROCESS ---
def run_playground_script():
    """
    Launches the standalone inference.py script exactly as requested.
    """
    print("\n🚀 Launching ASL Playground (Inference Engine)...")
    print("   (Close the popup window to return to menu)")
    
    # This runs: python inference.py --model asl_model.pth
    try:
        import os
        script_path = "inference.py"
        if os.path.exists("src/inference.py"):
            script_path = "src/inference.py"

        subprocess.run([sys.executable, script_path, "--model", "asl_model.pth"])
    except Exception as e:
        print(f"❌ Error launching playground: {e}")

# --- THE MAIN MENU LOOP ---
def main():
    print("--- LAUNCHING ASL TRAINER ---")
    current_user = user_manager.login()
    if not current_user: return

    # Load AI once for Game Mode
    ai_brain = ASLInferenceBridge(TRAINED_MODEL_PATH)

    while True:
        print(f"\n=== MAIN MENU ({current_user.username}) ===")
        print("1. Practice Next Level (Auto)")
        print("2. Select Level to Practice")
        print("3. Check Level Stats")
        print("4. Global Stats")
        print("5. Skip Current Level (Cheat)")
        print("6. Delete Account")
        print("7. ASL Playground (Free Mode)")
        print("8. Quit")
        
        choice = input("Select an option: ")
        
        if choice == '1':
            practice_mode(current_user, ai_brain)
        elif choice == '2':
            selected_lesson = user_manager.select_lesson_menu()
            if selected_lesson:
                practice_mode(current_user, ai_brain, specific_lesson=selected_lesson)
        elif choice == '3':
            user_manager.check_lesson_stats(current_user)
        elif choice == '4':
            user_manager.print_user_stats(current_user)
        elif choice == '5':
            user_manager.skip_current_lesson(current_user)
        elif choice == '6':
            if user_manager.delete_user(current_user.username): return
        elif choice == '7':
            run_playground_script()
        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()