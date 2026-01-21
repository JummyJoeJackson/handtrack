import cv2
import sys
import time
import torch
import subprocess
import os
import user_manager
import numpy as np
import pyfiglet
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from inference import HoldToConfirm
from model import get_model, normalize_landmarks, ASL_CLASSES

# Suppress MediaPipe/TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

from contextlib import contextmanager, redirect_stderr, redirect_stdout

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull at the FD level."""
    # On Windows, redirecting FDs while Rich is active can cause OSError [WinError 1].
    # We'll use a safer approach: only redirect FD 2 (stderr) which has most noise,
    # and handle FD 1 (stdout) carefully if needed.
    
    stderr_fd = sys.stderr.fileno()
    stdout_fd = sys.stdout.fileno()
    
    # Duplicate original FDs
    old_stderr = os.dup(stderr_fd)
    old_stdout = os.dup(stdout_fd)
    
    try:
        with open(os.devnull, 'w') as devnull:
            # Force redirection at FD level
            os.dup2(devnull.fileno(), stderr_fd)
            os.dup2(devnull.fileno(), stdout_fd)
            yield
    finally:
        # Restore original FDs
        os.dup2(old_stderr, stderr_fd)
        os.dup2(old_stdout, stdout_fd)
        os.close(old_stderr)
        os.close(old_stdout)

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "title": "bold magenta"
})
console = Console(theme=custom_theme)


MODEL_PATH = "hand_landmarker.task"
TRAINED_MODEL_PATH = "asl_model.pth"
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


class ASLInferenceBridge:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[info]Loading AI Model on {self.device}...[/info]")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            num_classes = checkpoint.get('num_classes', len(ASL_CLASSES))
            self.model = get_model(model_type="mlp", num_classes=num_classes)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            self.idx_to_label = checkpoint.get('idx_to_label', {i: c for i, c in enumerate(ASL_CLASSES)})
            console.print("✅ [success]AI Model loaded successfully![/success]")
        except FileNotFoundError:
            console.print(f"❌ [error]ERROR: Could not find '{model_path}'.[/error]")
            self.model = None

    def predict(self, landmarks_list, mirror=True):
        if not self.model: return "?", 0.0

        landmarks_np = np.array(landmarks_list, dtype=np.float32)
        norm_landmarks = normalize_landmarks(landmarks_np, mirror=mirror)
        input_tensor = torch.tensor(norm_landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        idx = predicted_idx.item()
        label = self.idx_to_label.get(idx, "?") if isinstance(self.idx_to_label, dict) else ASL_CLASSES[idx]
        return label, confidence.item()


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


def practice_mode(user, ai_brain, specific_lesson=None):
    # Setup Lesson
    if specific_lesson:
        current_lesson = specific_lesson
        console.print(Panel(f"[title]REVIEW MODE: {current_lesson.title}[/title]", expand=False))
    else:
        current_lesson = user_manager.get_next_lesson(user)
        if not current_lesson:
            console.print("[warning]All lessons complete! Use 'Select Level' to review.[/warning]")
            return
        console.print(Panel(f"[title]CAMPAIGN MODE: {current_lesson.title}[/title]", expand=False))

    with suppress_stdout_stderr():
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Warm-up detection to catch and suppress initial logs
        success, frame = cap.read()
        if success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect(mp_image)

    stats = user_manager.get_lesson_status(user, current_lesson)
    target_letter = min(stats, key=stats.get)
    
    hold_tracker = HoldToConfirm(hold_time=2.0, confidence_threshold=0.6)
    console.print(f"[bold]TASK:[/bold] Sign [bold green]'{target_letter}'[/bold green] (Press 'q' in camera window to quit)")

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Silence MediaPipe detection logs
        with suppress_stdout_stderr():
            detection_result = landmarker.detect(mp_image)
        
        predicted_char = "?"
        confidence = 0.0
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            draw_skeleton(frame, hand_landmarks)
            
            raw_landmarks = []
            for lm in hand_landmarks:
                raw_landmarks.extend([lm.x, lm.y, lm.z])
            
            # Silence inference prediction logs if any
            with suppress_stdout_stderr():
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


def run_playground_script():
    """
    Launches the standalone inference.py script exactly as requested.
    """
    console.print("\n🚀 [info]Launching ASL Playground (Inference Engine)...[/info]")
    console.print("   [dim](Close the popup window to return to menu)[/dim]")
    
    try:
        import os
        script_path = "inference.py"
        if os.path.exists("src/inference.py"):
            script_path = "src/inference.py"

        subprocess.run([sys.executable, script_path, "--model", "asl_model.pth"])
    except Exception as e:
        console.print(f"❌ [error]Error launching playground: {e}[/error]")


def show_banner():
    banner = pyfiglet.figlet_format("SignCLI", font="slant")
    console.print(f"[title]{banner}[/title]")
    console.print("[dim]AI-Powered ASL Learning Terminal[/dim]\n")


def main():
    console.clear()
    show_banner()
    # Silence user_manager login prints if possible (or refactor login)
    current_user = user_manager.login()
    if not current_user: return

    # Load AI once for Game Mode
    ai_brain = ASLInferenceBridge(TRAINED_MODEL_PATH)

    while True:
        console.clear()
        show_banner()
        console.print(f"[bold magenta]MAIN MENU[/bold magenta] ([cyan]{current_user.username}[/cyan])")
        
        menu_table = Table(show_header=False, box=None)
        menu_table.add_row("[bold green]1.[/bold green]", "Practice Next Level (Auto)")
        menu_table.add_row("[bold green]2.[/bold green]", "Select Level to Practice")
        menu_table.add_row("[bold green]3.[/bold green]", "Check Level Stats")
        menu_table.add_row("[bold green]4.[/bold green]", "Global Stats")
        menu_table.add_row("[bold green]5.[/bold green]", "Skip Current Level (Cheat)")
        menu_table.add_row("[bold red]6.[/bold red]", "Delete Account")
        menu_table.add_row("[bold cyan]7.[/bold cyan]", "ASL Playground (Free Mode)")
        menu_table.add_row("[bold white]8.[/bold white]", "Quit")
        
        console.print(menu_table)
        
        choice = Prompt.ask("Select an option", choices=[str(i) for i in range(1, 9)], default="1")
        
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
            console.print("[info]Goodbye![/info]")
            break
        else:
            console.print("[error]Invalid option.[/error]")

if __name__ == "__main__":
    main()
