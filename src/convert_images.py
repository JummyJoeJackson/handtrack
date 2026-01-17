import cv2
import csv
import os
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

HAND_LANDMARKER_PATH = "hand_landmarker.task"
OUTPUT_CSV = "asl_data.csv"


def init_csv(output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["label"]
        for i in range(21):
            headers.extend([f"x{i}", f"y{i}", f"z{i}"])
        writer.writerow(headers)


def extract_and_normalize(landmarker, image_path):
    cv_img = cv2.imread(image_path)
    if cv_img is None:
        return None

    # Add padding and enhance contrast (same as asl.py)
    cv_img = cv2.copyMakeBorder(cv_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv_img = cv2.convertScaleAbs(cv_img, alpha=1.3, beta=10)
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    results = landmarker.detect(mp_image)

    if not results.hand_landmarks:
        return None

    hand_landmarks = results.hand_landmarks[0]

    # Normalize same way as main.py: center on wrist + scale
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]

    scale_factor = ((wrist.x - middle_mcp.x)**2 +
                    (wrist.y - middle_mcp.y)**2 +
                    (wrist.z - middle_mcp.z)**2)**0.5

    if scale_factor == 0:
        scale_factor = 1

    normalized = []
    for lm in hand_landmarks:
        normalized.extend([
            (lm.x - wrist.x) / scale_factor,
            (lm.y - wrist.y) / scale_factor,
            (lm.z - wrist.z) / scale_factor
        ])

    return normalized


def convert_dataset(data_dir, output_csv, max_per_class=None):
    data_path = Path(data_dir)

    # Find the actual train folder (handles nested structure)
    if (data_path / "asl_alphabet_train" / "asl_alphabet_train").exists():
        train_path = data_path / "asl_alphabet_train" / "asl_alphabet_train"
    elif (data_path / "asl_alphabet_train").exists():
        train_path = data_path / "asl_alphabet_train"
    else:
        train_path = data_path

    print(f"Looking for classes in: {train_path}")

    class_folders = sorted([f for f in train_path.iterdir() if f.is_dir()])
    print(f"Found {len(class_folders)} classes: {[f.name for f in class_folders]}")

    init_csv(output_csv)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3
    )

    total_processed = 0
    total_success = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for class_folder in class_folders:
            label = class_folder.name
            images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))

            if max_per_class:
                images = images[:max_per_class]

            class_success = 0

            for i, img_path in enumerate(images):
                landmarks = extract_and_normalize(landmarker, str(img_path))
                total_processed += 1

                if landmarks:
                    with open(output_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([label] + landmarks)
                    class_success += 1
                    total_success += 1

                if (i + 1) % 100 == 0:
                    print(f"  {label}: {i + 1}/{len(images)} processed, {class_success} success")

            print(f"{label}: {class_success}/{len(images)} landmarks extracted")

    print(f"\nDone! {total_success}/{total_processed} images converted")
    print(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ASL images to landmarks CSV")
    parser.add_argument("--data-dir", type=str, default="data/asl_alphabet",
                        help="Path to dataset folder")
    parser.add_argument("--output", type=str, default="asl_data.csv",
                        help="Output CSV file")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Max images per class (for testing)")

    args = parser.parse_args()

    convert_dataset(args.data_dir, args.output, args.max_per_class)
