#!/usr/bin/env python3
"""Quick test: check if goalkeepers are detected and how they're classified."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up sys.argv for parse_args
sys.argv = [
    "test_goalkeepers.py",
    "08fd33_4.mp4",
    "--max-frames",
    "500",
    "--imgsz",
    "1920",
    "--device",
    "mps",
    "--detector-weights-mode",
    "football_finetuned",
    "--detector-weights-path",
    "models/colab_v3_51e_1920_b2-best.pt",
]

from src.config import parse_args
from src.detection_tracking import DetectionTracker
import cv2

args = parse_args()
video_path = args.video_path
max_frames = args.max_frames or 500

cap = cv2.VideoCapture(str(video_path))
detector = DetectionTracker(args)

frame_idx = 0
goalkeeper_count = 0
player_count = 0
referee_count = 0

while cap.isOpened() and frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.infer_and_track(frame, frame_idx)

    for det in detections:
        if det.object_type == "goalkeeper":
            goalkeeper_count += 1
            if goalkeeper_count == 1:  # Report first goalkeeper found
                print(f"First goalkeeper found at frame {frame_idx}:")
                print(
                    f"  track_id={det.track_id}, model_class={det.model_class}, object_type={det.object_type}"
                )
        elif det.object_type == "player":
            player_count += 1
        elif det.object_type == "referee":
            referee_count += 1

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(
            f"Processed {frame_idx} frames... (goalkeepers: {goalkeeper_count}, players: {player_count}, referees: {referee_count})"
        )

cap.release()

print(f"\nTotal counts in {frame_idx} frames:")
print(f"  Goalkeepers: {goalkeeper_count}")
print(f"  Players: {player_count}")
print(f"  Referees: {referee_count}")
