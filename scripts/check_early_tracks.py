#!/usr/bin/env python3
"""Check if team assignments for early tracks are now stable."""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

sys.argv = [
    "check_early_tracks.py",
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
from src.team_assignment import TeamAssigner
import cv2

args = parse_args()
cap = cv2.VideoCapture(str(args.video_path))
detector = DetectionTracker(args)
team_assigner = TeamAssigner()

# Track team assignments for early-appearing tracks
team_timeline = defaultdict(list)

frame_idx = 0
while cap.isOpened() and frame_idx < 500:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.infer_and_track(frame, frame_idx)
    team_assigner.assign(frame, detections)

    for det in detections:
        if det.track_id in {1, 4}:  # Focus on likely keeper tracks
            team_timeline[det.track_id].append((frame_idx, det.team_id))

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"  Frame {frame_idx}...", file=sys.stderr)

cap.release()

print("\n" + "=" * 80)
print("EARLY TRACK TEAM STABILITY")
print("=" * 80)

for track_id in sorted(team_timeline.keys()):
    timeline = team_timeline[track_id]
    if not timeline:
        continue

    frame_range = (timeline[0][0], timeline[-1][0])
    teams = [t[1] for t in timeline]

    # Count transitions
    transitions = sum(
        1
        for i in range(1, len(teams))
        if teams[i] != teams[i - 1]
        and teams[i] is not None
        and teams[i - 1] is not None
    )

    unique_teams = set(t for t in teams if t is not None)

    print(f"\nTrack {track_id}:")
    print(f"  Frames: {frame_range[0]}-{frame_range[1]} ({len(timeline)} detections)")
    print(f"  Unique teams: {unique_teams}")
    print(f"  Team transitions: {transitions}")
    print(f"  Team sequence: {teams[:30]}{'...' if len(teams) > 30 else ''}")
