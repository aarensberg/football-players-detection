#!/usr/bin/env python3
"""
Analyze why first goalkeeper is not detected as goalkeeper.
Track all player/goalkeeper detections chronologically.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

sys.argv = [
    "analyze_first_goalkeeper.py",
    "08fd33_4.mp4",
    "--max-frames",
    "650",
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

all_objects = (
    []
)  # List of all detections: (frame, track_id, model_class, object_type, team_id)

frame_idx = 0
while cap.isOpened() and frame_idx < args.max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.infer_and_track(frame, frame_idx)
    team_assigner.assign(frame, detections)

    for det in detections:
        if det.object_type in {"goalkeeper", "player"} and det.track_id >= 0:
            all_objects.append(
                (frame_idx, det.track_id, det.model_class, det.object_type, det.team_id)
            )

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"  Frame {frame_idx}...", file=sys.stderr)

cap.release()

# Find first goalkeeper detections
first_gk_detection = next((obj for obj in all_objects if obj[3] == "goalkeeper"), None)
first_player_detection = next((obj for obj in all_objects if obj[3] == "player"), None)

print("\n" + "=" * 80)
print("GOALKEEPER DETECTION ANALYSIS")
print("=" * 80)

if first_player_detection:
    print(
        f"\nFirst PLAYER detection: frame {first_player_detection[0]}, track {first_player_detection[1]}"
    )

if first_gk_detection:
    print(
        f"First GOALKEEPER detection: frame {first_gk_detection[0]}, track {first_gk_detection[1]}"
    )

# List tracks by first appearance
tracks_by_first_frame = defaultdict(
    lambda: {"first_frame": None, "first_type": None, "states": []}
)
for frame, track_id, model_class, object_type, team_id in all_objects:
    if (
        track_id not in tracks_by_first_frame
        or frame < tracks_by_first_frame[track_id]["first_frame"]
    ):
        tracks_by_first_frame[track_id]["first_frame"] = frame
        tracks_by_first_frame[track_id]["first_type"] = object_type
    tracks_by_first_frame[track_id]["states"].append((frame, object_type, team_id))

# Show tracks that become goalkeepers
print("\n" + "=" * 80)
print("TRACKS THAT APPEAR AS GOALKEEPERS")
print("=" * 80)

goalkeeper_tracks = [
    t
    for t in tracks_by_first_frame
    if any(s[1] == "goalkeeper" for s in tracks_by_first_frame[t]["states"])
]
goalkeeper_tracks = sorted(
    goalkeeper_tracks, key=lambda t: tracks_by_first_frame[t]["first_frame"]
)

print(f"\nTotal tracks that appear as goalkeeper: {len(goalkeeper_tracks)}\n")

for track_id in goalkeeper_tracks[:10]:  # Show first 10
    info = tracks_by_first_frame[track_id]
    first_frame = info["first_frame"]
    first_type = info["first_type"]

    # Find when it becomes goalkeeper
    states = info["states"]
    gk_frame = next(s[0] for s in states if s[1] == "goalkeeper")

    print(f"Track {track_id:3d}:")
    print(f"  First appearance: frame {first_frame}, as {first_type}")
    print(
        f"  Becomes goalkeeper: frame {gk_frame} ({gk_frame - first_frame} frames later)"
    )
    print(f"  States: {[(s[0], s[1], s[2]) for s in states[:8]]}")
    if len(states) > 8:
        print(f"  ... ({len(states) - 8} more states)")
    print()

# Check for tracks that NEVER become goalkeepers but appear early
print("\n" + "=" * 80)
print("EARLY PLAYER TRACKS THAT NEVER BECOME GOALKEEPERS")
print("=" * 80)
print("(These might be the 'first goalkeeper' that should be detected)")
print()

early_player_tracks = [
    t
    for t in sorted(
        tracks_by_first_frame.keys(),
        key=lambda x: tracks_by_first_frame[x]["first_frame"],
    )
    if tracks_by_first_frame[t]["first_frame"] < 300
    and all(s[1] == "player" for s in tracks_by_first_frame[t]["states"])
    and any(s[2] is not None for s in tracks_by_first_frame[t]["states"])  # Has a team
]

for track_id in early_player_tracks[:5]:
    info = tracks_by_first_frame[track_id]
    first_frame = info["first_frame"]
    states = info["states"]
    teams = set(s[2] for s in states if s[2] is not None)

    print(
        f"Track {track_id:3d}: frames {first_frame}-{states[-1][0]}, teams={teams}, count={len(states)}"
    )
    print(f"  First states: {states[:5]}")
    print()
