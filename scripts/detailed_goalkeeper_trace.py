#!/usr/bin/env python3
"""
Detailed frame-by-frame trace of goalkeeper detection and team assignment.
This helps identify where the detection/assignment goes wrong.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up sys.argv for parse_args
sys.argv = [
    "detailed_goalkeeper_trace.py",
    "08fd33_4.mp4",
    "--max-frames",
    "750",
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
video_path = args.video_path
max_frames = args.max_frames or 750

cap = cv2.VideoCapture(str(video_path))
detector = DetectionTracker(args)
team_assigner = TeamAssigner()

goalkeeper_timeline = defaultdict(
    list
)  # track_id -> [(frame, model_class, object_type, team_id)]
frame_idx = 0

print("Tracing goalkeeper detections and assignments...")
print("=" * 80)

while cap.isOpened() and frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.infer_and_track(frame, frame_idx)
    team_assigner.assign(frame, detections)

    for det in detections:
        # Track all goalkeepers AND all players/objects to see transitions
        if det.object_type in {"goalkeeper", "player"}:
            # Check if this is a goalkeeper-like detection
            track_id = det.track_id
            model_class = det.model_class
            object_type = det.object_type
            team_id = det.team_id

            goalkeeper_timeline[track_id].append(
                (frame_idx, model_class, object_type, team_id)
            )

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"  Frame {frame_idx}...", file=sys.stderr)

cap.release()

print("\n" + "=" * 80)
print("DETAILED GOALKEEPER TIMELINE")
print("=" * 80)

# Focus on goalkeepers (those detected as goalkeeper at some point)
goalkeeper_track_ids = set()
for track_id, timeline in goalkeeper_timeline.items():
    if any(t[2] == "goalkeeper" for t in timeline):
        goalkeeper_track_ids.add(track_id)

print(f"\nFound {len(goalkeeper_track_ids)} goalkeeper tracks\n")

for track_id in sorted(goalkeeper_track_ids):
    timeline = goalkeeper_timeline[track_id]
    frame_range = (timeline[0][0], timeline[-1][0])

    print(f"\n{'='*80}")
    print(
        f"TRACK {track_id}: frames {frame_range[0]}-{frame_range[1]} ({len(timeline)} detections)"
    )
    print(f"{'='*80}")

    # Group by state to show transitions
    transitions = []
    current_state = (timeline[0][1], timeline[0][2], timeline[0][3])
    transition_start = 0  # index, not frame

    for i in range(1, len(timeline)):
        frame, model_class, object_type, team_id = timeline[i]
        state = (model_class, object_type, team_id)
        if state != current_state:
            # Completed a transition, save it
            trans_frame_start, _, _, _ = timeline[transition_start]
            trans_frame_end, _, _, _ = timeline[i - 1]
            transitions.append(
                {
                    "start": trans_frame_start,
                    "end": trans_frame_end,
                    "model_class": current_state[0],
                    "object_type": current_state[1],
                    "team_id": current_state[2],
                    "frames": i - transition_start,
                }
            )
            current_state = state
            transition_start = i

    # Add final transition
    trans_frame_start, _, _, _ = timeline[transition_start]
    trans_frame_end, _, _, _ = timeline[-1]
    transitions.append(
        {
            "start": trans_frame_start,
            "end": trans_frame_end,
            "model_class": current_state[0],
            "object_type": current_state[1],
            "team_id": current_state[2],
            "frames": len(timeline) - transition_start,
        }
    )

    print(f"\nTransitions ({len(transitions)} total):")
    print(
        f"{'Frame':>6} {'Range':>20} {'Model Class':>15} {'Object Type':>12} {'Team':>6} {'Duration':>8}"
    )
    print("-" * 80)

    for i, trans in enumerate(transitions):
        frame_str = f"{trans['start']}-{trans['end']}"
        dur = trans["end"] - trans["start"] + 1
        print(
            f"  {i+1:2d}   {frame_str:>20} {trans['model_class']:>15} {trans['object_type']:>12} {str(trans['team_id']):>6} {dur:>8}"
        )

    # Show detailed timeline for first goalkeeper
    if len(goalkeeper_track_ids) > 0 and track_id == min(goalkeeper_track_ids):
        print(f"\nFrame-by-frame detail (first 50 frames of this track):")
        print(f"{'Frame':>6} {'Model':>12} {'Object Type':>12} {'Team':>6}")
        print("-" * 40)
        for frame, model_class, object_type, team_id in timeline[:50]:
            print(f"{frame:6d} {model_class:>12} {object_type:>12} {str(team_id):>6}")
        if len(timeline) > 50:
            print(f"... ({len(timeline) - 50} more frames)")
