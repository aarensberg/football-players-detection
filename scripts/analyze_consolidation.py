#!/usr/bin/env python3
"""Analyze goalkeeper consolidation after fixes."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import parse_args
from src.detection_tracking import DetectionTracker
from src.team_assignment import TeamAssigner
from src.video_io import VideoReader
import sys

# Parse arguments
sys.argv = [
    "analyze_consolidation.py",
    "08fd33_4.mp4",
    "--output-dir",
    "output",
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

config = parse_args()
reader = VideoReader(config.video_path)
detector = DetectionTracker(config)
team_assigner = TeamAssigner()

print("Analyzing goalkeeper consolidation...")
print("=" * 80)

# Track all player/goalkeeper detections
all_detections = {}

for frame_idx, frame in reader.frames():
    if frame_idx % 100 == 0:
        print(f"Frame {frame_idx}...", flush=True)

    detections = detector.infer_and_track(frame, frame_idx)
    detector._try_reactivate_from_track_memory(detections, frame_idx)

    for det in detections:
        if det.object_type not in {"player", "goalkeeper"}:
            continue

        if det.track_id not in all_detections:
            all_detections[det.track_id] = []

        all_detections[det.track_id].append(
            {
                "frame_idx": frame_idx,
                "object_type": det.object_type,
                "bbox": det.bbox,
                "score": det.score,
            }
        )

reader.release()

# Analyze goalkeeper persistence
print("\n" + "=" * 80)
print("GOALKEEPER TRACKS (>20 frames):")
print("=" * 80)

keeper_tracks = {}
for track_id, detections in all_detections.items():
    # Check if this track is a goalkeeper (either classified as goalkeeper or long-lived player in boundary zones)
    is_keeper = any(d["object_type"] == "goalkeeper" for d in detections)
    if is_keeper and len(detections) > 20:
        keeper_tracks[track_id] = detections
        frame_range = f"{detections[0]['frame_idx']}-{detections[-1]['frame_idx']}"
        print(f"TRACK {track_id}: {frame_range} ({len(detections)} frames)")

        # Show first 3 and last 3 frames
        for d in detections[:3]:
            print(
                f"  Frame {d['frame_idx']}: {d['object_type']:12} score={d['score']:.3f}"
            )
        if len(detections) > 6:
            print("  ...")
            for d in detections[-3:]:
                print(
                    f"  Frame {d['frame_idx']}: {d['object_type']:12} score={d['score']:.3f}"
                )
        print()

# Analyze track continuity (check if tracks don't fragment as much)
print("=" * 80)
print("EARLY GAME TRACKS (frames 0-200):")
print("=" * 80)

early_tracks = {
    tid: dets
    for tid, dets in all_detections.items()
    if any(0 <= d["frame_idx"] <= 200 for d in dets) and len(dets) > 5
}

for track_id in sorted(early_tracks.keys()):
    dets = early_tracks[track_id]
    frame_range = f"{dets[0]['frame_idx']}-{dets[-1]['frame_idx']}"
    is_keeper = any(d["object_type"] == "goalkeeper" for d in dets)
    type_str = "GOALKEEPER" if is_keeper else "player"
    print(f"TRACK {track_id}: {frame_range} ({len(dets):3d} frames) - {type_str}")

print("\nTotal unique goalkeeper tracks (>20 frames): ", len(keeper_tracks))
