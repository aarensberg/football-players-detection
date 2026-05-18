#!/usr/bin/env python3
"""Debug script to trace the first goalkeeper (id=99 -> id=116) fragmentation."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import parse_args
from src.detection_tracking import DetectionTracker
from src.team_assignment import TeamAssigner
from src.video_io import VideoReader
import sys

# Parse arguments with defaults
sys.argv = [
    "trace_first_keeper_debug.py",
    "08fd33_4.mp4",
    "--output-dir",
    "output",
    "--max-frames",
    "0",
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

print("Tracking first goalkeeper fragmentation (id=99 -> id=116)...")
print("=" * 80)

target_ids = {99, 116}
keeper_tracks = {}

for frame_idx, frame in reader.frames():
    if frame_idx > 550:  # Only look at early frames
        break
    if frame_idx % 50 == 0:
        print(f"Frame {frame_idx}...", flush=True)

    detections = detector.infer_and_track(frame, frame_idx)

    for det in detections:
        if det.object_type != "player" and det.object_type != "goalkeeper":
            continue

        track_id = det.track_id

        if track_id not in keeper_tracks:
            keeper_tracks[track_id] = []

        keeper_tracks[track_id].append(
            {
                "frame_idx": frame_idx,
                "object_type": det.object_type,
                "bbox": det.bbox,
                "score": det.score,
            }
        )

reader.release()

# Print results
print("\n" + "=" * 80)
print("TRACE FOR TARGET TRACKS (id=99 and id=116):")
print("=" * 80)

for track_id in sorted(target_ids):
    if track_id not in keeper_tracks:
        print(f"\nTRACK {track_id}: Not found in output")
        continue

    frames = keeper_tracks[track_id]
    print(
        f"\nTRACK {track_id}: frames {frames[0]['frame_idx']}-{frames[-1]['frame_idx']} ({len(frames)} detections)"
    )
    for i, f in enumerate(frames[:20]):
        print(f"  Frame {f['frame_idx']}: {f['object_type']:12} score={f['score']:.3f}")

# Check for spatial proximity between id=99 and id=116 detections
print("\n" + "=" * 80)
print("SPATIAL PROXIMITY ANALYSIS (frames 0-120):")
print("=" * 80)

if 99 in keeper_tracks and 116 in keeper_tracks:
    frames_99 = {f["frame_idx"]: f for f in keeper_tracks[99]}
    frames_116 = {f["frame_idx"]: f for f in keeper_tracks[116]}

    common_frames = sorted(set(frames_99.keys()) & set(frames_116.keys()))
    print(f"Frames where both id=99 and id=116 appear: {common_frames}")

    for frame_idx in common_frames[:5]:
        f99 = frames_99[frame_idx]
        f116 = frames_116[frame_idx]

        x99_c = (f99["bbox"][0] + f99["bbox"][2]) / 2
        y99_c = (f99["bbox"][1] + f99["bbox"][3]) / 2
        x116_c = (f116["bbox"][0] + f116["bbox"][2]) / 2
        y116_c = (f116["bbox"][1] + f116["bbox"][3]) / 2

        dist = ((x99_c - x116_c) ** 2 + (y99_c - y116_c) ** 2) ** 0.5
        print(
            f"Frame {frame_idx}: distance={dist:.1f}px, id=99 score={f99['score']:.3f}, id=116 score={f116['score']:.3f}"
        )
