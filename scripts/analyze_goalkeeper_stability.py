#!/usr/bin/env python3
"""
Analyze goalkeeper stability from pipeline output.
Logs goalkeeper detections frame-by-frame to detect oscillations.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

# Set up sys.argv for parse_args
sys.argv = [
    "analyze_goalkeeper_stability.py",
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


def analyze_goalkeeper_stability():
    """Analyze goalkeeper detections frame-by-frame."""

    args = parse_args()
    video_path = args.video_path
    max_frames = args.max_frames or 300

    cap = cv2.VideoCapture(str(video_path))
    detector = DetectionTracker(args)
    team_assigner = TeamAssigner()

    goalkeeper_history = defaultdict(list)  # track_id -> [(frame, team_id, confidence)]

    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Infer and track
        detections = detector.infer_and_track(frame, frame_idx)

        # Assign teams
        team_assigner.assign(frame, detections)

        # Capture goalkeeper data
        for det in detections:
            if det.object_type == "goalkeeper":
                track_id = det.track_id
                team_id = det.team_id
                score = det.score  # Use 'score' instead of 'confidence'
                goalkeeper_history[track_id].append((frame_idx, team_id, score))

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}...", file=sys.stderr)

    cap.release()

    # Analyze stability
    print("\n" + "=" * 60)
    print("GOALKEEPER STABILITY ANALYSIS")
    print("=" * 60)

    for track_id in sorted(goalkeeper_history.keys()):
        history = goalkeeper_history[track_id]
        frame_range = (history[0][0], history[-1][0])
        team_ids = [t[1] for t in history]
        scores = [t[2] for t in history]

        # Count transitions
        team_transitions = sum(
            1
            for i in range(1, len(team_ids))
            if team_ids[i] != team_ids[i - 1]
            and team_ids[i] is not None
            and team_ids[i - 1] is not None
        )

        none_count = sum(1 for t in team_ids if t is None)

        status = (
            "✅ STABLE"
            if team_transitions == 0
            else f"❌ OSCILLATES ({team_transitions} switches)"
        )

        print(
            f"\nTrack {track_id:3d}: frames {frame_range[0]:3d}-{frame_range[1]:3d} ({len(history):3d} detections)"
        )
        print(f"  Status: {status}")
        print(f"  Teams:  {team_ids[:30]}{'...' if len(team_ids) > 30 else ''}")
        if none_count > 0:
            print(f"  None assignments: {none_count}/{len(team_ids)}")

        # Show confidence stats
        valid_scores = [c for c in scores if c is not None]
        if valid_scores:
            print(
                f"  Scores: min={min(valid_scores):.2f}, max={max(valid_scores):.2f}, mean={np.mean(valid_scores):.2f}"
            )


if __name__ == "__main__":
    analyze_goalkeeper_stability()
