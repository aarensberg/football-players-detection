#!/usr/bin/env python3
"""
Export detections and tracks to JSON for analysis.
Helps verify goalkeeper stability and team consistency.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import parse_args, Config
from src.detection_tracking import DetectionTracker
from src.team_assignment import TeamAssigner
from src.ball_ownership import BallPossessionTracker
from src.camera_motion import CameraMotionEstimator
import cv2
import numpy as np


def export_tracks_and_detections(
    video_path: str, max_frames: int = 200, output_file: str = None
):
    """Export frame-by-frame detections with model_class and team assignments."""

    if output_file is None:
        output_file = f"detections_export_{max_frames}frames.json"

    args = parse_args(
        [
            video_path,
            "--max-frames",
            str(max_frames),
            "--imgsz",
            "1920",
            "--device",
            "mps",
            "--detector-weights-mode",
            "football_finetuned",
            "--detector-weights-path",
            "models/colab_v3_51e_1920_b2-best.pt",
        ]
    )

    cap = cv2.VideoCapture(video_path)
    tracker = DetectionTracker(args.config)
    team_assigner = TeamAssigner(args.config)

    # Export structure
    export_data = {"video": video_path, "max_frames": max_frames, "frames": {}}

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections
        detections = tracker.infer_and_track(frame, frame_idx)

        # Assign teams
        for det in detections:
            team_assigner.update(frame, det)

        # Export frame data
        frame_data = {"detections": []}

        for det in detections:
            det_data = {
                "track_id": det.track_id,
                "class": det.model_class,
                "confidence": float(det.confidence),
                "bbox": [int(b) for b in det.bbox],
                "team_id": team_assigner.track_to_team.get(det.track_id),
            }
            frame_data["detections"].append(det_data)

        export_data["frames"][str(frame_idx)] = frame_data

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Exported {frame_idx} frames...")

    cap.release()

    # Write JSON
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nExported to: {output_file}")

    # Analyze goalkeeper stability
    goalkeeper_tracks = defaultdict(list)
    for frame_str, frame_data in export_data["frames"].items():
        frame_idx = int(frame_str)
        for det in frame_data["detections"]:
            if det["class"] == "goalkeeper":
                track_id = det["track_id"]
                goalkeeper_tracks[track_id].append(
                    {
                        "frame": frame_idx,
                        "class": det["class"],
                        "team_id": det["team_id"],
                        "confidence": det["confidence"],
                    }
                )

    print("\n=== GOALKEEPER STABILITY ===")
    print(f"Total goalkeeper tracks: {len(goalkeeper_tracks)}")

    for track_id in sorted(goalkeeper_tracks.keys()):
        track_data = goalkeeper_tracks[track_id]
        team_ids = [t["team_id"] for t in track_data]

        # Count team transitions
        team_transitions = sum(
            1 for i in range(1, len(team_ids)) if team_ids[i] != team_ids[i - 1]
        )

        frames_range = f"{track_data[0]['frame']}-{track_data[-1]['frame']}"

        if team_transitions > 0:
            print(
                f"❌ Track {track_id}: frames {frames_range}, {team_transitions} TEAM SWITCHES"
            )
            print(f"   Teams: {team_ids[:20]}{'...' if len(team_ids) > 20 else ''}")
        else:
            team_val = team_ids[0] if team_ids else None
            print(
                f"✅ Track {track_id}: frames {frames_range}, STABLE (team={team_val})"
            )


if __name__ == "__main__":
    export_tracks_and_detections("08fd33_4.mp4", max_frames=300)
