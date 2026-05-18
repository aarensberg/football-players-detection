#!/usr/bin/env python3
"""Debug goalkeeper stability: trace each goalkeeper's class and team across frames."""

import os
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import parse_args
from src.pipeline import run_pipeline


def trace_goalkeeper_stability(
    video_path: str, max_frames: int = 200, device: str = "cpu"
):
    """Run pipeline and collect frame-by-frame goalkeeper data."""

    from src.detection_tracking import DetectionTracker
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    # Collect goalkeeper trace
    goalkeeper_trace = {}  # track_id -> list of (frame_idx, model_class, team_id)

    args = parse_args(
        [
            video_path,
            "--max-frames",
            str(max_frames),
            "--imgsz",
            "1920",
            "--device",
            device,
            "--detector-weights-mode",
            "football_finetuned",
            "--detector-weights-path",
            "models/colab_v3_51e_1920_b2-best.pt",
        ]
    )

    # Initialize tracker
    tracker = DetectionTracker(args.config)

    # Run pipeline frame by frame
    from src.ball_ownership import BallPossessionTracker
    from src.camera_motion import CameraMotionEstimator

    ball_possession = BallPossessionTracker(args.config)
    camera_motion = (
        CameraMotionEstimator(args.config) if args.config.enable_camera_motion else None
    )
    prev_frame = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Infer and track
        detections = tracker.infer_and_track(frame, frame_count)

        # Find goalkeepers
        for det in detections:
            if det.model_class == "goalkeeper":
                track_id = det.track_id
                if track_id not in goalkeeper_trace:
                    goalkeeper_trace[track_id] = []

                # Get team assignment (simplified: check if it would be assigned)
                team_id = None  # Would need full team assignment pipeline
                goalkeeper_trace[track_id].append(
                    (frame_count, det.model_class, team_id)
                )

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Frame {frame_count}...")
        prev_frame = frame

    cap.release()

    # Analyze goalkeeper stability
    print("\n\n=== GOALKEEPER STABILITY ANALYSIS ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Goalkeepers tracked: {len(goalkeeper_trace)}")

    for track_id in sorted(goalkeeper_trace.keys()):
        trace = goalkeeper_trace[track_id]
        frames = [t[0] for t in trace]
        classes = [t[1] for t in trace]

        # Count class transitions
        transitions = sum(
            1 for i in range(1, len(classes)) if classes[i] != classes[i - 1]
        )

        if transitions > 0:
            print(
                f"\n❌ Track {track_id}: {len(frames)} frames, {transitions} CLASS TRANSITIONS"
            )
            print(f"  Frame range: {frames[0]}-{frames[-1]}")
            print(f"  Classes: {classes[:20]}{'...' if len(classes) > 20 else ''}")
        else:
            print(
                f"\n✅ Track {track_id}: {len(frames)} frames, STABLE (class={classes[0]})"
            )
            print(f"  Frame range: {frames[0]}-{frames[-1]}")


if __name__ == "__main__":
    video = "08fd33_4.mp4"
    trace_goalkeeper_stability(video, max_frames=200, device="mps")
