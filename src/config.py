from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    video_path: Path
    output_dir: Path
    max_frames: Optional[int]
    start_frame: int
    end_frame: Optional[int]
    stride: int
    conf_threshold: float
    conf_explicitly_set: bool
    iou_threshold: float
    device: str
    model_path: Optional[Path]
    tracker_config: str


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Milestone 1: football detection + tracking + annotated video output."
    )
    parser.add_argument("video_path", type=str, help="Input video path")
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Directory for output video"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of processed frames (<=0 means all)",
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="First frame index to process"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame index to process (inclusive)",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Process every Nth frame (N >= 1)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Inference device (e.g. "cpu", "mps", "0")',
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model weights path. If omitted, uses models/best.pt if available, else yolov8n.pt.",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config",
    )

    args = parser.parse_args()

    conf_explicitly_set = any(arg == "--conf" or arg.startswith("--conf=") for arg in sys.argv[1:])
    max_frames = None if args.max_frames is not None and args.max_frames <= 0 else args.max_frames
    model_path = Path(args.model_path) if args.model_path else None

    return PipelineConfig(
        video_path=Path(args.video_path),
        output_dir=Path(args.output_dir),
        max_frames=max_frames,
        start_frame=max(0, args.start_frame),
        end_frame=args.end_frame if args.end_frame is None else max(0, args.end_frame),
        stride=max(1, args.stride),
        conf_threshold=max(0.0, min(1.0, args.conf)),
        conf_explicitly_set=conf_explicitly_set,
        iou_threshold=max(0.0, min(1.0, args.iou)),
        device=args.device,
        model_path=model_path,
        tracker_config=args.tracker_config,
    )
