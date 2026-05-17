from __future__ import annotations

import argparse
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
    iou_threshold: float
    ball_conf_threshold: Optional[float]
    ball_min_bbox_area_px: Optional[float]
    ball_max_bbox_area_px: Optional[float]
    ball_min_bbox_area_ratio: Optional[float]
    ball_max_bbox_area_ratio: Optional[float]
    min_bbox_area_px: Optional[float]
    goalkeeper_switch_frames: int
    ball_interpolation_max_gap: int
    imgsz: int
    device: str
    detector_weights_mode: str
    detector_weights_path: Path
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
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.12,
        help="Ball-only confidence threshold (more permissive than --conf by default).",
    )
    parser.add_argument(
        "--ball-min-area-px",
        type=float,
        default=6.0,
        help="Minimum ball bbox area in pixels (post-filter).",
    )
    parser.add_argument(
        "--ball-max-area-px",
        type=float,
        default=None,
        help="Optional maximum ball bbox area in pixels (post-filter).",
    )
    parser.add_argument(
        "--ball-min-area-ratio",
        type=float,
        default=None,
        help="Optional minimum ball bbox area ratio vs frame area (0..1).",
    )
    parser.add_argument(
        "--ball-max-area-ratio",
        type=float,
        default=0.0035,
        help="Maximum ball bbox area ratio vs frame area (0..1).",
    )
    parser.add_argument(
        "--min-area-px",
        type=float,
        default=36.0,
        help="General minimum bbox area in pixels for non-ball objects.",
    )
    parser.add_argument(
        "--goalkeeper-switch-frames",
        type=int,
        default=3,
        help="Consecutive frames required before player/goalkeeper class switch is accepted.",
    )
    parser.add_argument(
        "--ball-interp-max-gap",
        type=int,
        default=4,
        help="Max missing-frame gap for linear ball track interpolation.",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (higher improves small-object recall, slower runtime)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Inference device (e.g. "cpu", "mps", "0")',
    )
    parser.add_argument(
        "--detector-weights-mode",
        type=str,
        choices=["generic", "football_finetuned"],
        default="generic",
        help=(
            "Detector weights mode. "
            "'generic' uses pretrained yolov8n.pt. "
            "'football_finetuned' uses --detector-weights-path if available."
        ),
    )
    parser.add_argument(
        "--detector-weights-path",
        type=str,
        default="models/football_yolov8_best.pt",
        help="Path to football-finetuned detector weights",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config",
    )

    args = parser.parse_args()

    max_frames = None if args.max_frames is not None and args.max_frames <= 0 else args.max_frames
    detector_weights_mode = args.detector_weights_mode
    detector_weights_path = Path(args.detector_weights_path)

    # Backward-compatibility alias from Milestone 1.
    if args.model_path:
        detector_weights_mode = "football_finetuned"
        detector_weights_path = Path(args.model_path)
        print(
            "[INFO] --model-path is deprecated; use "
            "--detector-weights-mode football_finetuned --detector-weights-path <path>."
        )

    return PipelineConfig(
        video_path=Path(args.video_path),
        output_dir=Path(args.output_dir),
        max_frames=max_frames,
        start_frame=max(0, args.start_frame),
        end_frame=args.end_frame if args.end_frame is None else max(0, args.end_frame),
        stride=max(1, args.stride),
        conf_threshold=max(0.0, min(1.0, args.conf)),
        iou_threshold=max(0.0, min(1.0, args.iou)),
        ball_conf_threshold=(
            None if args.ball_conf is None else max(0.0, min(1.0, args.ball_conf))
        ),
        ball_min_bbox_area_px=(
            None if args.ball_min_area_px is None else max(0.0, float(args.ball_min_area_px))
        ),
        ball_max_bbox_area_px=(
            None if args.ball_max_area_px is None else max(0.0, float(args.ball_max_area_px))
        ),
        ball_min_bbox_area_ratio=(
            None
            if args.ball_min_area_ratio is None
            else max(0.0, min(1.0, float(args.ball_min_area_ratio)))
        ),
        ball_max_bbox_area_ratio=(
            None
            if args.ball_max_area_ratio is None
            else max(0.0, min(1.0, float(args.ball_max_area_ratio)))
        ),
        min_bbox_area_px=(
            None if args.min_area_px is None else max(0.0, float(args.min_area_px))
        ),
        goalkeeper_switch_frames=max(1, int(args.goalkeeper_switch_frames)),
        ball_interpolation_max_gap=max(0, int(args.ball_interp_max_gap)),
        imgsz=max(64, int(args.imgsz)),
        device=args.device,
        detector_weights_mode=detector_weights_mode,
        detector_weights_path=detector_weights_path,
        tracker_config=args.tracker_config,
    )
