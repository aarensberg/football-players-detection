"""
Run multiple pipeline variants and collect metrics/videos for experiments.

Usage example:
  python scripts/run_experiments.py --video 08fd33_4.mp4 --variants baseline finetuned_full
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict

from src import config as cfgmod
from src.pipeline import run_pipeline
from src.video_io import VideoReader
from src.metrics import MetricsCollector

VARIANTS: Dict[str, Dict] = {
    "baseline": {
        "detector_weights_mode": "generic",
        "detector_weights_path": Path("yolov8n.pt"),
        "enable_camera_motion": False,
        "enable_ball_interpolation": False,
        "enable_team_assignment": False,
    },
    "finetuned_full": {
        "detector_weights_mode": "football_finetuned",
        "detector_weights_path": Path("models/colab_v3_51e_1920_b2-best.pt"),
        "enable_camera_motion": True,
        "enable_ball_interpolation": True,
        "enable_team_assignment": True,
    },
    "finetuned_no_interp": {
        "detector_weights_mode": "football_finetuned",
        "detector_weights_path": Path("models/colab_v3_51e_1920_b2-best.pt"),
        "enable_camera_motion": True,
        "enable_ball_interpolation": False,
        "enable_team_assignment": True,
    },
    "finetuned_no_cam": {
        "detector_weights_mode": "football_finetuned",
        "detector_weights_path": Path("models/colab_v3_51e_1920_b2-best.pt"),
        "enable_camera_motion": False,
        "enable_ball_interpolation": True,
        "enable_team_assignment": True,
    },
    "finetuned_no_team": {
        "detector_weights_mode": "football_finetuned",
        "detector_weights_path": Path("models/colab_v3_51e_1920_b2-best.pt"),
        "enable_camera_motion": True,
        "enable_ball_interpolation": True,
        "enable_team_assignment": False,
    },
}


def build_base_config(
    video: str,
    max_frames: int | None,
    imgsz: int | None,
    meters_per_pixel: float | None,
) -> cfgmod.PipelineConfig:
    # call existing parse_args by temporarily overriding sys.argv to build a base config
    saved_argv = sys.argv
    try:
        argv = ["run_experiments", video]
        if max_frames is not None:
            argv += ["--max-frames", str(max_frames)]
        if imgsz is not None:
            argv += ["--imgsz", str(imgsz)]
        if meters_per_pixel is not None:
            argv += ["--meters-per-pixel", str(meters_per_pixel)]
        sys.argv = argv
        base = cfgmod.parse_args()
        return base
    finally:
        sys.argv = saved_argv


def run_variant(variant_name: str, base_cfg: cfgmod.PipelineConfig) -> None:
    spec = VARIANTS[variant_name]
    out_dir = Path("output") / "experiments" / variant_name
    overrides = {
        "output_dir": out_dir,
        "detector_weights_mode": spec.get(
            "detector_weights_mode", base_cfg.detector_weights_mode
        ),
        "detector_weights_path": Path(
            spec.get("detector_weights_path", base_cfg.detector_weights_path)
        ),
        "enable_camera_motion": spec.get(
            "enable_camera_motion", base_cfg.enable_camera_motion
        ),
        "enable_ball_interpolation": spec.get(
            "enable_ball_interpolation", base_cfg.enable_ball_interpolation
        ),
        "enable_team_assignment": spec.get(
            "enable_team_assignment", base_cfg.enable_team_assignment
        ),
    }
    cfg = replace(base_cfg, **overrides)

    # get fps from reader
    reader = VideoReader(cfg.video_path)
    fps = reader.meta.fps
    reader.release()

    print(f"Running variant {variant_name} -> output: {cfg.output_dir}")
    out = run_pipeline(cfg, return_detector=True)
    if isinstance(out, tuple):
        video_path, detector = out
    else:
        video_path = out
        detector = None

    # collect metrics
    mc = MetricsCollector(fps=fps, meters_per_pixel=cfg.meters_per_pixel)
    if detector is not None:
        summary = mc.finalize(detector, cfg.output_dir, variant_name)
        print(f"Wrote video: {video_path}")
        print(f"Wrote metrics to: {cfg.output_dir}")
    else:
        print(f"No detector returned for variant {variant_name}; skipping metrics.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiments for football pipeline"
    )
    parser.add_argument("--video", type=str, default="08fd33_4.mp4")
    parser.add_argument(
        "--variants", type=str, nargs="+", default=list(VARIANTS.keys())
    )
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--meters-per-pixel", type=float, default=None)
    args = parser.parse_args()

    base_cfg = build_base_config(
        args.video, args.max_frames, args.imgsz, args.meters_per_pixel
    )

    for v in args.variants:
        if v not in VARIANTS:
            print(f"Unknown variant: {v}")
            continue
        run_variant(v, base_cfg)


if __name__ == "__main__":
    main()
