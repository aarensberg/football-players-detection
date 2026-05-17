from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.detection_tracking import DetectionTracker
from src.video_io import VideoReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug global-vs-fused class counts for ball fusion behavior."
    )
    parser.add_argument("video_path", type=str, help="Input video path")
    parser.add_argument("--max-frames", type=int, default=80, help="Number of frames to inspect")
    parser.add_argument("--imgsz", type=int, default=1920, help="Inference image size")
    parser.add_argument("--device", type=str, default="mps", help="Inference device")
    parser.add_argument(
        "--detector-weights-mode",
        type=str,
        choices=["generic", "football_finetuned"],
        default="football_finetuned",
    )
    parser.add_argument(
        "--detector-weights-path",
        type=str,
        default="models/colab_v3_51e_1920_b2-best.pt",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        video_path=Path(args.video_path),
        output_dir=Path("output"),
        max_frames=args.max_frames,
        start_frame=0,
        end_frame=None,
        stride=1,
        conf_threshold=0.25,
        iou_threshold=0.5,
        ball_iou_threshold=0.35,
        agnostic_nms=False,
        ball_conf_threshold=0.12,
        ball_min_bbox_area_px=6.0,
        ball_max_bbox_area_px=None,
        ball_min_bbox_area_ratio=None,
        ball_max_bbox_area_ratio=0.0035,
        min_bbox_area_px=36.0,
        player_max_bbox_area_ratio=None,
        goalkeeper_switch_frames=3,
        ball_interpolation_max_gap=6,
        ball_interpolation_max_center_speed_px_per_frame=120.0,
        ball_interpolation_max_endpoint_area_change_ratio=3.5,
        ball_roi_recovery_enabled=True,
        ball_roi_conf_threshold=0.05,
        ball_roi_window_scale=2.5,
        ball_roi_max_missed_frames=4,
        imgsz=args.imgsz,
        device=args.device,
        detector_weights_mode=args.detector_weights_mode,
        detector_weights_path=Path(args.detector_weights_path),
        tracker_config="bytetrack.yaml",
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    tracker = DetectionTracker(cfg)
    reader = VideoReader(cfg.video_path)

    global_totals = {"player": 0, "goalkeeper": 0, "referee": 0, "ball": 0}
    fused_totals = {"player": 0, "goalkeeper": 0, "referee": 0, "ball": 0}
    ball_from_global = 0
    ball_from_ball_only = 0
    processed = 0
    try:
        for frame_idx, frame in reader.frames():
            tracker.infer_and_track(frame, frame_idx)
            debug = tracker.last_frame_debug
            g = debug.get("global_counts", {})
            f = debug.get("fused_counts", {})
            for key in global_totals:
                global_totals[key] += int(g.get(key, 0))
                fused_totals[key] += int(f.get(key, 0))
            if debug.get("ball_source") == "global":
                ball_from_global += 1
            elif debug.get("ball_source") == "ball_only":
                ball_from_ball_only += 1

            print(
                f"frame={frame_idx} "
                f"global={g} fused={f} "
                f"ball_global_candidates={debug.get('ball_candidates_global', 0)} "
                f"ball_only_candidates={debug.get('ball_candidates_ball_only', 0)} "
                f"ball_source={debug.get('ball_source', 'none')}"
            )
            processed += 1
            if processed >= cfg.max_frames:
                break
    finally:
        reader.release()

    print("\n=== Totals ===")
    print(f"Processed frames: {processed}")
    print(f"Global totals: {global_totals}")
    print(f"Fused totals:  {fused_totals}")
    print(f"Ball chosen from global: {ball_from_global}")
    print(f"Ball chosen from ball-only pass: {ball_from_ball_only}")


if __name__ == "__main__":
    main()
