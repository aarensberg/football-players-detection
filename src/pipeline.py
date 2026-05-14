from __future__ import annotations

from pathlib import Path

from src.config import PipelineConfig
from src.detection_tracking import DetectionTracker
from src.video_io import VideoReader, VideoWriter
from src.visualization import draw_detections


def run_pipeline(config: PipelineConfig) -> Path:
    if not config.video_path.exists() or not config.video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {config.video_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"{config.video_path.stem}_annotated.avi"

    reader = VideoReader(config.video_path)
    writer = VideoWriter(output_path, reader.meta)
    detector = DetectionTracker(config)

    processed_count = 0
    try:
        for frame_idx, frame in reader.frames():
            if frame_idx < config.start_frame:
                continue
            if config.end_frame is not None and frame_idx > config.end_frame:
                break
            if frame_idx % config.stride != 0:
                continue

            detections = detector.infer_and_track(frame, frame_idx)
            annotated = draw_detections(frame, detections)
            writer.write(annotated)
            processed_count += 1

            if processed_count % 25 == 0:
                print(f"Processed {processed_count} frames (source frame {frame_idx})")

            if config.max_frames is not None and processed_count >= config.max_frames:
                break
    finally:
        reader.release()
        writer.release()

    print(f"Tracking keys: {list(detector.tracks.keys())}")
    for bucket, frames in detector.tracks.items():
        tracked_objects = sum(len(frame_tracks) for frame_tracks in frames.values())
        print(
            f"Track summary - {bucket}: frames_with_tracks={len(frames)}, "
            f"tracked_objects={tracked_objects}"
        )
    return output_path
