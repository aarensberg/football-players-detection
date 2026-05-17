from __future__ import annotations

from pathlib import Path

from src.ball_interpolation import BallInterpolator
from src.config import PipelineConfig
from src.detection_tracking import DetectionTracker
from src.team_assignment import TeamAssigner
from src.video_io import VideoReader, VideoWriter, build_output_video_path
from src.visualization import draw_detections


def run_pipeline(config: PipelineConfig) -> Path:
    if not config.video_path.exists() or not config.video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {config.video_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(config.video_path)
    detector = DetectionTracker(config)
    output_path = build_output_video_path(
        config.output_dir,
        config.video_path,
        detector.model_path_used,
        extension="mp4",
    )
    writer = VideoWriter(output_path, reader.meta)
    team_assigner = TeamAssigner()
    ball_interpolator = BallInterpolator(
        max_gap_frames=config.ball_interpolation_max_gap,
        max_center_speed_px_per_frame=config.ball_interpolation_max_center_speed_px_per_frame,
        max_endpoint_area_change_ratio=config.ball_interpolation_max_endpoint_area_change_ratio,
    )

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
            ball_interpolator.record_frame_tracks(
                frame_idx, detector.tracks.get("ball", {}).get(frame_idx, {})
            )
            team_assigner.assign(frame, detections)
            for det in detections:
                if det.object_type not in {"player", "goalkeeper"}:
                    continue
                if det.track_id < 0 or det.team_id is None:
                    continue
                player_tracks = detector.tracks.get("players", {}).get(frame_idx, {})
                if det.track_id in player_tracks:
                    player_tracks[det.track_id]["team_id"] = det.team_id
                    player_tracks[det.track_id]["team_color"] = det.team_color
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

    interpolation_stats = ball_interpolator.interpolate_tracks(detector.tracks)
    print(
        "Ball interpolation summary - "
        f"observed_frames={interpolation_stats.observed_frames}, "
        f"interpolated_frames={interpolation_stats.interpolated_frames}, "
        f"max_gap={config.ball_interpolation_max_gap}"
    )

    print(f"Tracking keys: {list(detector.tracks.keys())}")
    for bucket, frames in detector.tracks.items():
        tracked_objects = sum(len(frame_tracks) for frame_tracks in frames.values())
        print(
            f"Track summary - {bucket}: frames_with_tracks={len(frames)}, "
            f"tracked_objects={tracked_objects}"
        )

    team_summary = team_assigner.summary()
    t_counts = team_summary["track_team_counts"]
    print(
        "Team assignment summary - "
        f"fitted={team_summary['fitted']}, "
        f"track_counts: team1={t_counts[1]}, team2={t_counts[2]}, "
        f"assignment_events={team_summary['assignment_events']}"
    )
    for warning in team_summary.get("warnings", []):
        print(warning)
    return writer.output_path
