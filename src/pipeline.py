from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.ball_ownership import BallPossessionTracker
from src.camera_motion import (
    CameraMotionEstimate,
    CameraMotionEstimator,
    build_pitch_band_mask,
)
from src.ball_interpolation import BallInterpolator
from src.config import PipelineConfig
from src.detection_tracking import DetectionTracker
from src.pitch_geometry import build_pitch_transform
from src.team_assignment import TeamAssigner
from src.video_io import VideoReader, VideoWriter, build_output_video_path
from src.visualization import draw_detections


def run_pipeline(config: PipelineConfig, return_detector: bool = False):
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
    team_assigner = TeamAssigner() if config.enable_team_assignment else None
    camera_motion_estimator = CameraMotionEstimator()
    pitch_transform = build_pitch_transform(
        (reader.meta.width, reader.meta.height),
        mode=config.pitch_transform_mode,
        field_length_m=config.field_length_m,
        field_width_m=config.field_width_m,
        meters_per_pixel=config.meters_per_pixel,
    )
    possession_tracker = BallPossessionTracker(
        max_distance_px=config.possession_max_distance_px,
        max_distance_m=config.possession_max_distance_m,
    )
    ball_interpolator = (
        BallInterpolator(
            max_gap_frames=config.ball_interpolation_max_gap,
            max_center_speed_px_per_frame=config.ball_interpolation_max_center_speed_px_per_frame,
            max_endpoint_area_change_ratio=config.ball_interpolation_max_endpoint_area_change_ratio,
        )
        if config.enable_ball_interpolation
        else None
    )

    prev_frame = None
    cumulative_camera_offset = np.array([0.0, 0.0], dtype=np.float32)

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

            # Suppress duplicate detections using IoU-based deduplication
            # Prefer detections from tracks that already have a team assignment.
            # Save removed detections to track memory for potential reactivation on next frames.
            def _dedupe_detections_iou(dets, iou_thresh=0.4):
                kept = []
                kept_bboxes = []
                removed = []  # Track removed detections

                # Prefer detections with team assignment, then by score
                def _sort_key(d):
                    has_team = False
                    try:
                        has_team = (
                            team_assigner is not None
                            and d.track_id in team_assigner.track_to_team
                        )
                    except Exception:
                        has_team = False
                    return (0 if has_team else 1, -getattr(d, "score", 0.0))

                dets_sorted = sorted(dets, key=_sort_key)
                for d in dets_sorted:
                    d_bbox = np.array(d.bbox, dtype=np.float32)
                    overlaps = False
                    for k_bbox in kept_bboxes:
                        iou = detector._bbox_iou(d_bbox, k_bbox)
                        if iou > iou_thresh:
                            overlaps = True
                            break
                    if not overlaps:
                        kept.append(d)
                        kept_bboxes.append(d_bbox)
                    else:
                        removed.append(d)

                # Save removed detections to track memory for reactivation on future frames
                for d in removed:
                    if d.object_type != "ball" and d.track_id >= 0:
                        detector.track_memory[d.track_id] = {
                            "frame_idx": frame_idx,
                            "bbox": d.bbox,
                            "object_type": d.object_type,
                            "class_id": d.class_id,
                        }
                return kept

            detections = _dedupe_detections_iou(detections, iou_thresh=0.4)

            # Try to reactivate recently terminated tracks from memory
            # This happens AFTER deduplication so track memory has recent data
            detector._try_reactivate_from_track_memory(detections, frame_idx)

            # Inherit team assignments for fragmented tracks by proximity to known tracks
            if team_assigner is not None:
                # Use a slightly larger proximity threshold to catch fragmented
                # tracks that appear near previous stable tracks.
                team_assigner.inherit_team_by_proximity(
                    detections, frame=frame, max_dist=0.12
                )
            if ball_interpolator is not None:
                ball_interpolator.record_frame_tracks(
                    frame_idx, detector.tracks.get("ball", {}).get(frame_idx, {})
                )
            if team_assigner is not None:
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
            motion: CameraMotionEstimate | None = None
            if config.enable_camera_motion and prev_frame is not None:
                mask = build_pitch_band_mask(
                    frame.shape,
                    top_ratio=config.camera_motion_top_ratio,
                    bottom_ratio=config.camera_motion_bottom_ratio,
                    side_margin_ratio=config.camera_motion_side_margin_ratio,
                )
                motion = camera_motion_estimator.update(
                    prev_frame, frame, mask=mask, frame_idx=frame_idx
                )
                if motion.num_points >= config.camera_motion_min_points:
                    cumulative_camera_offset += np.array(
                        [motion.dx_px, motion.dy_px], dtype=np.float32
                    )
                else:
                    motion = CameraMotionEstimate(
                        frame_idx=frame_idx,
                        dx_px=0.0,
                        dy_px=0.0,
                        rotation_deg=0.0,
                        confidence=0.0,
                        num_points=0,
                    )
            else:
                motion = CameraMotionEstimate(
                    frame_idx=frame_idx,
                    dx_px=0.0,
                    dy_px=0.0,
                    rotation_deg=0.0,
                    confidence=0.0,
                    num_points=0,
                )

            frame_state: Dict[str, object] = {
                "camera_motion": {
                    "frame_idx": motion.frame_idx,
                    "dx_px": motion.dx_px,
                    "dy_px": motion.dy_px,
                    "rotation_deg": motion.rotation_deg,
                    "confidence": motion.confidence,
                    "num_points": motion.num_points,
                },
                "camera_offset_px": (
                    float(cumulative_camera_offset[0]),
                    float(cumulative_camera_offset[1]),
                ),
                "player_like_points": {},
                "ball_point": None,
                "possession": None,
                "pitch_transform": {
                    "mode": pitch_transform.mode,
                    "field_length_m": pitch_transform.field_length_m,
                    "field_width_m": pitch_transform.field_width_m,
                },
            }

            for det in detections:
                bbox = np.asarray(det.bbox, dtype=np.float32)
                raw_point = np.array(
                    [(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float32
                )
                if det.object_type == "ball":
                    raw_point = np.array(
                        [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                        dtype=np.float32,
                    )
                    stabilized_point = raw_point - cumulative_camera_offset
                    field_point = pitch_transform.image_to_field(stabilized_point)
                    frame_state["ball_point"] = {
                        "track_id": det.track_id,
                        "raw_px": raw_point.tolist(),
                        "stabilized_px": stabilized_point.tolist(),
                        "field_m": field_point.tolist(),
                    }
                elif det.object_type in {"player", "goalkeeper"}:
                    stabilized_point = raw_point - cumulative_camera_offset
                    field_point = pitch_transform.image_to_field(stabilized_point)
                    frame_state["player_like_points"][int(det.track_id)] = {
                        "track_id": int(det.track_id),
                        "object_type": det.object_type,
                        "team_id": det.team_id,
                        "raw_px": raw_point.tolist(),
                        "stabilized_px": stabilized_point.tolist(),
                        "field_m": field_point.tolist(),
                    }

            possession = possession_tracker.update(
                frame_idx,
                detections,
                pitch_transform=pitch_transform,
                camera_motion=motion,
                cumulative_camera_offset=(
                    float(cumulative_camera_offset[0]),
                    float(cumulative_camera_offset[1]),
                ),
            )
            frame_state["possession"] = {
                "frame_idx": possession.frame_idx,
                "player_track_id": possession.player_track_id,
                "team_id": possession.team_id,
                "distance": possession.distance,
                "coordinate_space": possession.coordinate_space,
            }

            detector.frame_geometry[frame_idx] = frame_state
            annotated = draw_detections(frame, detections, frame_state=frame_state)
            writer.write(annotated)
            processed_count += 1
            prev_frame = frame.copy()

            if processed_count % 25 == 0:
                print(f"Processed {processed_count} frames (source frame {frame_idx})")

            if config.max_frames is not None and processed_count >= config.max_frames:
                break
    finally:
        reader.release()
        writer.release()

    if ball_interpolator is not None:
        interpolation_stats = ball_interpolator.interpolate_tracks(detector.tracks)
        print(
            "Ball interpolation summary - "
            f"observed_frames={interpolation_stats.observed_frames}, "
            f"interpolated_frames={interpolation_stats.interpolated_frames}, "
            f"max_gap={config.ball_interpolation_max_gap}"
        )
    else:
        print("Ball interpolation disabled for this run.")

    print(f"Tracking keys: {list(detector.tracks.keys())}")
    for bucket, frames in detector.tracks.items():
        tracked_objects = sum(len(frame_tracks) for frame_tracks in frames.values())
        print(
            f"Track summary - {bucket}: frames_with_tracks={len(frames)}, "
            f"tracked_objects={tracked_objects}"
        )

    if team_assigner is not None:
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
    else:
        print("Team assignment disabled for this run.")
    detector.possession_summary = possession_tracker.summary()
    print(
        "Possession summary - "
        f"team1={detector.possession_summary.get('team1_pct', 0.0):.1f}%, "
        f"team2={detector.possession_summary.get('team2_pct', 0.0):.1f}%, "
        f"unknown={detector.possession_summary.get('unknown_pct', 0.0):.1f}%"
    )
    if return_detector:
        return writer.output_path, detector
    return writer.output_path
