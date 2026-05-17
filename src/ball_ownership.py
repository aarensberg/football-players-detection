from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.camera_motion import CameraMotionEstimate
from src.detection_tracking import Detection
from src.pitch_geometry import PitchTransform


@dataclass
class PossessionObservation:
    frame_idx: int
    player_track_id: Optional[int]
    team_id: Optional[int]
    distance: Optional[float]
    coordinate_space: str


@dataclass
class PossessionSummary:
    total_visible_frames: int = 0
    team1_frames: int = 0
    team2_frames: int = 0
    unknown_frames: int = 0
    observations: List[PossessionObservation] = field(default_factory=list)

    @property
    def team1_pct(self) -> float:
        return 100.0 * self.team1_frames / max(1, self.total_visible_frames)

    @property
    def team2_pct(self) -> float:
        return 100.0 * self.team2_frames / max(1, self.total_visible_frames)

    @property
    def unknown_pct(self) -> float:
        return 100.0 * self.unknown_frames / max(1, self.total_visible_frames)


class BallPossessionTracker:
    def __init__(
        self,
        max_distance_px: float = 48.0,
        max_distance_m: float = 2.8,
    ) -> None:
        self.max_distance_px = float(max(1.0, max_distance_px))
        self.max_distance_m = float(max(0.25, max_distance_m))
        self.summary_state = PossessionSummary()
        self.by_frame: Dict[int, PossessionObservation] = {}

    @staticmethod
    def _point_from_bbox(
        bbox: List[float], anchor: str = "bottom_center"
    ) -> np.ndarray:
        x1, y1, x2, y2 = map(float, bbox)
        if anchor == "bottom_center":
            return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def update(
        self,
        frame_idx: int,
        detections: List[Detection],
        pitch_transform: Optional[PitchTransform] = None,
        camera_motion: Optional[CameraMotionEstimate] = None,
        cumulative_camera_offset: Optional[Tuple[float, float]] = None,
    ) -> PossessionObservation:
        ball_det = next((det for det in detections if det.object_type == "ball"), None)
        player_dets = [
            det for det in detections if det.object_type in {"player", "goalkeeper"}
        ]

        if ball_det is None or not player_dets:
            obs = PossessionObservation(
                frame_idx=int(frame_idx),
                player_track_id=None,
                team_id=None,
                distance=None,
                coordinate_space="unknown",
            )
            self.summary_state.observations.append(obs)
            self.summary_state.unknown_frames += 1
            self.by_frame[int(frame_idx)] = obs
            return obs

        ball_point_px = self._point_from_bbox(ball_det.bbox, anchor="center")
        player_points: List[Tuple[Detection, np.ndarray]] = []
        for det in player_dets:
            point_px = self._point_from_bbox(det.bbox, anchor="bottom_center")
            player_points.append((det, point_px))

        if cumulative_camera_offset is not None:
            offset = np.array(cumulative_camera_offset, dtype=np.float32)
            ball_point_px = ball_point_px - offset
            player_points = [
                (det, point_px - offset) for det, point_px in player_points
            ]

        coordinate_space = "stabilized_px"
        ball_point = ball_point_px
        if pitch_transform is not None:
            coordinate_space = "field_m" if pitch_transform.is_metric else "identity"
            ball_point = pitch_transform.image_to_field(ball_point_px)
            player_points = [
                (det, pitch_transform.image_to_field(point_px))
                for det, point_px in player_points
            ]

        best_det: Optional[Detection] = None
        best_distance = float("inf")
        for det, point in player_points:
            dist = float(np.linalg.norm(point - ball_point))
            if dist < best_distance:
                best_distance = dist
                best_det = det

        threshold = (
            self.max_distance_m
            if pitch_transform is not None and pitch_transform.is_metric
            else self.max_distance_px
        )
        team_id = None
        track_id = None
        if best_det is not None and best_distance <= threshold:
            team_id = best_det.team_id
            track_id = best_det.track_id

        obs = PossessionObservation(
            frame_idx=int(frame_idx),
            player_track_id=track_id,
            team_id=team_id,
            distance=float(best_distance) if np.isfinite(best_distance) else None,
            coordinate_space=coordinate_space,
        )
        self.summary_state.observations.append(obs)
        self.summary_state.total_visible_frames += 1
        if team_id == 1:
            self.summary_state.team1_frames += 1
        elif team_id == 2:
            self.summary_state.team2_frames += 1
        else:
            self.summary_state.unknown_frames += 1
        self.by_frame[int(frame_idx)] = obs
        return obs

    def summary(self) -> Dict[str, object]:
        return {
            "total_visible_frames": self.summary_state.total_visible_frames,
            "team1_frames": self.summary_state.team1_frames,
            "team2_frames": self.summary_state.team2_frames,
            "unknown_frames": self.summary_state.unknown_frames,
            "team1_pct": self.summary_state.team1_pct,
            "team2_pct": self.summary_state.team2_pct,
            "unknown_pct": self.summary_state.unknown_pct,
        }
