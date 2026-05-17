from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class BallInterpolationStats:
    observed_frames: int
    interpolated_frames: int


class BallInterpolator:
    """Track-level linear interpolation utility for short ball gaps."""

    def __init__(
        self,
        max_gap_frames: int = 6,
        max_center_speed_px_per_frame: Optional[float] = 120.0,
        max_endpoint_area_change_ratio: Optional[float] = 3.5,
    ) -> None:
        self.max_gap_frames = max(0, int(max_gap_frames))
        self.max_center_speed_px_per_frame = (
            None
            if max_center_speed_px_per_frame is None
            else max(0.0, float(max_center_speed_px_per_frame))
        )
        self.max_endpoint_area_change_ratio = (
            None
            if max_endpoint_area_change_ratio is None
            else max(1.0, float(max_endpoint_area_change_ratio))
        )
        self.observed_by_frame: Dict[int, Dict[int, Dict[str, object]]] = {}

    @staticmethod
    def _bbox_center(bbox: list[float]) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=float)

    @staticmethod
    def _bbox_area(bbox: list[float]) -> float:
        return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

    def record_frame_tracks(self, frame_idx: int, frame_tracks: Dict[int, Dict[str, object]]) -> None:
        if not frame_tracks:
            return
        self.observed_by_frame[int(frame_idx)] = {
            int(track_id): dict(attrs) for track_id, attrs in frame_tracks.items()
        }

    def interpolate_tracks(
        self, tracks: Dict[str, Dict[int, Dict[int, Dict[str, object]]]]
    ) -> BallInterpolationStats:
        ball_tracks = tracks.setdefault("ball", {})
        if not self.observed_by_frame:
            for frame_idx, frame_tracks in ball_tracks.items():
                self.record_frame_tracks(frame_idx, frame_tracks)

        per_track: Dict[int, Dict[int, Dict[str, object]]] = {}
        for frame_idx, frame_tracks in self.observed_by_frame.items():
            for track_id, attrs in frame_tracks.items():
                per_track.setdefault(track_id, {})[frame_idx] = attrs

        interpolated_count = 0
        for track_id, by_frame in per_track.items():
            observed_frames = sorted(by_frame.keys())
            for left_idx, right_idx in zip(observed_frames, observed_frames[1:]):
                gap = right_idx - left_idx - 1
                if gap <= 0 or gap > self.max_gap_frames:
                    continue

                left = by_frame[left_idx]
                right = by_frame[right_idx]
                left_bbox = [float(v) for v in left.get("bbox", [0.0, 0.0, 0.0, 0.0])]
                right_bbox = [float(v) for v in right.get("bbox", [0.0, 0.0, 0.0, 0.0])]
                dt = right_idx - left_idx
                if dt <= 0:
                    continue

                if self.max_center_speed_px_per_frame is not None:
                    c_left = self._bbox_center(left_bbox)
                    c_right = self._bbox_center(right_bbox)
                    speed = float(np.linalg.norm(c_right - c_left)) / float(dt)
                    if speed > self.max_center_speed_px_per_frame:
                        continue

                if self.max_endpoint_area_change_ratio is not None:
                    left_area = self._bbox_area(left_bbox)
                    right_area = self._bbox_area(right_bbox)
                    min_area = min(left_area, right_area)
                    max_area = max(left_area, right_area)
                    if min_area <= 0.0 or (max_area / min_area) > self.max_endpoint_area_change_ratio:
                        continue

                for step in range(1, gap + 1):
                    frame_idx = left_idx + step
                    alpha = step / float(gap + 1)
                    interp_bbox = [
                        left_bbox[i] + alpha * (right_bbox[i] - left_bbox[i]) for i in range(4)
                    ]
                    frame_bucket = ball_tracks.setdefault(frame_idx, {})
                    if track_id in frame_bucket:
                        continue
                    score = float(left.get("score", right.get("score", 0.0)))
                    frame_bucket[track_id] = {
                        "bbox": interp_bbox,
                        "score": score,
                        "class_id": left.get("class_id", right.get("class_id", -1)),
                        "model_class": left.get("model_class", right.get("model_class", "ball")),
                        "class_name": left.get("class_name", right.get("class_name", "ball")),
                        "object_type": "ball",
                        "status": "interpolated",
                        "interpolated": True,
                    }
                    interpolated_count += 1

        return BallInterpolationStats(
            observed_frames=len(self.observed_by_frame),
            interpolated_frames=interpolated_count,
        )
