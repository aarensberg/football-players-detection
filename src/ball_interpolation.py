from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class BallInterpolationStats:
    observed_frames: int
    interpolated_frames: int


class BallInterpolator:
    """Track-level linear interpolation utility for short ball gaps."""

    def __init__(self, max_gap_frames: int = 4) -> None:
        self.max_gap_frames = max(0, int(max_gap_frames))
        self.observed_by_frame: Dict[int, Dict[int, Dict[str, object]]] = {}

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
