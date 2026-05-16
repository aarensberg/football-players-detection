from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection_tracking import Detection


class TeamAssigner:
    """Inference-only team assignment from player jersey colors."""

    def __init__(
        self,
        bootstrap_frames: int = 80,
        min_features_to_fit: int = 30,
        upper_body_ratio: float = 0.55,
        min_crop_size: int = 4,
    ) -> None:
        self.bootstrap_frames = max(1, int(bootstrap_frames))
        self.min_features_to_fit = max(2, int(min_features_to_fit))
        self.upper_body_ratio = float(max(0.2, min(1.0, upper_body_ratio)))
        self.min_crop_size = max(2, int(min_crop_size))

        self.team_centroids: Optional[np.ndarray] = None  # shape (2, 3), BGR
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.track_to_team: Dict[int, int] = {}
        self._bootstrap_features: List[np.ndarray] = []
        self._frames_seen = 0
        self._assignment_events = {1: 0, 2: 0}

    @property
    def is_fitted(self) -> bool:
        return self.team_centroids is not None and len(self.team_colors) == 2

    def assign(self, frame: np.ndarray, detections: List[Detection]) -> None:
        self._frames_seen += 1

        player_like = [
            det for det in detections if det.object_type in {"player", "goalkeeper"}
        ]

        if not self.is_fitted and self._frames_seen <= self.bootstrap_frames:
            for det in player_like:
                feature = self._extract_feature(frame, det.bbox)
                if feature is not None:
                    self._bootstrap_features.append(feature)

            if len(self._bootstrap_features) >= self.min_features_to_fit:
                self._fit()

        for det in player_like:
            team_id, team_color = self._assign_detection(frame, det)
            det.team_id = team_id
            det.team_color = team_color
            if team_id is not None:
                self._assignment_events[team_id] += 1

    def _fit(self) -> None:
        if len(self._bootstrap_features) < self.min_features_to_fit:
            return

        samples = np.asarray(self._bootstrap_features, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 2:
            return

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            0.2,
        )
        compactness, labels, centers = cv2.kmeans(
            data=samples,
            K=2,
            bestLabels=None,
            criteria=criteria,
            attempts=10,
            flags=cv2.KMEANS_PP_CENTERS,
        )
        _ = compactness
        _ = labels
        if centers is None or centers.shape != (2, 3):
            return

        # Stable mapping: lower total intensity -> team 1, higher -> team 2.
        order = np.argsort(np.sum(centers, axis=1))
        ordered = centers[order].astype(np.float32)
        self.team_centroids = ordered
        self.team_colors = {
            idx + 1: tuple(int(np.clip(v, 0, 255)) for v in center)
            for idx, center in enumerate(ordered)
        }

    def _assign_detection(
        self, frame: np.ndarray, det: Detection
    ) -> tuple[Optional[int], Optional[Tuple[int, int, int]]]:
        if det.track_id >= 0 and det.track_id in self.track_to_team:
            team_id = self.track_to_team[det.track_id]
            return team_id, self.team_colors.get(team_id)

        if not self.is_fitted:
            return None, None

        feature = self._extract_feature(frame, det.bbox)
        if feature is None:
            return None, None

        distances = np.linalg.norm(self.team_centroids - feature.reshape(1, 3), axis=1)
        cluster_idx = int(np.argmin(distances))
        team_id = cluster_idx + 1
        team_color = self.team_colors.get(team_id)

        if det.track_id >= 0:
            self.track_to_team[det.track_id] = team_id

        return team_id, team_color

    def _extract_feature(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0 or len(bbox) != 4:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1_i = max(0, min(w - 1, int(np.floor(x1))))
        y1_i = max(0, min(h - 1, int(np.floor(y1))))
        x2_i = max(0, min(w, int(np.ceil(x2))))
        y2_i = max(0, min(h, int(np.ceil(y2))))

        if x2_i - x1_i < self.min_crop_size or y2_i - y1_i < self.min_crop_size:
            return None

        body_h = y2_i - y1_i
        upper_y2 = y1_i + max(self.min_crop_size, int(body_h * self.upper_body_ratio))
        upper_y2 = min(y2_i, upper_y2)
        if upper_y2 - y1_i < self.min_crop_size:
            return None

        crop = frame[y1_i:upper_y2, x1_i:x2_i]
        if crop.size == 0:
            return None

        pixels = crop.reshape(-1, 3).astype(np.float32)
        if len(pixels) == 0:
            return None

        return np.median(pixels, axis=0)

    def summary(self) -> Dict[str, object]:
        team_track_counts = {1: 0, 2: 0}
        for team_id in self.track_to_team.values():
            if team_id in team_track_counts:
                team_track_counts[team_id] += 1
        return {
            "fitted": self.is_fitted,
            "frames_seen": self._frames_seen,
            "bootstrap_features": len(self._bootstrap_features),
            "track_team_counts": team_track_counts,
            "assignment_events": dict(self._assignment_events),
        }
