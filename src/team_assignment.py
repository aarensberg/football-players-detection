from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection_tracking import Detection


class TeamAssigner:
    """Inference-only team assignment from robust jersey-color descriptors."""

    def __init__(
        self,
        bootstrap_frames: int = 100,
        min_features_to_fit: int = 36,
        min_crop_size: int = 8,
        torso_y_start_ratio: float = 0.16,
        torso_y_end_ratio: float = 0.62,
        torso_side_margin_ratio: float = 0.16,
        per_track_history: int = 10,
        per_track_bootstrap_limit: int = 8,
        init_votes_required: int = 3,
        init_majority_ratio: float = 0.6,
        switch_votes_required: int = 5,
        switch_majority_ratio: float = 0.78,
        confidence_margin: float = 0.06,
        goalkeeper_ambiguity_margin: float = 0.045,
        goalkeeper_max_distance: float = 0.60,
        cluster_imbalance_ratio: float = 0.15,
        side_separation_min: float = 0.08,
    ) -> None:
        self.bootstrap_frames = max(1, int(bootstrap_frames))
        self.min_features_to_fit = max(2, int(min_features_to_fit))
        self.min_crop_size = max(2, int(min_crop_size))
        self.torso_y_start_ratio = float(max(0.0, min(0.7, torso_y_start_ratio)))
        self.torso_y_end_ratio = float(max(self.torso_y_start_ratio + 0.1, min(1.0, torso_y_end_ratio)))
        self.torso_side_margin_ratio = float(max(0.0, min(0.35, torso_side_margin_ratio)))
        self.per_track_history = max(3, int(per_track_history))
        self.per_track_bootstrap_limit = max(1, int(per_track_bootstrap_limit))
        self.init_votes_required = max(1, int(init_votes_required))
        self.init_majority_ratio = float(max(0.5, min(1.0, init_majority_ratio)))
        self.switch_votes_required = max(2, int(switch_votes_required))
        self.switch_majority_ratio = float(max(0.5, min(1.0, switch_majority_ratio)))
        self.confidence_margin = float(max(0.0, confidence_margin))
        self.goalkeeper_ambiguity_margin = float(max(0.0, goalkeeper_ambiguity_margin))
        self.goalkeeper_max_distance = float(max(0.0, goalkeeper_max_distance))
        self.cluster_imbalance_ratio = float(max(0.01, min(0.49, cluster_imbalance_ratio)))
        self.side_separation_min = float(max(0.0, min(1.0, side_separation_min)))

        self.team_centroids: Optional[np.ndarray] = None  # shape (2, 3), normalized HSV
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.track_to_team: Dict[int, int] = {}
        self._track_feature_history: Dict[int, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.per_track_history)
        )
        self._track_votes: Dict[int, Deque[int]] = defaultdict(
            lambda: deque(maxlen=max(self.per_track_history, self.switch_votes_required + 2))
        )
        self._track_x_history: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=24))
        self._bootstrap_track_samples: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._bootstrap_track_x: Dict[int, List[float]] = defaultdict(list)
        self._bootstrap_track_type: Dict[int, str] = {}
        self._frames_seen = 0
        self._assignment_events = {1: 0, 2: 0}
        self._warnings: List[str] = []
        self._fit_imbalanced = False

    @property
    def is_fitted(self) -> bool:
        return self.team_centroids is not None and len(self.team_colors) == 2

    def assign(self, frame: np.ndarray, detections: List[Detection]) -> None:
        self._frames_seen += 1

        player_like = [det for det in detections if det.object_type in {"player", "goalkeeper"}]

        if self._frames_seen <= self.bootstrap_frames:
            for det in player_like:
                feature = self._extract_feature(frame, det.bbox)
                if feature is None:
                    continue
                self._update_track_feature(det.track_id, feature)
                if det.track_id >= 0:
                    current_samples = self._bootstrap_track_samples[det.track_id]
                    if len(current_samples) < self.per_track_bootstrap_limit:
                        current_samples.append(feature)
                        self._bootstrap_track_type[det.track_id] = det.object_type
                        self._bootstrap_track_x[det.track_id].append(self._detection_center_x(frame, det))
                elif not self.is_fitted:
                    # Negative track IDs are unstable; do not use for fitting.
                    continue

            if not self.is_fitted and self._bootstrap_sample_count() >= self.min_features_to_fit:
                self._fit()

        for det in player_like:
            team_id, team_color = self._assign_detection(frame, det)
            det.team_id = team_id
            det.team_color = team_color
            if team_id is not None:
                self._assignment_events[team_id] += 1

    def _bootstrap_sample_count(self) -> int:
        return int(sum(len(v) for v in self._bootstrap_track_samples.values()))

    def _fit(self) -> None:
        samples, sample_x = self._build_fit_samples()
        if samples is None or sample_x is None:
            return

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.1,
        )
        _, labels, centers = cv2.kmeans(
            data=samples,
            K=2,
            bestLabels=None,
            criteria=criteria,
            attempts=10,
            flags=cv2.KMEANS_PP_CENTERS,
        )
        if centers is None or centers.shape != (2, 3) or labels is None:
            return
        labels = labels.reshape(-1)
        order = self._cluster_to_team_order(labels, sample_x)
        ordered = centers[order].astype(np.float32)
        self.team_centroids = ordered
        self.team_colors = {idx + 1: self._hsv_feature_to_bgr(center) for idx, center in enumerate(ordered)}

        counts = np.bincount(labels, minlength=2).astype(int)
        total = int(np.sum(counts))
        if total > 0 and float(np.min(counts) / total) < self.cluster_imbalance_ratio:
            self._fit_imbalanced = True
            warning = (
                f"[WARN] Team-color clustering imbalance: counts={counts.tolist()} "
                f"(ratio={float(np.min(counts)/total):.3f}). Assignments will be cautious."
            )
            print(warning)
            self._warnings.append(warning)
        else:
            self._fit_imbalanced = False

    def _assign_detection(
        self, frame: np.ndarray, det: Detection
    ) -> tuple[Optional[int], Optional[Tuple[int, int, int]]]:
        self._update_track_x(det, frame=frame)
        feature = self._extract_feature(frame, det.bbox)
        if feature is not None and det.track_id >= 0:
            self._update_track_feature(det.track_id, feature)
        descriptor = self._track_descriptor(det.track_id, fallback_feature=feature)

        if not self.is_fitted or descriptor is None:
            if det.track_id >= 0 and det.track_id in self.track_to_team:
                cached_team = self.track_to_team[det.track_id]
                return cached_team, self.team_colors.get(cached_team)
            return None, None

        raw_team, min_dist, margin = self._closest_team(descriptor)
        if raw_team is None:
            return None, None

        confidence_ok = margin >= self.confidence_margin
        if self._fit_imbalanced:
            confidence_ok = confidence_ok and margin >= self.confidence_margin * 1.25

        if det.object_type == "goalkeeper":
            team_id = self._assign_goalkeeper_track(det.track_id, raw_team, min_dist, margin)
        else:
            vote = raw_team if confidence_ok else 0
            team_id = self._resolve_stable_track_team(det.track_id, vote)

        if team_id is None and det.track_id >= 0 and det.track_id in self.track_to_team:
            team_id = self.track_to_team[det.track_id]

        if team_id is not None and det.track_id >= 0:
            self.track_to_team[det.track_id] = team_id

        return team_id, self.team_colors.get(team_id)

    def _extract_feature(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0 or len(bbox) != 4:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1_i = max(0, min(w - 1, int(np.floor(x1))))
        y1_i = max(0, min(h - 1, int(np.floor(y1))))
        x2_i = max(0, min(w, int(np.ceil(x2))))
        y2_i = max(0, min(h, int(np.ceil(y2))))

        bbox_w = x2_i - x1_i
        bbox_h = y2_i - y1_i
        if bbox_w < self.min_crop_size or bbox_h < self.min_crop_size:
            return None

        left = x1_i + int(round(bbox_w * self.torso_side_margin_ratio))
        right = x2_i - int(round(bbox_w * self.torso_side_margin_ratio))
        top = y1_i + int(round(bbox_h * self.torso_y_start_ratio))
        bottom = y1_i + int(round(bbox_h * self.torso_y_end_ratio))
        left = max(x1_i, min(right - self.min_crop_size, left))
        right = min(x2_i, max(left + self.min_crop_size, right))
        top = max(y1_i, min(bottom - self.min_crop_size, top))
        bottom = min(y2_i, max(top + self.min_crop_size, bottom))

        if right - left < self.min_crop_size or bottom - top < self.min_crop_size:
            return None

        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv_crop.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 2:
            return None

        labels = self._kmeans_two_labels(pixels)
        if labels is None:
            return self._normalize_hsv(np.median(pixels, axis=0))

        jersey_mask = self._jersey_mask_from_corners(labels.reshape(hsv_crop.shape[:2]))
        jersey_pixels = pixels[jersey_mask.reshape(-1)]
        if len(jersey_pixels) < max(12, int(0.04 * len(pixels))):
            # fallback to robust saturated pixels if segmentation is weak
            sat_mask = pixels[:, 1] > np.percentile(pixels[:, 1], 40)
            jersey_pixels = pixels[sat_mask]
        if len(jersey_pixels) < 4:
            return self._normalize_hsv(np.median(pixels, axis=0))
        return self._normalize_hsv(np.median(jersey_pixels, axis=0))

    def _kmeans_two_labels(self, pixels: np.ndarray) -> Optional[np.ndarray]:
        if pixels.ndim != 2 or pixels.shape[1] != 3 or len(pixels) < 2:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 0.2)
        try:
            _, labels, _ = cv2.kmeans(
                data=pixels.astype(np.float32),
                K=2,
                bestLabels=None,
                criteria=criteria,
                attempts=2,
                flags=cv2.KMEANS_PP_CENTERS,
            )
        except cv2.error:
            return None
        if labels is None:
            return None
        return labels.reshape(-1).astype(int)

    def _jersey_mask_from_corners(self, labels_2d: np.ndarray) -> np.ndarray:
        h, w = labels_2d.shape[:2]
        patch_h = max(1, int(0.2 * h))
        patch_w = max(1, int(0.2 * w))
        corner_labels = np.concatenate(
            [
                labels_2d[:patch_h, :patch_w].reshape(-1),
                labels_2d[:patch_h, w - patch_w :].reshape(-1),
                labels_2d[h - patch_h :, :patch_w].reshape(-1),
                labels_2d[h - patch_h :, w - patch_w :].reshape(-1),
            ]
        )
        non_jersey = int(np.bincount(corner_labels, minlength=2).argmax())
        jersey = 1 - non_jersey
        return labels_2d == jersey

    @staticmethod
    def _normalize_hsv(hsv: np.ndarray) -> np.ndarray:
        arr = hsv.astype(np.float32).reshape(-1)
        if arr.shape[0] != 3:
            return np.zeros(3, dtype=np.float32)
        return np.asarray([arr[0] / 179.0, arr[1] / 255.0, arr[2] / 255.0], dtype=np.float32)

    @staticmethod
    def _hsv_feature_to_bgr(feature: np.ndarray) -> Tuple[int, int, int]:
        hsv = np.asarray(
            [
                np.clip(feature[0] * 179.0, 0.0, 179.0),
                np.clip(feature[1] * 255.0, 0.0, 255.0),
                np.clip(feature[2] * 255.0, 0.0, 255.0),
            ],
            dtype=np.uint8,
        ).reshape(1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1)
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _update_track_feature(self, track_id: int, feature: np.ndarray) -> None:
        if track_id < 0:
            return
        self._track_feature_history[track_id].append(feature.astype(np.float32))

    def _track_descriptor(
        self, track_id: int, fallback_feature: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if track_id >= 0:
            history = self._track_feature_history.get(track_id)
            if history:
                stacked = np.stack(list(history), axis=0)
                return np.median(stacked, axis=0).astype(np.float32)
        if fallback_feature is None:
            return None
        return fallback_feature.astype(np.float32)

    def _build_fit_samples(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        player_rows: List[np.ndarray] = []
        player_x: List[np.ndarray] = []
        keeper_rows: List[np.ndarray] = []
        keeper_x: List[np.ndarray] = []

        for track_id, feats in self._bootstrap_track_samples.items():
            if not feats:
                continue
            object_type = self._bootstrap_track_type.get(track_id, "player")
            x_vals = self._bootstrap_track_x.get(track_id, [])
            row_feats = np.asarray(feats, dtype=np.float32)
            if row_feats.ndim != 2 or row_feats.shape[1] != 3:
                continue
            row_x = np.asarray(x_vals[: len(row_feats)], dtype=np.float32)
            if len(row_x) != len(row_feats):
                row_x = np.full(len(row_feats), 0.5, dtype=np.float32)
            if object_type == "goalkeeper":
                keeper_rows.append(row_feats)
                keeper_x.append(row_x)
            else:
                player_rows.append(row_feats)
                player_x.append(row_x)

        if player_rows:
            samples = np.concatenate(player_rows, axis=0)
            sample_x = np.concatenate(player_x, axis=0)
        elif keeper_rows:
            samples = np.concatenate(keeper_rows, axis=0)
            sample_x = np.concatenate(keeper_x, axis=0)
        else:
            return None, None

        if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < self.min_features_to_fit:
            return None, None

        if keeper_rows and player_rows:
            limited_keepers = np.concatenate([x[:2] for x in keeper_rows], axis=0)
            limited_keepers_x = np.concatenate([x[:2] for x in keeper_x], axis=0)
            samples = np.concatenate([np.concatenate(player_rows, axis=0), limited_keepers], axis=0)
            sample_x = np.concatenate([np.concatenate(player_x, axis=0), limited_keepers_x], axis=0)

        return samples.astype(np.float32), sample_x.astype(np.float32)

    def _cluster_to_team_order(self, labels: np.ndarray, sample_x: np.ndarray) -> np.ndarray:
        counts = np.bincount(labels, minlength=2).astype(int)
        x_medians = np.asarray(
            [
                float(np.median(sample_x[labels == i])) if np.any(labels == i) else 0.5
                for i in range(2)
            ],
            dtype=np.float32,
        )
        if abs(float(x_medians[0] - x_medians[1])) >= self.side_separation_min:
            team1_cluster = int(np.argmin(x_medians))
            team2_cluster = 1 - team1_cluster
            return np.asarray([team1_cluster, team2_cluster], dtype=int)
        # Fallback: larger cluster is team 1 for deterministic mapping.
        team1_cluster = int(np.argmax(counts))
        team2_cluster = 1 - team1_cluster
        return np.asarray([team1_cluster, team2_cluster], dtype=int)

    def _closest_team(self, descriptor: np.ndarray) -> tuple[Optional[int], float, float]:
        if self.team_centroids is None:
            return None, float("inf"), 0.0
        distances = np.linalg.norm(self.team_centroids - descriptor.reshape(1, 3), axis=1)
        if distances.shape[0] != 2:
            return None, float("inf"), 0.0
        closest = int(np.argmin(distances))
        other = 1 - closest
        margin = float(distances[other] - distances[closest])
        return closest + 1, float(distances[closest]), margin

    def _resolve_stable_track_team(self, track_id: int, vote: int) -> Optional[int]:
        if track_id < 0:
            return vote if vote in {1, 2} else None
        if vote in {1, 2}:
            self._track_votes[track_id].append(vote)

        existing = self.track_to_team.get(track_id)
        votes = [v for v in self._track_votes.get(track_id, []) if v in {1, 2}]
        if not votes:
            return existing
        counter = Counter(votes)
        majority_team, majority_count = counter.most_common(1)[0]
        majority_ratio = majority_count / max(1, len(votes))

        if existing is None:
            if len(votes) >= self.init_votes_required and majority_ratio >= self.init_majority_ratio:
                return int(majority_team)
            return None

        if majority_team == existing:
            return existing

        if len(votes) >= self.switch_votes_required and majority_ratio >= self.switch_majority_ratio:
            return int(majority_team)
        return existing

    def _assign_goalkeeper_track(
        self, track_id: int, raw_team: int, min_dist: float, margin: float
    ) -> Optional[int]:
        use_color = margin >= self.goalkeeper_ambiguity_margin and min_dist <= self.goalkeeper_max_distance
        color_vote = raw_team if use_color else 0
        stable_from_color = self._resolve_stable_track_team(track_id, color_vote)
        if stable_from_color is not None:
            return stable_from_color
        fallback = self._goalkeeper_side_fallback(track_id)
        if fallback is not None:
            return fallback
        return raw_team if use_color else None

    def _goalkeeper_side_fallback(self, track_id: int) -> Optional[int]:
        if track_id < 0:
            return None
        xs = self._track_x_history.get(track_id)
        if not xs:
            return None
        median_x = float(np.median(np.asarray(xs, dtype=np.float32)))
        return 1 if median_x < 0.5 else 2

    def _update_track_x(self, det: Detection, frame: Optional[np.ndarray] = None) -> None:
        if det.track_id < 0:
            return
        if frame is None:
            # x history is optional for fit-time bookkeeping.
            return
        cx = self._detection_center_x(frame, det)
        self._track_x_history[det.track_id].append(cx)

    @staticmethod
    def _detection_center_x(frame: np.ndarray, det: Detection) -> float:
        x1, _, x2, _ = det.bbox
        w = max(1.0, float(frame.shape[1]))
        return float(np.clip((float(x1) + float(x2)) * 0.5 / w, 0.0, 1.0))

    def summary(self) -> Dict[str, object]:
        team_track_counts = {1: 0, 2: 0}
        for team_id in self.track_to_team.values():
            if team_id in team_track_counts:
                team_track_counts[team_id] += 1
        return {
            "fitted": self.is_fitted,
            "frames_seen": self._frames_seen,
            "bootstrap_features": self._bootstrap_sample_count(),
            "track_team_counts": team_track_counts,
            "assignment_events": dict(self._assignment_events),
            "warnings": list(self._warnings),
            "fit_imbalanced": self._fit_imbalanced,
        }
