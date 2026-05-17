from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotionEstimate:
    frame_idx: int
    dx_px: float
    dy_px: float
    rotation_deg: float
    confidence: float
    num_points: int


def build_pitch_band_mask(
    frame_shape: tuple[int, int, int],
    top_ratio: float = 0.22,
    bottom_ratio: float = 0.82,
    side_margin_ratio: float = 0.08,
) -> np.ndarray:
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    top_end = max(1, min(height, int(round(height * top_ratio))))
    bottom_start = max(0, min(height - 1, int(round(height * bottom_ratio))))
    side_margin = max(0, min(width // 2, int(round(width * side_margin_ratio))))

    if top_end > 0:
        mask[:top_end, side_margin : width - side_margin] = 255
    if bottom_start < height:
        mask[bottom_start:, side_margin : width - side_margin] = 255
    return mask


class CameraMotionEstimator:
    """Estimate small per-frame camera translations with sparse optical flow."""

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 12,
        block_size: int = 7,
    ) -> None:
        self.max_corners = max(20, int(max_corners))
        self.quality_level = float(max(1e-4, quality_level))
        self.min_distance = max(1, int(min_distance))
        self.block_size = max(3, int(block_size))

    def update(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        frame_idx: int = -1,
    ) -> CameraMotionEstimate:
        prev_gray = self._to_gray(prev_frame)
        curr_gray = self._to_gray(curr_frame)
        if prev_gray is None or curr_gray is None:
            return self._zero_estimate(frame_idx)

        features = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            mask=mask,
        )
        if features is None or len(features) == 0:
            return self._zero_estimate(frame_idx)

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            features,
            None,
        )
        if next_points is None or status is None:
            return self._zero_estimate(frame_idx)

        valid = status.reshape(-1).astype(bool)
        prev_valid = features.reshape(-1, 2)[valid]
        next_valid = next_points.reshape(-1, 2)[valid]
        if len(prev_valid) == 0:
            return self._zero_estimate(frame_idx)

        displacements = next_valid - prev_valid
        dx = float(np.median(displacements[:, 0]))
        dy = float(np.median(displacements[:, 1]))
        residual = np.linalg.norm(
            displacements - np.array([dx, dy], dtype=np.float32), axis=1
        )
        confidence = float(len(prev_valid) / max(1, len(features))) * float(
            np.clip(1.0 - (np.median(residual) / 25.0), 0.0, 1.0)
        )
        rotation_deg = 0.0
        if len(prev_valid) >= 3:
            centered_prev = prev_valid - np.mean(prev_valid, axis=0, keepdims=True)
            centered_next = next_valid - np.mean(next_valid, axis=0, keepdims=True)
            a = float(
                np.sum(
                    centered_prev[:, 0] * centered_next[:, 1]
                    - centered_prev[:, 1] * centered_next[:, 0]
                )
            )
            b = float(
                np.sum(
                    centered_prev[:, 0] * centered_next[:, 0]
                    + centered_prev[:, 1] * centered_next[:, 1]
                )
            )
            rotation_deg = float(np.degrees(np.arctan2(a, b)))

        return CameraMotionEstimate(
            frame_idx=int(frame_idx),
            dx_px=dx,
            dy_px=dy,
            rotation_deg=rotation_deg,
            confidence=confidence,
            num_points=int(len(prev_valid)),
        )

    @staticmethod
    def compensate_point(
        point_xy: Tuple[float, float], motion: CameraMotionEstimate
    ) -> np.ndarray:
        return np.array(
            [float(point_xy[0]) - motion.dx_px, float(point_xy[1]) - motion.dy_px],
            dtype=np.float32,
        )

    @staticmethod
    def compensate_bbox(
        bbox: list[float], motion: CameraMotionEstimate, anchor: str = "bottom_center"
    ) -> np.ndarray:
        x1, y1, x2, y2 = map(float, bbox)
        if anchor == "bottom_center":
            point = np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)
        else:
            point = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        return point - np.array([motion.dx_px, motion.dy_px], dtype=np.float32)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _zero_estimate(frame_idx: int) -> CameraMotionEstimate:
        return CameraMotionEstimate(
            frame_idx=int(frame_idx),
            dx_px=0.0,
            dy_px=0.0,
            rotation_deg=0.0,
            confidence=0.0,
            num_points=0,
        )
