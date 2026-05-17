from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PitchTransform:
    image_width: int
    image_height: int
    field_length_m: float = 105.0
    field_width_m: float = 68.0
    mode: str = "scaled"
    scale_x_m_per_px: float = 1.0
    scale_y_m_per_px: float = 1.0

    @property
    def is_metric(self) -> bool:
        return self.mode != "identity"

    def image_to_field(
        self, point_xy: np.ndarray | list[float] | tuple[float, float]
    ) -> np.ndarray:
        point = np.asarray(point_xy, dtype=np.float32).reshape(-1)
        if point.shape[0] != 2:
            return np.zeros(2, dtype=np.float32)
        if self.mode == "identity":
            return point.astype(np.float32)
        return np.array(
            [point[0] * self.scale_x_m_per_px, point[1] * self.scale_y_m_per_px],
            dtype=np.float32,
        )

    def field_to_image(
        self, point_xy: np.ndarray | list[float] | tuple[float, float]
    ) -> np.ndarray:
        point = np.asarray(point_xy, dtype=np.float32).reshape(-1)
        if point.shape[0] != 2:
            return np.zeros(2, dtype=np.float32)
        if self.mode == "identity":
            return point.astype(np.float32)
        return np.array(
            [point[0] / self.scale_x_m_per_px, point[1] / self.scale_y_m_per_px],
            dtype=np.float32,
        )


def build_pitch_transform(
    image_size: tuple[int, int],
    mode: str = "scaled",
    field_length_m: float = 105.0,
    field_width_m: float = 68.0,
    meters_per_pixel: Optional[float] = None,
) -> PitchTransform:
    image_width, image_height = int(image_size[0]), int(image_size[1])
    if image_width <= 0 or image_height <= 0:
        image_width, image_height = 1, 1

    mode = (mode or "scaled").lower().strip()
    if mode == "identity":
        return PitchTransform(
            image_width=image_width,
            image_height=image_height,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
            mode="identity",
            scale_x_m_per_px=1.0,
            scale_y_m_per_px=1.0,
        )

    if meters_per_pixel is not None and meters_per_pixel > 0:
        scale_x = float(meters_per_pixel)
        scale_y = float(meters_per_pixel)
    else:
        scale_x = float(field_length_m) / float(image_width)
        scale_y = float(field_width_m) / float(image_height)

    return PitchTransform(
        image_width=image_width,
        image_height=image_height,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        mode="scaled",
        scale_x_m_per_px=scale_x,
        scale_y_m_per_px=scale_y,
    )


def bbox_to_field_point(
    bbox: list[float],
    transform: PitchTransform,
    anchor: str = "bottom_center",
) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox)
    if anchor == "bottom_center":
        point = np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)
    else:
        point = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    return transform.image_to_field(point)
