from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.detection_tracking import Detection


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    rendered = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = _color_for_label(det.label)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label} id={det.track_id} conf={det.score:.2f}"
        cv2.putText(
            rendered,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered


def _color_for_label(label: str) -> tuple[int, int, int]:
    name = label.lower()
    if name == "ball":
        return (0, 255, 255)
    if name == "referee":
        return (255, 0, 255)
    if name in {"player", "goalkeeper"}:
        return (0, 255, 0)
    return (255, 255, 255)
