from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.detection_tracking import Detection


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    rendered = frame.copy()
    h, w = rendered.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = _clip_bbox(det.bbox, w, h)
        if x2 <= x1 or y2 <= y1:
            continue

        obj_type = det.object_type.lower()
        base_color = (
            _normalize_color(det.team_color)
            if obj_type in {"player", "goalkeeper"} and det.team_color is not None
            else _color_for_label(obj_type)
        )

        if obj_type == "ball":
            _draw_ball_marker(rendered, x1, y1, x2, y2)
        elif obj_type == "goalkeeper":
            _draw_player_marker(rendered, x1, y1, x2, y2, base_color, box_thickness=3)
            _draw_corner_marks(rendered, x1, y1, x2, y2, (255, 255, 255))
        elif obj_type == "referee":
            referee_color = (160, 160, 160)
            _draw_player_marker(
                rendered, x1, y1, x2, y2, referee_color, box_thickness=2, ellipse_fill=False
            )
        else:
            _draw_player_marker(rendered, x1, y1, x2, y2, base_color, box_thickness=2)

        label = _build_label(det)
        _draw_label(rendered, label, x1, y1, base_color)
    return rendered


def _color_for_label(label: str) -> tuple[int, int, int]:
    name = label.lower()
    if name == "ball":
        return (0, 220, 255)
    if name == "referee":
        return (160, 160, 160)
    if name == "goalkeeper":
        return (0, 140, 255)
    if name == "player":
        return (0, 255, 0)
    return (255, 255, 255)


def _clip_bbox(bbox: List[float], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w - 1, x2))
    y2 = max(0, min(frame_h - 1, y2))
    return x1, y1, x2, y2


def _normalize_color(color: Tuple[int, int, int] | None) -> Tuple[int, int, int]:
    if color is None:
        return (255, 255, 255)
    return tuple(int(np.clip(c, 0, 255)) for c in color)


def _draw_player_marker(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    box_thickness: int = 2,
    ellipse_fill: bool = True,
) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
    cx = (x1 + x2) // 2
    cy = y2
    rx = max(8, (x2 - x1) // 2)
    ry = max(4, (y2 - y1) // 8)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, color, 2)
    if ellipse_fill:
        cv2.ellipse(frame, (cx, cy), (max(2, rx - 2), max(1, ry - 2)), 0, 0, 360, color, -1)


def _draw_corner_marks(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]
) -> None:
    corner_len = max(6, min((x2 - x1) // 4, (y2 - y1) // 4))
    thickness = 2
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)


def _draw_ball_marker(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = max(4, min(14, int(0.45 * max(x2 - x1, y2 - y1))))
    cv2.circle(frame, (cx, cy), radius + 2, (0, 0, 0), -1)
    cv2.circle(frame, (cx, cy), radius, (0, 220, 255), -1)
    cv2.circle(frame, (cx, cy), radius + 1, (255, 255, 255), 2)


def _build_label(det: Detection) -> str:
    parts = [det.object_type, f"id={det.track_id}"]
    if det.team_id is not None:
        parts.append(f"team={det.team_id}")
    if det.model_class.lower() != det.object_type.lower():
        parts.append(f"raw={det.model_class}")
    parts.append(f"conf={det.score:.2f}")
    return " ".join(parts)


def _draw_label(
    frame: np.ndarray, text: str, x: int, y: int, accent_color: Tuple[int, int, int]
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x0 = max(0, x)
    y0 = max(th + 8, y - 8)
    y1 = y0 - th - baseline - 6
    x1 = x0 + tw + 8

    cv2.rectangle(frame, (x0, y1), (x1, y0), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y1), (x1, y0), accent_color, 1)
    cv2.putText(
        frame,
        text,
        (x0 + 4, y0 - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
