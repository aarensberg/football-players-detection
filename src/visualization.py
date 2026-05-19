from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection_tracking import Detection
from src.pitch_geometry import build_pitch_transform, PitchTransform


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    frame_state: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
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
                rendered,
                x1,
                y1,
                x2,
                y2,
                referee_color,
                box_thickness=2,
                ellipse_fill=False,
            )
        else:
            _draw_player_marker(rendered, x1, y1, x2, y2, base_color, box_thickness=2)

        label = _build_label(det)
        _draw_label(rendered, label, x1, y1, base_color)

    if frame_state:
        _draw_hud(rendered, frame_state)
        _draw_team_lines(rendered, detections, frame_state)
        _draw_minimap(rendered, detections, frame_state)
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


def _clip_bbox(
    bbox: List[float], frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
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
        cv2.ellipse(
            frame, (cx, cy), (max(2, rx - 2), max(1, ry - 2)), 0, 0, 360, color, -1
        )


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


def _draw_hud(frame: np.ndarray, frame_state: Dict[str, Any]) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (430, 118), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)

    motion = frame_state.get("camera_motion") or {}
    possession = frame_state.get("possession") or {}
    offset = frame_state.get("camera_offset_px") or (0.0, 0.0)
    transform = frame_state.get("pitch_transform") or {}

    lines = [
        f"Camera dx/dy: {motion.get('dx_px', 0.0):.1f}, {motion.get('dy_px', 0.0):.1f} px",
        f"Camera offset: {offset[0]:.1f}, {offset[1]:.1f} px",
        f"Possession: team {possession.get('team_id', 'N/A')} | track {possession.get('player_track_id', 'N/A')}",
        f"Field mode: {transform.get('mode', 'n/a')} | {transform.get('field_length_m', 0.0):.0f}x{transform.get('field_width_m', 0.0):.0f} m",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 34 + idx * 22),
            font,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_team_lines(
    frame: np.ndarray, detections: List[Detection], frame_state: Dict[str, Any]
) -> None:
    """Draw side-view formation lines for each team using x-axis grouping."""
    h, w = frame.shape[:2]

    # Group players by team and use stabilized positions when available.
    teams_players: Dict[int, List[Tuple[int, int, Detection]]] = {}
    stabilized_points = frame_state.get("player_like_points", {}) if frame_state else {}
    for det in detections:
        if det.object_type not in {"player", "goalkeeper"}:
            continue
        if det.team_id is None:
            continue

        if det.team_id not in teams_players:
            teams_players[det.team_id] = []

        # Get center bottom of bbox (where player stands)
        x1, y1, x2, y2 = _clip_bbox(det.bbox, w, h)
        cx = (x1 + x2) // 2
        cy = y2

        # If a stabilized position exists, reproject it to current frame
        offset_px = (0.0, 0.0)
        try:
            offset_px = (
                frame_state.get("camera_offset_px", (0.0, 0.0))
                if frame_state
                else (0.0, 0.0)
            )
        except Exception:
            offset_px = (0.0, 0.0)

        tracked_player = stabilized_points.get(int(det.track_id))
        if tracked_player and tracked_player.get("stabilized_px"):
            stabilized_x, stabilized_y = tracked_player["stabilized_px"]
            cx = int(stabilized_x + float(offset_px[0]))
            cy = int(stabilized_y + float(offset_px[1]))

        teams_players[det.team_id].append((cx, cy, det))

    # Draw formation lines by slicing the pitch along the horizontal axis.
    for team_id, players in teams_players.items():
        if len(players) < 2:
            continue

        # Estimate formation layers along x: defensive, midfield, attacking.
        xs = np.array([x for x, _, _ in players], dtype=np.float32)
        if len(players) >= 5:
            thresholds = np.percentile(xs, [33.3, 66.6])
            band_count = 3
        elif len(players) >= 3:
            thresholds = np.percentile(xs, [50.0])
            band_count = 2
        else:
            thresholds = np.array([], dtype=np.float32)
            band_count = 1

        bands: List[List[Tuple[int, int, Detection]]] = [[] for _ in range(band_count)]
        for x, y, det in players:
            band_idx = int(np.searchsorted(thresholds, x, side="right"))
            bands[band_idx].append((x, y, det))

        for band_idx, band_players in enumerate(bands):
            if len(band_players) < 2:
                continue

            # Sort top-to-bottom within each band and connect with the team color.
            band_players_sorted = sorted(band_players, key=lambda p: p[1])
            color = _normalize_color(band_players_sorted[0][2].team_color)
            thickness = 3 if band_idx == 1 and band_count == 3 else 2

            for i in range(len(band_players_sorted) - 1):
                x1, y1, _ = band_players_sorted[i]
                x2, y2, _ = band_players_sorted[i + 1]
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def _draw_minimap(
    frame: np.ndarray, detections: List[Detection], frame_state: Dict[str, Any]
) -> None:
    """Draw a small field map in bottom-right corner with a dynamic visible pitch window."""
    h, w = frame.shape[:2]

    # Build pitch transform based on frame dimensions
    pitch_transform = build_pitch_transform(
        image_size=(w, h),
        mode="scaled",
        field_length_m=105.0,
        field_width_m=68.0,
    )

    # Mini-map dimensions and position
    minimap_width = 320
    minimap_height = 200
    minimap_x = w - minimap_width - 15
    minimap_y = h - minimap_height - 15

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (minimap_x, minimap_y), (w - 15, h - 15), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw field outline (represents full 105m x 68m)
    cv2.rectangle(frame, (minimap_x, minimap_y), (w - 15, h - 15), (255, 255, 255), 2)

    # Draw center line and circle (field reference markings)
    center_x = minimap_x + minimap_width // 2
    center_y = minimap_y + minimap_height // 2
    cv2.line(frame, (center_x, minimap_y), (center_x, h - 15), (200, 200, 200), 1)
    cv2.circle(frame, (center_x, center_y), 15, (200, 200, 200), 1)

    # Add small "goal" rectangles at ends
    cv2.rectangle(
        frame,
        (minimap_x, center_y - 10),
        (minimap_x + 3, center_y + 10),
        (100, 100, 255),
        -1,
    )
    cv2.rectangle(
        frame, (w - 18, center_y - 10), (w - 15, center_y + 10), (255, 100, 100), -1
    )

    # Estimate the visible horizontal pitch window from the actors currently on screen.
    visible_field_points: List[Tuple[float, float]] = []
    player_like_points = (frame_state or {}).get("player_like_points", {})
    for player_state in player_like_points.values():
        field_point = player_state.get("field_m")
        if field_point and len(field_point) == 2:
            visible_field_points.append((float(field_point[0]), float(field_point[1])))

    ball_state = (frame_state or {}).get("ball_point")
    if ball_state and ball_state.get("field_m") and len(ball_state["field_m"]) == 2:
        visible_field_points.append(
            (float(ball_state["field_m"][0]), float(ball_state["field_m"][1]))
        )

    if visible_field_points:
        x_values = np.array([pt[0] for pt in visible_field_points], dtype=np.float32)
        x_min = float(np.percentile(x_values, 5))
        x_max = float(np.percentile(x_values, 95))
        x_span = max(12.0, x_max - x_min)
        x_margin = max(8.0, 0.22 * x_span)
        window_x_min = max(0.0, x_min - x_margin)
        window_x_max = min(105.0, x_max + x_margin)
        if window_x_max - window_x_min < 18.0:
            center_x = float(np.median(x_values))
            window_x_min = max(0.0, center_x - 12.0)
            window_x_max = min(105.0, center_x + 12.0)
    else:
        window_x_min = 0.0
        window_x_max = 105.0

    # Convert detections to field coordinates and map to minimap
    for det in detections:
        x1, y1, x2, y2 = map(float, det.bbox)
        cx = (x1 + x2) / 2.0
        cy_bottom = y2

        # Convert image position to field coordinates (meters)
        field_pos = pitch_transform.image_to_field([cx, cy_bottom])
        field_x, field_y = field_pos[0], field_pos[1]

        # Normalize using the visible pitch window so video borders do not become pitch borders.
        x_norm = (field_x - window_x_min) / max(1e-6, window_x_max - window_x_min)
        y_norm = field_y / 68.0

        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        y_norm = float(np.clip(y_norm, 0.0, 1.0))

        # Map to minimap pixel coordinates
        px = int(minimap_x + x_norm * minimap_width)
        py = int(minimap_y + y_norm * minimap_height)

        # Clamp to minimap bounds
        px = max(minimap_x + 2, min(w - 17, px))
        py = max(minimap_y + 2, min(h - 17, py))

        # Draw based on object type
        if det.object_type == "ball":
            cv2.circle(frame, (px, py), 4, (0, 220, 255), -1)
            cv2.circle(frame, (px, py), 4, (255, 255, 255), 1)
        elif det.object_type in {"player", "goalkeeper"}:
            color = (
                _normalize_color(det.team_color) if det.team_color else (200, 200, 200)
            )
            radius = 4 if det.object_type == "goalkeeper" else 3
            cv2.circle(frame, (px, py), radius, color, -1)
            cv2.circle(frame, (px, py), radius, (255, 255, 255), 1)
        elif det.object_type == "referee":
            cv2.circle(frame, (px, py), 3, (160, 160, 160), -1)
            cv2.circle(frame, (px, py), 3, (255, 255, 255), 1)
