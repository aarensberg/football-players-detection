from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from src.config import PipelineConfig


@dataclass
class Detection:
    """Stable detection payload for downstream modules."""

    bbox: List[float]
    class_id: int
    model_class: str
    object_type: str
    score: float
    track_id: int
    team_id: Optional[int] = None
    team_color: Optional[Tuple[int, int, int]] = None

    @property
    def label(self) -> str:
        # Backward-compatible alias used by existing visualization.
        return self.object_type

    @property
    def class_name(self) -> str:
        # Backward-compatible alias for older callers.
        return self.model_class


class DetectionTracker:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        model_path, self.using_fallback_model = self._resolve_model_path()
        self.model_path_used = Path(model_path)
        self.model = YOLO(str(model_path))
        self.generic_aliases = {"person": "player", "sports ball": "ball"}
        self.football_aliases = {
            "player": "player",
            "goalkeeper": "goalkeeper",
            "referee": "referee",
            "ball": "ball",
        }
        self.allowed_object_types = {"player", "goalkeeper", "referee", "ball"}
        self.tracks: Dict[str, Dict[int, Dict[int, Dict[str, object]]]] = {
            "players": {},
            "referees": {},
            "ball": {},
        }
        self.player_goalkeeper_state: Dict[int, Dict[str, object]] = {}
        if self.using_fallback_model:
            print(
                "[INFO] Using generic YOLO model yolov8n.pt; "
                "label aliases active (person->player, sports ball->ball)."
            )
        else:
            print(f"[INFO] Using football-finetuned model: {self.model_path_used}")

    def _resolve_model_path(self) -> tuple[Path | str, bool]:
        if self.config.detector_weights_mode == "football_finetuned":
            if self.config.detector_weights_path.exists():
                return self.config.detector_weights_path, False
            print(
                "[INFO] football_finetuned mode requested but weights not found at "
                f"{self.config.detector_weights_path}. Falling back to yolov8n.pt."
            )
        return "yolov8n.pt", True

    @staticmethod
    def _canonicalize(name: str) -> str:
        return name.strip().lower().replace("_", " ").replace("-", " ")

    def _to_object_type(self, class_name: str) -> str | None:
        normalized = self._canonicalize(class_name)
        if self.using_fallback_model:
            return self.generic_aliases.get(normalized)
        return self.football_aliases.get(
            normalized,
            normalized if normalized in self.allowed_object_types else None,
        )

    @staticmethod
    def _track_bucket(object_type: str) -> str | None:
        name = object_type.lower()
        if name in {"player", "goalkeeper"}:
            return "players"
        if name == "referee":
            return "referees"
        if name == "ball":
            return "ball"
        return None

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        w = max(0.0, float(bbox[2] - bbox[0]))
        h = max(0.0, float(bbox[3] - bbox[1]))
        return w * h

    def _passes_size_filters(
        self, object_type: str, bbox: np.ndarray, frame_area: float
    ) -> bool:
        area_px = self._bbox_area(bbox)
        if area_px <= 0.0:
            return False

        if object_type == "ball":
            min_px = self.config.ball_min_bbox_area_px
            max_px = self.config.ball_max_bbox_area_px
            if min_px is not None and area_px < min_px:
                return False
            if max_px is not None and area_px > max_px:
                return False

            ratio = area_px / max(frame_area, 1.0)
            min_ratio = self.config.ball_min_bbox_area_ratio
            max_ratio = self.config.ball_max_bbox_area_ratio
            if min_ratio is not None and ratio < min_ratio:
                return False
            if max_ratio is not None and ratio > max_ratio:
                return False
            return True

        if self.config.min_bbox_area_px is not None and area_px < self.config.min_bbox_area_px:
            return False
        return True

    def _smooth_object_type(self, track_id: int, object_type: str) -> str:
        if track_id < 0 or object_type not in {"player", "goalkeeper"}:
            return object_type

        required = self.config.goalkeeper_switch_frames
        state = self.player_goalkeeper_state.setdefault(
            track_id,
            {"stable": object_type, "candidate": None, "candidate_count": 0},
        )
        stable = str(state["stable"])
        if object_type == stable:
            state["candidate"] = None
            state["candidate_count"] = 0
            return stable

        if state.get("candidate") == object_type:
            state["candidate_count"] = int(state.get("candidate_count", 0)) + 1
        else:
            state["candidate"] = object_type
            state["candidate_count"] = 1

        if int(state["candidate_count"]) >= required:
            state["stable"] = object_type
            state["candidate"] = None
            state["candidate_count"] = 0
            return object_type
        return stable

    def infer_and_track(self, frame: np.ndarray, frame_idx: int) -> List[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.config.tracker_config,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        track_ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else np.full(len(xyxy), -1, dtype=int)
        )
        names = result.names if result.names is not None else self.model.names
        frame_area = float(frame.shape[0] * frame.shape[1])

        detections: List[Detection] = []
        ball_candidates: List[Detection] = []
        for bbox, class_id, score, track_id in zip(xyxy, cls_ids, scores, track_ids):
            if isinstance(names, dict):
                model_class = names.get(int(class_id), str(class_id))
            elif isinstance(names, list) and int(class_id) < len(names):
                model_class = names[int(class_id)]
            else:
                model_class = str(class_id)
            model_class = str(model_class)
            object_type = self._to_object_type(model_class)
            if object_type is None or object_type not in self.allowed_object_types:
                continue
            if (
                object_type == "ball"
                and self.config.ball_conf_threshold is not None
                and float(score) < self.config.ball_conf_threshold
            ):
                continue
            if not self._passes_size_filters(object_type, bbox, frame_area):
                continue
            object_type = self._smooth_object_type(int(track_id), object_type)

            det = Detection(
                bbox=bbox.astype(float).tolist(),
                class_id=int(class_id),
                model_class=model_class,
                object_type=object_type,
                score=float(score),
                track_id=int(track_id),
            )
            detections.append(det)

            bucket = self._track_bucket(object_type)
            if bucket == "ball":
                ball_candidates.append(det)
                continue

            if bucket and det.track_id >= 0:
                self.tracks.setdefault(bucket, {}).setdefault(frame_idx, {})[det.track_id] = {
                    "bbox": det.bbox,
                    "score": det.score,
                    "class_id": det.class_id,
                    "model_class": det.model_class,
                    "class_name": det.class_name,
                    "object_type": det.object_type,
                    "status": "active",
                    "interpolated": False,
                }

        if ball_candidates:
            # Keep a single canonical ball track to stabilize downstream interpolation.
            best_ball = max(ball_candidates, key=lambda d: d.score)
            self.tracks.setdefault("ball", {}).setdefault(frame_idx, {})[1] = {
                "bbox": best_ball.bbox,
                "score": best_ball.score,
                "class_id": best_ball.class_id,
                "model_class": best_ball.model_class,
                "class_name": best_ball.class_name,
                "object_type": best_ball.object_type,
                "raw_track_id": best_ball.track_id,
                "status": "active",
                "interpolated": False,
            }

        return detections
