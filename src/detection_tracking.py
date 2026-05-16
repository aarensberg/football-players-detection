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
    class_name: str
    object_type: str
    score: float
    track_id: int
    team_id: Optional[int] = None
    team_color: Optional[Tuple[int, int, int]] = None

    @property
    def label(self) -> str:
        # Backward-compatible alias used by existing visualization.
        return self.object_type


class DetectionTracker:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        model_path, self.using_fallback_model = self._resolve_model_path()
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
        if self.using_fallback_model:
            print(
                "[INFO] Using generic YOLO model yolov8n.pt; "
                "label aliases active (person->player, sports ball->ball)."
            )
        else:
            print(f"[INFO] Using football-finetuned model: {model_path}")

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

        detections: List[Detection] = []
        for bbox, class_id, score, track_id in zip(xyxy, cls_ids, scores, track_ids):
            if isinstance(names, dict):
                class_name = names.get(int(class_id), str(class_id))
            elif isinstance(names, list) and int(class_id) < len(names):
                class_name = names[int(class_id)]
            else:
                class_name = str(class_id)
            object_type = self._to_object_type(class_name)
            if object_type is None or object_type not in self.allowed_object_types:
                continue
            if (
                object_type == "ball"
                and self.config.ball_conf_threshold is not None
                and float(score) < self.config.ball_conf_threshold
            ):
                continue

            det = Detection(
                bbox=bbox.astype(float).tolist(),
                class_id=int(class_id),
                class_name=str(class_name),
                object_type=object_type,
                score=float(score),
                track_id=int(track_id),
            )
            detections.append(det)

            bucket = self._track_bucket(object_type)
            if bucket and det.track_id >= 0:
                self.tracks.setdefault(bucket, {}).setdefault(frame_idx, {})[det.track_id] = {
                    "bbox": det.bbox,
                    "score": det.score,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "object_type": det.object_type,
                    "status": "active",
                }

        return detections
