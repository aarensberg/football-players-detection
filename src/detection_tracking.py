from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from src.config import PipelineConfig


@dataclass
class Detection:
    bbox: List[float]
    class_id: int
    label: str
    score: float
    track_id: int


class DetectionTracker:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        model_path, self.using_fallback_model = self._resolve_model_path()
        self.model = YOLO(str(model_path))
        self.effective_conf_threshold = self.config.conf_threshold
        self.label_aliases = {
            "person": "player",
            "sports ball": "ball",
        }
        self.tracks: Dict[str, Dict[int, Dict[int, Dict[str, object]]]] = {
            "players": {},
            "referees": {},
            "ball": {},
        }
        if self.using_fallback_model:
            print(
                "[INFO] Using fallback model yolov8n.pt; "
                "label normalization active (person->player, sports ball->ball)."
            )
            if not self.config.conf_explicitly_set and self.effective_conf_threshold > 0.1:
                original_conf = self.effective_conf_threshold
                self.effective_conf_threshold = 0.1
                print(
                    "[INFO] Fallback confidence adjustment active: "
                    f"effective_conf={self.effective_conf_threshold:.2f} "
                    f"(default was {original_conf:.2f}). "
                    "Use --conf to override."
                )

    def _resolve_model_path(self) -> tuple[Path | str, bool]:
        if self.config.model_path is not None:
            if not self.config.model_path.exists():
                raise FileNotFoundError(f"Model path not found: {self.config.model_path}")
            return self.config.model_path, False

        custom = Path("models/best.pt")
        if custom.exists():
            return custom, False

        return "yolov8n.pt", True

    def _normalize_label(self, label: str) -> str:
        name = label.strip().lower()
        return self.label_aliases.get(name, name)

    @staticmethod
    def _track_bucket(label: str) -> str | None:
        name = label.lower()
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
            conf=self.effective_conf_threshold,
            iou=self.config.iou_threshold,
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
            raw_label = names[int(class_id)] if isinstance(names, dict) else str(class_id)
            label = self._normalize_label(raw_label)
            det = Detection(
                bbox=bbox.astype(float).tolist(),
                class_id=int(class_id),
                label=label,
                score=float(score),
                track_id=int(track_id),
            )
            detections.append(det)

            bucket = self._track_bucket(label)
            if bucket and det.track_id >= 0:
                self.tracks.setdefault(bucket, {}).setdefault(frame_idx, {})[det.track_id] = {
                    "bbox": det.bbox,
                    "score": det.score,
                    "class_id": det.class_id,
                    "label": det.label,
                    "status": "active",
                }

        return detections
