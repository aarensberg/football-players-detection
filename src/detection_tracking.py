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
        self.ball_roi_model = YOLO(str(model_path))
        self.generic_aliases = {"person": "player", "sports ball": "ball"}
        self.football_aliases = {
            "player": "player",
            "goalkeeper": "goalkeeper",
            "referee": "referee",
            "ball": "ball",
        }
        self.allowed_object_types = {"player", "goalkeeper", "referee", "ball"}
        self.ball_class_ids = self._resolve_ball_class_ids()
        self.canonical_ball_track_id = 1
        self.ball_history: List[Dict[str, object]] = []
        self.ball_history_limit = 6
        self.ball_missed_frames = 0
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

    def _resolve_ball_class_ids(self) -> List[int]:
        names = self.model.names
        ball_ids: List[int] = []
        if isinstance(names, dict):
            items = names.items()
        else:
            items = enumerate(names)
        for class_id, name in items:
            if self._to_object_type(str(name)) == "ball":
                ball_ids.append(int(class_id))
        return ball_ids

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

    @staticmethod
    def _bbox_center_xy(bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=float)

    @staticmethod
    def _bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        x1 = max(float(bbox_a[0]), float(bbox_b[0]))
        y1 = max(float(bbox_a[1]), float(bbox_b[1]))
        x2 = min(float(bbox_a[2]), float(bbox_b[2]))
        y2 = min(float(bbox_a[3]), float(bbox_b[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        area_a = DetectionTracker._bbox_area(bbox_a)
        area_b = DetectionTracker._bbox_area(bbox_b)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union

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
        if (
            object_type in {"player", "goalkeeper"}
            and self.config.player_max_bbox_area_ratio is not None
            and area_px / max(frame_area, 1.0) > self.config.player_max_bbox_area_ratio
        ):
            return False
        return True

    def _predict_ball_center(self, current_frame_idx: int) -> Optional[np.ndarray]:
        if not self.ball_history:
            return None
        last = self.ball_history[-1]
        predicted = np.array(last["center"], dtype=float)
        if len(self.ball_history) >= 2:
            prev = self.ball_history[-2]
            dt = int(last["frame_idx"]) - int(prev["frame_idx"])
            if dt > 0:
                velocity = (np.array(last["center"], dtype=float) - np.array(prev["center"], dtype=float)) / float(dt)
                ahead = max(0, current_frame_idx - int(last["frame_idx"]))
                predicted = predicted + velocity * float(ahead)
        return predicted

    def _update_ball_history(self, frame_idx: int, detection: Detection, status: str) -> None:
        bbox = np.array(detection.bbox, dtype=float)
        self.ball_history.append(
            {
                "frame_idx": int(frame_idx),
                "center": self._bbox_center_xy(bbox).tolist(),
                "bbox": bbox.tolist(),
                "score": float(detection.score),
                "class_id": int(detection.class_id),
                "model_class": str(detection.model_class),
                "status": status,
            }
        )
        if len(self.ball_history) > self.ball_history_limit:
            self.ball_history = self.ball_history[-self.ball_history_limit :]

    def _should_attempt_ball_recovery(self) -> bool:
        if not self.config.ball_roi_recovery_enabled:
            return False
        if not self.ball_history:
            return False
        return self.ball_missed_frames <= self.config.ball_roi_max_missed_frames

    def _roi_bounds_from_prediction(
        self, frame_shape: tuple[int, int, int], predicted_center: np.ndarray
    ) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]
        last_bbox = np.array(self.ball_history[-1]["bbox"], dtype=float)
        last_w = max(2.0, float(last_bbox[2] - last_bbox[0]))
        last_h = max(2.0, float(last_bbox[3] - last_bbox[1]))
        side = max(32.0, max(last_w, last_h) * self.config.ball_roi_window_scale)
        half_side = side / 2.0
        cx = float(predicted_center[0])
        cy = float(predicted_center[1])
        x1 = int(max(0.0, np.floor(cx - half_side)))
        y1 = int(max(0.0, np.floor(cy - half_side)))
        x2 = int(min(float(frame_w), np.ceil(cx + half_side)))
        y2 = int(min(float(frame_h), np.ceil(cy + half_side)))
        if x2 <= x1:
            x2 = min(frame_w, x1 + 1)
        if y2 <= y1:
            y2 = min(frame_h, y1 + 1)
        return x1, y1, x2, y2

    def _attempt_ball_roi_recovery(self, frame: np.ndarray, frame_idx: int) -> Optional[Detection]:
        predicted_center = self._predict_ball_center(frame_idx)
        if predicted_center is None:
            return None
        x1, y1, x2, y2 = self._roi_bounds_from_prediction(frame.shape, predicted_center)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        if not self.ball_class_ids:
            return None

        roi_results = self.ball_roi_model.predict(
            source=roi,
            conf=self.config.ball_roi_conf_threshold,
            iou=self.config.ball_iou_threshold,
            imgsz=max(64, min(self.config.imgsz, max(roi.shape[0], roi.shape[1]))),
            device=self.config.device,
            classes=self.ball_class_ids,
            agnostic_nms=False,
            verbose=False,
        )
        roi_result = roi_results[0]
        if roi_result.boxes is None or len(roi_result.boxes) == 0:
            return None

        boxes = roi_result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        names = roi_result.names if roi_result.names is not None else self.model.names
        frame_area = float(frame.shape[0] * frame.shape[1])

        best_det: Optional[Detection] = None
        best_metric = float("-inf")
        for roi_bbox, class_id, score in zip(xyxy, cls_ids, scores):
            full_bbox = np.array(
                [roi_bbox[0] + x1, roi_bbox[1] + y1, roi_bbox[2] + x1, roi_bbox[3] + y1], dtype=float
            )
            if isinstance(names, dict):
                model_class = str(names.get(int(class_id), str(class_id)))
            elif isinstance(names, list) and int(class_id) < len(names):
                model_class = str(names[int(class_id)])
            else:
                model_class = str(class_id)

            object_type = self._to_object_type(model_class)
            if object_type != "ball":
                continue
            if not self._passes_size_filters("ball", full_bbox, frame_area):
                continue
            center = self._bbox_center_xy(full_bbox)
            dist = float(np.linalg.norm(center - predicted_center))
            max_dist = max(8.0, 0.6 * np.hypot(float(x2 - x1), float(y2 - y1)))
            if dist > max_dist:
                continue

            metric = float(score) - 0.0025 * dist
            if metric <= best_metric:
                continue
            best_metric = metric
            best_det = Detection(
                bbox=full_bbox.astype(float).tolist(),
                class_id=int(class_id),
                model_class=model_class,
                object_type="ball",
                score=float(score),
                track_id=self.canonical_ball_track_id,
            )
        return best_det

    def _select_best_ball_candidate(self, candidates: List[Detection], frame_idx: int) -> Detection:
        if len(candidates) == 1:
            return candidates[0]

        sorted_candidates = sorted(candidates, key=lambda d: d.score, reverse=True)
        filtered: List[Detection] = []
        iou_thr = self.config.ball_iou_threshold
        for cand in sorted_candidates:
            cand_bbox = np.array(cand.bbox, dtype=float)
            if any(self._bbox_iou(cand_bbox, np.array(k.bbox, dtype=float)) > iou_thr for k in filtered):
                continue
            filtered.append(cand)
        if not filtered:
            filtered = sorted_candidates

        predicted_center = self._predict_ball_center(frame_idx)
        if predicted_center is None:
            return filtered[0]

        best_det = filtered[0]
        best_metric = float("-inf")
        for cand in filtered:
            dist = float(np.linalg.norm(self._bbox_center_xy(np.array(cand.bbox, dtype=float)) - predicted_center))
            metric = float(cand.score) - 0.0015 * dist
            if metric > best_metric:
                best_metric = metric
                best_det = cand
        return best_det

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
            agnostic_nms=self.config.agnostic_nms,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            xyxy = np.empty((0, 4), dtype=float)
            cls_ids = np.empty((0,), dtype=int)
            scores = np.empty((0,), dtype=float)
            track_ids = np.empty((0,), dtype=int)
        else:
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
            best_ball = self._select_best_ball_candidate(ball_candidates, frame_idx)
            self.tracks.setdefault("ball", {}).setdefault(frame_idx, {})[self.canonical_ball_track_id] = {
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
            self.ball_missed_frames = 0
            self._update_ball_history(frame_idx, best_ball, status="active")
        else:
            self.ball_missed_frames += 1
            if self._should_attempt_ball_recovery():
                recovered_ball = self._attempt_ball_roi_recovery(frame, frame_idx)
                if recovered_ball is not None:
                    detections.append(recovered_ball)
                    self.tracks.setdefault("ball", {}).setdefault(frame_idx, {})[
                        self.canonical_ball_track_id
                    ] = {
                        "bbox": recovered_ball.bbox,
                        "score": recovered_ball.score,
                        "class_id": recovered_ball.class_id,
                        "model_class": recovered_ball.model_class,
                        "class_name": recovered_ball.class_name,
                        "object_type": recovered_ball.object_type,
                        "raw_track_id": recovered_ball.track_id,
                        "status": "recovered_roi",
                        "interpolated": False,
                    }
                    self.ball_missed_frames = 0
                    self._update_ball_history(frame_idx, recovered_ball, status="recovered_roi")

        return detections
