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
        # NOTE ABOUT THE 03_42 "ball-only collapse" bug:
        # A previous version reused the same YOLO instance for the main `track(...)` pass and
        # the ball ROI recovery pass. Mixing tracking and ball-only recovery calls on one model
        # polluted tracker state across frames, which caused non-ball classes to disappear after
        # a short time. Keep a dedicated predict-only model for ball recovery to isolate state.
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
        self.last_frame_debug: Dict[str, object] = {}
        self.frame_geometry: Dict[int, Dict[str, object]] = {}
        self.possession_summary: Dict[str, object] = {}
        # Track memory: keep recently terminated tracks for reactivation
        self.track_memory: Dict[int, Dict[str, object]] = (
            {}
        )  # track_id -> {"frame_idx", "bbox", "object_type", "class_id"}
        self.track_memory_frames = 5  # Keep track memory for 5 frames
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
        return np.array(
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=float
        )

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

        if (
            self.config.min_bbox_area_px is not None
            and area_px < self.config.min_bbox_area_px
        ):
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
                velocity = (
                    np.array(last["center"], dtype=float)
                    - np.array(prev["center"], dtype=float)
                ) / float(dt)
                ahead = max(0, current_frame_idx - int(last["frame_idx"]))
                predicted = predicted + velocity * float(ahead)
        return predicted

    def _update_ball_history(
        self, frame_idx: int, detection: Detection, status: str
    ) -> None:
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

    def _roi_bounds_from_prediction(
        self,
        frame_shape: tuple[int, int, int],
        predicted_center: np.ndarray,
        reference_bbox: Optional[np.ndarray] = None,
    ) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]
        if reference_bbox is not None:
            base_bbox = np.array(reference_bbox, dtype=float)
        elif self.ball_history:
            base_bbox = np.array(self.ball_history[-1]["bbox"], dtype=float)
        else:
            base_bbox = np.array([0.0, 0.0, 16.0, 16.0], dtype=float)
        last_w = max(2.0, float(base_bbox[2] - base_bbox[0]))
        last_h = max(2.0, float(base_bbox[3] - base_bbox[1]))
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

    def _attempt_ball_roi_recovery(
        self,
        frame: np.ndarray,
        frame_idx: int,
        predicted_center: Optional[np.ndarray] = None,
        reference_bbox: Optional[np.ndarray] = None,
    ) -> Optional[Detection]:
        if predicted_center is None:
            predicted_center = self._predict_ball_center(frame_idx)
        if predicted_center is None:
            return None
        x1, y1, x2, y2 = self._roi_bounds_from_prediction(
            frame.shape, predicted_center, reference_bbox=reference_bbox
        )
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        if not self.ball_class_ids:
            return None

        roi_results = self.ball_roi_model.predict(
            source=roi,
            conf=self.config.ball_roi_conf_threshold,
            iou=self.config.ball_iou_threshold,
            imgsz=int(
                np.ceil(
                    max(64, min(self.config.imgsz, max(roi.shape[0], roi.shape[1])))
                    / 32.0
                )
                * 32
            ),
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
                [
                    roi_bbox[0] + x1,
                    roi_bbox[1] + y1,
                    roi_bbox[2] + x1,
                    roi_bbox[3] + y1,
                ],
                dtype=float,
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

    def _select_best_ball_candidate(
        self, candidates: List[Detection], frame_idx: int
    ) -> Detection:
        if len(candidates) == 1:
            return candidates[0]

        sorted_candidates = sorted(candidates, key=lambda d: d.score, reverse=True)
        filtered: List[Detection] = []
        iou_thr = self.config.ball_iou_threshold
        for cand in sorted_candidates:
            cand_bbox = np.array(cand.bbox, dtype=float)
            if any(
                self._bbox_iou(cand_bbox, np.array(k.bbox, dtype=float)) > iou_thr
                for k in filtered
            ):
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
            dist = float(
                np.linalg.norm(
                    self._bbox_center_xy(np.array(cand.bbox, dtype=float))
                    - predicted_center
                )
            )
            metric = float(cand.score) - 0.0015 * dist
            if metric > best_metric:
                best_metric = metric
                best_det = cand
        return best_det

    @staticmethod
    def _count_detections_by_type(detections: List[Detection]) -> Dict[str, int]:
        counts = {"player": 0, "goalkeeper": 0, "referee": 0, "ball": 0}
        for det in detections:
            if det.object_type in counts:
                counts[det.object_type] += 1
        return counts

    def _run_global_tracking_pass(
        self, frame: np.ndarray, frame_idx: int
    ) -> tuple[List[Detection], List[Detection], List[Detection]]:
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

        global_detections: List[Detection] = []
        non_ball_detections: List[Detection] = []
        global_ball_candidates: List[Detection] = []
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
            global_detections.append(det)

            if object_type == "ball":
                global_ball_candidates.append(det)
                continue

            non_ball_detections.append(det)
            bucket = self._track_bucket(object_type)
            if bucket and det.track_id >= 0:
                self.tracks.setdefault(bucket, {}).setdefault(frame_idx, {})[
                    det.track_id
                ] = {
                    "bbox": det.bbox,
                    "score": det.score,
                    "class_id": det.class_id,
                    "model_class": det.model_class,
                    "class_name": det.class_name,
                    "object_type": det.object_type,
                    "status": "active",
                    "interpolated": False,
                }
        return global_detections, non_ball_detections, global_ball_candidates

    def _run_ball_only_full_frame(self, frame: np.ndarray) -> List[Detection]:
        if not self.ball_class_ids:
            return []
        conf = (
            self.config.ball_conf_threshold
            if self.config.ball_conf_threshold is not None
            else self.config.conf_threshold
        )
        results = self.ball_roi_model.predict(
            source=frame,
            conf=conf,
            iou=self.config.ball_iou_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device,
            classes=self.ball_class_ids,
            agnostic_nms=False,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        names = result.names if result.names is not None else self.ball_roi_model.names
        frame_area = float(frame.shape[0] * frame.shape[1])

        candidates: List[Detection] = []
        for bbox, class_id, score in zip(xyxy, cls_ids, scores):
            if isinstance(names, dict):
                model_class = str(names.get(int(class_id), str(class_id)))
            elif isinstance(names, list) and int(class_id) < len(names):
                model_class = str(names[int(class_id)])
            else:
                model_class = str(class_id)
            if self._to_object_type(model_class) != "ball":
                continue
            if not self._passes_size_filters("ball", bbox, frame_area):
                continue
            candidates.append(
                Detection(
                    bbox=bbox.astype(float).tolist(),
                    class_id=int(class_id),
                    model_class=model_class,
                    object_type="ball",
                    score=float(score),
                    track_id=self.canonical_ball_track_id,
                )
            )
        return candidates

    def _run_ball_only_detection(
        self, frame: np.ndarray, frame_idx: int, global_ball_candidates: List[Detection]
    ) -> List[Detection]:
        if not self.config.ball_roi_recovery_enabled:
            return []
        candidates = self._run_ball_only_full_frame(frame)

        predicted_center: Optional[np.ndarray] = None
        reference_bbox: Optional[np.ndarray] = None
        if global_ball_candidates:
            best_global = self._select_best_ball_candidate(
                global_ball_candidates, frame_idx
            )
            predicted_center = self._bbox_center_xy(
                np.array(best_global.bbox, dtype=float)
            )
            reference_bbox = np.array(best_global.bbox, dtype=float)
        else:
            predicted_center = self._predict_ball_center(frame_idx)
            if self.ball_history:
                reference_bbox = np.array(self.ball_history[-1]["bbox"], dtype=float)

        if predicted_center is not None and (
            self.ball_missed_frames <= self.config.ball_roi_max_missed_frames
            or global_ball_candidates
        ):
            roi_candidate = self._attempt_ball_roi_recovery(
                frame,
                frame_idx,
                predicted_center=predicted_center,
                reference_bbox=reference_bbox,
            )
            if roi_candidate is not None:
                candidates.append(roi_candidate)
        return candidates

    def _smooth_object_type(self, track_id: int, object_type: str) -> str:
        if track_id < 0 or object_type not in {"player", "goalkeeper"}:
            return object_type

        # Use a sliding window with majority vote instead of consecutive frames.
        # This is more robust to detector oscillations.
        # CRITICAL: Detector oscillates every 1-3 frames, so window must be MUCH larger
        # to filter out these oscillations. Using 15 frames instead of 5-7.
        window_size = 15  # Increased from max(5, config.goalkeeper_switch_frames + 2)
        state = self.player_goalkeeper_state.setdefault(
            track_id,
            {"history": [], "stable": object_type},
        )
        history: list = state.get("history", [])
        history.append(object_type)

        # Keep only the last window_size observations
        if len(history) > window_size:
            history = history[-window_size:]
        state["history"] = history

        stable = state.get("stable", object_type)

        # Count votes for each class
        goalkeeper_count = sum(1 for x in history if x == "goalkeeper")
        player_count = sum(1 for x in history if x == "player")

        # Majority threshold: need more than half the votes
        threshold = (len(history) + 1) // 2

        # Accept the new type if it has majority vote
        if goalkeeper_count >= threshold:
            state["stable"] = "goalkeeper"
            return "goalkeeper"
        elif player_count >= threshold:
            state["stable"] = "player"
            return "player"
        else:
            # No consensus yet, return current stable
            return stable

    def _try_reactivate_from_track_memory(
        self, detections: List[Detection], frame_idx: int
    ) -> Dict[int, int]:
        """Try to reactivate recently terminated tracks if nearby detections appear.
        Returns a mapping of (new_track_id -> old_track_id) for relabeled detections.
        """
        remap = {}
        frame_cutoff = frame_idx - self.track_memory_frames
        memory_to_remove = [
            tid
            for tid, info in self.track_memory.items()
            if info.get("frame_idx", 0) <= frame_cutoff
        ]
        for tid in memory_to_remove:
            del self.track_memory[tid]

        # For each detection (even newly created ones from ByteTrack), try to match with memory
        # This allows reactivation of fragmented tracks (e.g., track 99 -> track 116)
        for det in detections:
            if det.object_type == "ball":
                continue

            det_bbox = np.array(det.bbox, dtype=float)
            best_match = None
            best_iou = 0.0
            for mem_id, mem_info in self.track_memory.items():
                mem_bbox = np.array(mem_info["bbox"], dtype=float)
                iou = self._bbox_iou(det_bbox, mem_bbox)
                # Reactivate if IoU > 0.2 and same/compatible object type
                if iou > 0.2 and iou > best_iou:
                    mem_type = mem_info.get("object_type", "player")
                    if mem_type == det.object_type or (
                        mem_type in {"player", "goalkeeper"}
                        and det.object_type in {"player", "goalkeeper"}
                    ):
                        best_match = mem_id
                        best_iou = iou

            if best_match is not None:
                # Reactivate the old track_id
                old_id = det.track_id
                det.track_id = best_match
                remap[old_id] = best_match
                del self.track_memory[best_match]

        return remap

    def infer_and_track(self, frame: np.ndarray, frame_idx: int) -> List[Detection]:
        global_detections, non_ball_detections, global_ball_candidates = (
            self._run_global_tracking_pass(frame, frame_idx)
        )
        ball_only_candidates = self._run_ball_only_detection(
            frame, frame_idx, global_ball_candidates
        )

        fused_candidates = [*global_ball_candidates, *ball_only_candidates]
        fused_ball: Optional[Detection] = None
        fused_status = "missing"
        selected_from_ball_only = False
        if fused_candidates:
            fused_ball = self._select_best_ball_candidate(fused_candidates, frame_idx)
            selected_from_ball_only = fused_ball in ball_only_candidates
            if selected_from_ball_only and global_ball_candidates:
                fused_status = "fused_ball_only"
            elif selected_from_ball_only:
                fused_status = "recovered_ball_only"
            else:
                fused_status = "active"

        detections = list(non_ball_detections)
        if fused_ball is not None:
            final_ball = Detection(
                bbox=fused_ball.bbox,
                class_id=fused_ball.class_id,
                model_class=fused_ball.model_class,
                object_type="ball",
                score=fused_ball.score,
                track_id=self.canonical_ball_track_id,
            )
            detections.append(final_ball)
            self.tracks.setdefault("ball", {}).setdefault(frame_idx, {})[
                self.canonical_ball_track_id
            ] = {
                "bbox": final_ball.bbox,
                "score": final_ball.score,
                "class_id": final_ball.class_id,
                "model_class": final_ball.model_class,
                "class_name": final_ball.class_name,
                "object_type": final_ball.object_type,
                "raw_track_id": fused_ball.track_id,
                "status": fused_status,
                "interpolated": False,
            }
            self.ball_missed_frames = 0
            self._update_ball_history(frame_idx, final_ball, status=fused_status)
        else:
            self.ball_missed_frames += 1

        self.last_frame_debug = {
            "frame_idx": int(frame_idx),
            "global_counts": self._count_detections_by_type(global_detections),
            "fused_counts": self._count_detections_by_type(detections),
            "ball_candidates_global": len(global_ball_candidates),
            "ball_candidates_ball_only": len(ball_only_candidates),
            "ball_source": (
                "ball_only"
                if selected_from_ball_only and fused_ball is not None
                else "global" if fused_ball is not None else "none"
            ),
        }
        return detections
