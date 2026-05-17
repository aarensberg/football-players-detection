from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SummaryMetrics:
    variant: str
    metrics: Dict[str, object]


class MetricsCollector:
    def __init__(self, fps: float, meters_per_pixel: Optional[float] = None) -> None:
        self.fps = float(fps)
        self.mpp = None if meters_per_pixel is None else float(meters_per_pixel)
        self.frames: List[int] = []

    def record_frame(self, frame_idx: int, detections: List[object]) -> None:
        self.frames.append(int(frame_idx))

    @staticmethod
    def _center(bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(float, bbox)
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    @staticmethod
    def _bottom_center(bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(float, bbox)
        return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)

    @staticmethod
    def _as_point(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 2:
            return None
        return arr

    def _frame_geometry(self, detector, frame_idx: int) -> Dict[str, Any]:
        return getattr(detector, "frame_geometry", {}).get(int(frame_idx), {})

    def _point_for_sample(
        self,
        detector,
        bucket_name: str,
        frame_idx: int,
        track_id: int,
        attrs: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
        geom = self._frame_geometry(detector, frame_idx)
        team_id = attrs.get("team_id")
        point_px = self._center(attrs.get("bbox", [0.0, 0.0, 0.0, 0.0]))
        point_m = point_px.copy()

        if bucket_name == "ball":
            ball_point = geom.get("ball_point", {})
            ball_px = self._as_point(ball_point.get("stabilized_px"))
            ball_m = self._as_point(ball_point.get("field_m"))
            if ball_px is not None:
                point_px = ball_px
            if ball_m is not None:
                point_m = ball_m
        elif bucket_name == "players":
            player_like = geom.get("player_like_points", {})
            entry = player_like.get(int(track_id), {})
            player_px = self._as_point(entry.get("stabilized_px"))
            player_m = self._as_point(entry.get("field_m"))
            if player_px is not None:
                point_px = player_px
            else:
                point_px = self._bottom_center(attrs.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            if player_m is not None:
                point_m = player_m
            else:
                point_m = point_px.copy()
            team_id = entry.get("team_id", team_id)
        elif bucket_name == "referees":
            point_px = self._bottom_center(attrs.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            point_m = point_px.copy()

        if np.array_equal(point_m, point_px) and self.mpp is not None:
            point_m = point_px * float(self.mpp)

        return point_m.astype(np.float32), point_px.astype(np.float32), team_id

    def _track_samples(
        self,
        detector,
        bucket_name: str,
        object_type: Optional[str] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        bucket = detector.tracks.get(bucket_name, {})
        per_track: Dict[int, List[Dict[str, Any]]] = {}
        for frame_idx, frame_tracks in bucket.items():
            for track_id, attrs in frame_tracks.items():
                attrs_type = str(attrs.get("object_type", "")).lower()
                if object_type is not None and attrs_type != object_type:
                    continue
                point_m, point_px, team_id = self._point_for_sample(
                    detector,
                    bucket_name,
                    int(frame_idx),
                    int(track_id),
                    attrs,
                )
                per_track.setdefault(int(track_id), []).append(
                    {
                        "frame_idx": int(frame_idx),
                        "point_m": point_m,
                        "point_px": point_px,
                        "team_id": team_id,
                        "object_type": attrs_type,
                        "interpolated": bool(attrs.get("interpolated", False)),
                    }
                )
        for samples in per_track.values():
            samples.sort(key=lambda sample: sample["frame_idx"])
        return per_track

    def _speed_distance_stats(
        self,
        per_track: Dict[int, List[Dict[str, Any]]],
        speed_threshold_kmh: float,
    ) -> Tuple[Dict[str, Any], Dict[int, List[float]]]:
        step_speeds: List[float] = []
        max_speeds: List[float] = []
        total_distance = 0.0
        unrealistic_events = 0
        frame_speed_map: Dict[int, List[float]] = defaultdict(list)

        for samples in per_track.values():
            if len(samples) < 2:
                continue
            prev = samples[0]
            track_speeds: List[float] = []
            for sample in samples[1:]:
                frame_gap = int(sample["frame_idx"]) - int(prev["frame_idx"])
                if frame_gap <= 0:
                    prev = sample
                    continue
                dt = frame_gap / self.fps
                if dt <= 0:
                    prev = sample
                    continue
                dist_m = float(np.linalg.norm(sample["point_m"] - prev["point_m"]))
                speed_kmh = (dist_m / dt) * 3.6
                step_speeds.append(speed_kmh)
                track_speeds.append(speed_kmh)
                total_distance += dist_m
                frame_speed_map[int(sample["frame_idx"])].append(speed_kmh)
                if speed_kmh > speed_threshold_kmh:
                    unrealistic_events += 1
                prev = sample
            if track_speeds:
                max_speeds.append(max(track_speeds))

        if max_speeds:
            arr = np.asarray(max_speeds, dtype=np.float32)
            stats = {
                "min": float(np.min(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
                "mean_of_track_max": float(np.mean(arr)),
            }
        else:
            stats = {
                "min": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "max": 0.0,
                "mean_of_track_max": 0.0,
            }

        if step_speeds:
            speeds_arr = np.asarray(step_speeds, dtype=np.float32)
            stats.update(
                {
                    "mean_step_speed": float(np.mean(speeds_arr)),
                    "median_step_speed": float(np.median(speeds_arr)),
                    "distance_total_m": float(total_distance),
                }
            )
        else:
            stats.update(
                {
                    "mean_step_speed": 0.0,
                    "median_step_speed": 0.0,
                    "distance_total_m": 0.0,
                }
            )

        stats["unrealistic_events"] = unrealistic_events
        return stats, frame_speed_map

    def finalize(self, detector, out_dir: Path, variant_name: str) -> SummaryMetrics:
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.frames:
            frames_set = set()
            for bucket in detector.tracks.values():
                for frame_idx in bucket.keys():
                    frames_set.add(int(frame_idx))
            for frame_idx in getattr(detector, "frame_geometry", {}).keys():
                frames_set.add(int(frame_idx))
            self.frames = sorted(frames_set)

        player_samples = self._track_samples(detector, "players", object_type="player")
        goalkeeper_samples = self._track_samples(
            detector, "players", object_type="goalkeeper"
        )
        referee_samples = self._track_samples(
            detector, "referees", object_type="referee"
        )
        ball_samples = self._track_samples(detector, "ball", object_type="ball")

        player_speed_stats, player_frame_speed_map = self._speed_distance_stats(
            player_samples, 40.0
        )
        goalkeeper_speed_stats, goalkeeper_frame_speed_map = self._speed_distance_stats(
            goalkeeper_samples, 35.0
        )

        frame_csv = out_dir / "frame_metrics.csv"
        with frame_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "frame_idx",
                    "count_ball",
                    "count_player",
                    "count_goalkeeper",
                    "count_referee",
                    "mean_player_speed_kmh",
                    "mean_goalkeeper_speed_kmh",
                    "ball_observed",
                    "ball_interpolated",
                    "possession_team_id",
                    "camera_dx_px",
                    "camera_dy_px",
                ]
            )

            for frame_idx in sorted(self.frames):
                counts = {"ball": 0, "player": 0, "goalkeeper": 0, "referee": 0}
                counts["ball"] = sum(
                    1
                    for samples in ball_samples.values()
                    for item in samples
                    if item["frame_idx"] == frame_idx
                )
                counts["player"] = sum(
                    1
                    for samples in player_samples.values()
                    for item in samples
                    if item["frame_idx"] == frame_idx
                )
                counts["goalkeeper"] = sum(
                    1
                    for samples in goalkeeper_samples.values()
                    for item in samples
                    if item["frame_idx"] == frame_idx
                )
                counts["referee"] = sum(
                    1
                    for samples in referee_samples.values()
                    for item in samples
                    if item["frame_idx"] == frame_idx
                )

                geom = self._frame_geometry(detector, frame_idx)
                possession = geom.get("possession", {})
                camera_motion = geom.get("camera_motion", {})
                ball_obs = any(
                    item["frame_idx"] == frame_idx and not item["interpolated"]
                    for samples in ball_samples.values()
                    for item in samples
                )
                ball_interp = any(
                    item["frame_idx"] == frame_idx and item["interpolated"]
                    for samples in ball_samples.values()
                    for item in samples
                )
                mean_player_speed = (
                    float(np.mean(player_frame_speed_map[frame_idx]))
                    if frame_idx in player_frame_speed_map
                    else float("nan")
                )
                mean_goalkeeper_speed = (
                    float(np.mean(goalkeeper_frame_speed_map[frame_idx]))
                    if frame_idx in goalkeeper_frame_speed_map
                    else float("nan")
                )

                writer.writerow(
                    [
                        frame_idx,
                        counts["ball"],
                        counts["player"],
                        counts["goalkeeper"],
                        counts["referee"],
                        mean_player_speed,
                        mean_goalkeeper_speed,
                        int(bool(ball_obs)),
                        int(bool(ball_interp)),
                        possession.get("team_id"),
                        camera_motion.get("dx_px", 0.0),
                        camera_motion.get("dy_px", 0.0),
                    ]
                )

        def class_summary(
            samples_by_track: Dict[int, List[Dict[str, Any]]],
        ) -> Tuple[int, int, int, float, int]:
            frame_ids = set()
            total_dets = 0
            track_lengths: List[int] = []
            for samples in samples_by_track.values():
                if samples:
                    track_lengths.append(len(samples))
                for sample in samples:
                    frame_ids.add(int(sample["frame_idx"]))
                    total_dets += 1
            return (
                len(frame_ids),
                total_dets,
                len(samples_by_track),
                float(np.mean(track_lengths)) if track_lengths else 0.0,
                int(max(track_lengths)) if track_lengths else 0,
            )

        (
            ball_frames_with_detection,
            ball_total_dets,
            ball_total_tracks,
            ball_avg_len,
            ball_max_len,
        ) = class_summary(ball_samples)
        (
            player_frames_with_detection,
            player_total_dets,
            player_total_tracks,
            player_avg_len,
            player_max_len,
        ) = class_summary(player_samples)
        (
            goalkeeper_frames_with_detection,
            goalkeeper_total_dets,
            goalkeeper_total_tracks,
            goalkeeper_avg_len,
            goalkeeper_max_len,
        ) = class_summary(goalkeeper_samples)
        (
            referee_frames_with_detection,
            referee_total_dets,
            referee_total_tracks,
            referee_avg_len,
            referee_max_len,
        ) = class_summary(referee_samples)

        observed = 0
        interpolated = 0
        max_interp_gap = 0
        for samples in ball_samples.values():
            current_gap = 0
            for sample in samples:
                if sample["interpolated"]:
                    interpolated += 1
                    current_gap += 1
                    max_interp_gap = max(max_interp_gap, current_gap)
                else:
                    observed += 1
                    current_gap = 0

        summary: Dict[str, object] = {
            "variant": variant_name,
            "ball_frames_with_detection": ball_frames_with_detection,
            "ball_avg_detections_per_frame": (
                (ball_total_dets / ball_frames_with_detection)
                if ball_frames_with_detection
                else 0.0
            ),
            "ball_total_tracks": ball_total_tracks,
            "ball_avg_track_length_frames": ball_avg_len,
            "ball_max_track_length_frames": ball_max_len,
            "ball_frames_with_observed": observed,
            "ball_frames_with_interpolated": interpolated,
            "ball_max_interp_gap": max_interp_gap,
            "ball_coverage_pct": 100.0
            * ball_frames_with_detection
            / max(1, len(self.frames)),
            "player_frames_with_detection": player_frames_with_detection,
            "player_avg_detections_per_frame": (
                (player_total_dets / player_frames_with_detection)
                if player_frames_with_detection
                else 0.0
            ),
            "player_total_tracks": player_total_tracks,
            "player_avg_track_length_frames": player_avg_len,
            "player_max_track_length_frames": player_max_len,
            "goalkeeper_frames_with_detection": goalkeeper_frames_with_detection,
            "goalkeeper_avg_detections_per_frame": (
                (goalkeeper_total_dets / goalkeeper_frames_with_detection)
                if goalkeeper_frames_with_detection
                else 0.0
            ),
            "goalkeeper_total_tracks": goalkeeper_total_tracks,
            "goalkeeper_avg_track_length_frames": goalkeeper_avg_len,
            "goalkeeper_max_track_length_frames": goalkeeper_max_len,
            "referee_frames_with_detection": referee_frames_with_detection,
            "referee_avg_detections_per_frame": (
                (referee_total_dets / referee_frames_with_detection)
                if referee_frames_with_detection
                else 0.0
            ),
            "referee_total_tracks": referee_total_tracks,
            "referee_avg_track_length_frames": referee_avg_len,
            "referee_max_track_length_frames": referee_max_len,
            "player_mean_speed_kmh": player_speed_stats["mean_step_speed"],
            "player_median_speed_kmh": player_speed_stats["median_step_speed"],
            "player_max_speed_kmh": player_speed_stats["max"],
            "player_speed_p25_kmh": player_speed_stats["p25"],
            "player_speed_p75_kmh": player_speed_stats["p75"],
            "player_distance_total_m": player_speed_stats["distance_total_m"],
            "player_unrealistic_speed_events": player_speed_stats["unrealistic_events"],
            "goalkeeper_mean_speed_kmh": goalkeeper_speed_stats["mean_step_speed"],
            "goalkeeper_median_speed_kmh": goalkeeper_speed_stats["median_step_speed"],
            "goalkeeper_max_speed_kmh": goalkeeper_speed_stats["max"],
            "goalkeeper_speed_p25_kmh": goalkeeper_speed_stats["p25"],
            "goalkeeper_speed_p75_kmh": goalkeeper_speed_stats["p75"],
            "goalkeeper_distance_total_m": goalkeeper_speed_stats["distance_total_m"],
            "goalkeeper_unrealistic_speed_events": goalkeeper_speed_stats[
                "unrealistic_events"
            ],
        }

        possession_summary = getattr(detector, "possession_summary", {}) or {}
        if not possession_summary:
            possession_frames = 0
            team1_frames = 0
            team2_frames = 0
            unknown_frames = 0
            for frame_idx in self.frames:
                poss = self._frame_geometry(detector, frame_idx).get("possession", {})
                if poss:
                    possession_frames += 1
                    team_id = poss.get("team_id")
                    if team_id == 1:
                        team1_frames += 1
                    elif team_id == 2:
                        team2_frames += 1
                    else:
                        unknown_frames += 1
            possession_summary = {
                "total_visible_frames": possession_frames,
                "team1_frames": team1_frames,
                "team2_frames": team2_frames,
                "unknown_frames": unknown_frames,
                "team1_pct": 100.0 * team1_frames / max(1, possession_frames),
                "team2_pct": 100.0 * team2_frames / max(1, possession_frames),
                "unknown_pct": 100.0 * unknown_frames / max(1, possession_frames),
            }
        summary.update(
            {
                "possession_total_visible_frames": possession_summary.get(
                    "total_visible_frames", 0
                ),
                "possession_team1_frames": possession_summary.get("team1_frames", 0),
                "possession_team2_frames": possession_summary.get("team2_frames", 0),
                "possession_unknown_frames": possession_summary.get(
                    "unknown_frames", 0
                ),
                "possession_team1_pct": possession_summary.get("team1_pct", 0.0),
                "possession_team2_pct": possession_summary.get("team2_pct", 0.0),
                "possession_unknown_pct": possession_summary.get("unknown_pct", 0.0),
            }
        )

        player_team_counts = {1: 0, 2: 0}
        team_switch_counts: List[int] = []
        for samples in player_samples.values():
            teams = [
                sample.get("team_id")
                for sample in samples
                if sample.get("team_id") in {1, 2}
            ]
            if not teams:
                continue
            counts = {1: teams.count(1), 2: teams.count(2)}
            if counts[1] >= counts[2]:
                player_team_counts[1] += 1
            else:
                player_team_counts[2] += 1

            last_team = None
            switches = 0
            for team in teams:
                if last_team is None:
                    last_team = team
                    continue
                if team != last_team:
                    switches += 1
                    last_team = team
            team_switch_counts.append(switches)

        summary.update(
            {
                "team_avg_switches_per_track": (
                    float(np.mean(team_switch_counts)) if team_switch_counts else 0.0
                ),
                "team_max_switches_per_track": (
                    int(max(team_switch_counts)) if team_switch_counts else 0
                ),
                "team_balance_team1_tracks": player_team_counts[1],
                "team_balance_team2_tracks": player_team_counts[2],
            }
        )

        summary_csv = out_dir / "summary_metrics.csv"
        with summary_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value"])
            for key, value in summary.items():
                writer.writerow([key, value])

        return SummaryMetrics(variant=variant_name, metrics=summary)
