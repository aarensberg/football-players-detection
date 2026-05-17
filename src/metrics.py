from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import math

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
        self.frame_counts: Dict[int, Dict[str, int]] = {}

    def record_frame(self, frame_idx: int, detections: List[object]) -> None:
        counts: Dict[str, int] = {"ball": 0, "player": 0, "goalkeeper": 0, "referee": 0}
        for d in detections:
            t = getattr(d, "object_type", None) or getattr(d, "model_class", None)
            if t in counts:
                counts[t] += 1
        self.frames.append(int(frame_idx))
        self.frame_counts[int(frame_idx)] = counts

    def _build_per_track(
        self, tracks: Dict[int, Dict[int, Dict[str, object]]]
    ) -> Dict[int, List[Tuple[int, Dict[str, object]]]]:
        per_track: Dict[int, List[Tuple[int, Dict[str, object]]]] = {}
        for frame_idx, frame_tracks in tracks.items():
            for track_id, attrs in frame_tracks.items():
                per_track.setdefault(int(track_id), []).append((int(frame_idx), attrs))
        for tid, lst in per_track.items():
            lst.sort(key=lambda x: x[0])
        return per_track

    def finalize(self, detector, out_dir: Path, variant_name: str) -> SummaryMetrics:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Build frame index set from tracks if none were recorded per-frame
        if not self.frames:
            frames_set = set()
            for bucket in detector.tracks.values():
                for f in bucket.keys():
                    frames_set.add(int(f))
            self.frames = sorted(frames_set)

        # Frame-level CSV
        frame_csv = out_dir / "frame_metrics.csv"
        with frame_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            header = [
                "frame_idx",
                "count_ball",
                "count_player",
                "count_goalkeeper",
                "count_referee",
                "mean_player_speed_kmh",
                "ball_observed",
                "ball_interpolated",
            ]
            writer.writerow(header)
            for f in sorted(self.frames):
                # compute counts from detector.tracks
                counts = {"ball": 0, "player": 0, "goalkeeper": 0, "referee": 0}
                for key, bucket in detector.tracks.items():
                    frame_tracks = bucket.get(f, {})
                    if key == "players":
                        # players bucket contains players & goalkeepers with object_type stored in attrs
                        for _, attrs in frame_tracks.items():
                            ot = attrs.get("object_type") or attrs.get("model_class")
                            if ot == "player":
                                counts["player"] += 1
                            elif ot == "goalkeeper":
                                counts["goalkeeper"] += 1
                    elif key == "referees":
                        counts["referee"] += len(frame_tracks)
                    elif key == "ball":
                        counts["ball"] += len(frame_tracks)
                # speeds computed per-track below; put placeholder NaN here
                ball_obs = any(
                    not attrs.get("interpolated")
                    for attrs in detector.tracks.get("ball", {}).get(f, {}).values()
                )
                ball_interp = any(
                    attrs.get("interpolated")
                    for attrs in detector.tracks.get("ball", {}).get(f, {}).values()
                )
                writer.writerow(
                    [
                        f,
                        counts["ball"],
                        counts["player"],
                        counts["goalkeeper"],
                        counts["referee"],
                        float("nan"),
                        int(bool(ball_obs)),
                        int(bool(ball_interp)),
                    ]
                )

        # Summary metrics
        summary: Dict[str, object] = {}
        summary["variant"] = variant_name
        # detection & tracking per class
        for cls in ("ball", "player", "goalkeeper", "referee"):
            frames_with = 0
            total_dets = 0
            track_ids = set()
            track_lengths: List[int] = []
            tracks_bucket = (
                detector.tracks.get(
                    "players" if cls in {"player", "goalkeeper"} else cls, {}
                )
                if cls != "referee"
                else detector.tracks.get("referees", {})
            )
            # tracks_bucket is mapping frame-> {track_id: attrs}
            for frame_idx, frame_tracks in tracks_bucket.items():
                if frame_tracks:
                    frames_with += 1
                    total_dets += len(frame_tracks)
                    for tid in frame_tracks.keys():
                        track_ids.add(int(tid))
            per_track = self._build_per_track(tracks_bucket)
            for tid, lst in per_track.items():
                track_lengths.append(len(lst))
            summary[f"{cls}_frames_with_detection"] = frames_with
            summary[f"{cls}_avg_detections_per_frame"] = (
                (total_dets / frames_with) if frames_with > 0 else 0.0
            )
            summary[f"{cls}_total_tracks"] = len(track_ids)
            summary[f"{cls}_avg_track_length_frames"] = (
                float(np.mean(track_lengths)) if track_lengths else 0.0
            )
            summary[f"{cls}_max_track_length_frames"] = (
                int(max(track_lengths)) if track_lengths else 0
            )

        # Ball-specific metrics
        ball_tracks = detector.tracks.get("ball", {})
        observed = 0
        interpolated = 0
        max_interp_gap = 0
        per_ball = self._build_per_track(ball_tracks)
        for tid, lst in per_ball.items():
            # lst is list of (frame_idx, attrs)
            gaps = 0
            cur_gap = 0
            for _, attrs in lst:
                if attrs.get("interpolated"):
                    interpolated += 1
                    cur_gap += 1
                    max_interp_gap = max(max_interp_gap, cur_gap)
                else:
                    observed += 1
                    cur_gap = 0
        summary["ball_frames_with_observed"] = observed
        summary["ball_frames_with_interpolated"] = interpolated
        summary["ball_max_interp_gap"] = max_interp_gap

        # Player/goalkeeper speed stats (requires meters_per_pixel)
        speed_threshold_kmh = 40.0
        player_max_speeds: List[float] = []
        unrealistic_events = 0
        for bucket_name in ("players",):
            tracks_bucket = detector.tracks.get(bucket_name, {})
            per = self._build_per_track(tracks_bucket)
            for tid, lst in per.items():
                speeds_kmh: List[float] = []
                prev_center = None
                prev_frame = None
                for frame_idx, attrs in lst:
                    bbox = attrs.get("bbox")
                    if bbox is None:
                        continue
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    if prev_center is not None and prev_frame is not None:
                        dt = (frame_idx - prev_frame) / self.fps
                        if dt > 0:
                            dist_px = math.hypot(
                                cx - prev_center[0], cy - prev_center[1]
                            )
                            if self.mpp is not None:
                                speed_kmh = (dist_px / dt) * self.mpp * 3.6
                                speeds_kmh.append(speed_kmh)
                                if speed_kmh > speed_threshold_kmh:
                                    unrealistic_events += 1
                    prev_center = (cx, cy)
                    prev_frame = frame_idx
                if speeds_kmh:
                    player_max_speeds.append(max(speeds_kmh))

        if player_max_speeds:
            arr = np.array(player_max_speeds)
            summary["player_max_speed_stats_kmh"] = {
                "min": float(np.min(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
            }
        else:
            summary["player_max_speed_stats_kmh"] = None
        summary["player_unrealistic_speed_events"] = unrealistic_events

        # Team assignment stability
        players_bucket = detector.tracks.get("players", {})
        per_players = self._build_per_track(players_bucket)
        team_switch_counts: List[int] = []
        for tid, lst in per_players.items():
            last_team = None
            switches = 0
            for _, attrs in lst:
                team = attrs.get("team_id")
                if last_team is None:
                    last_team = team
                else:
                    if team is not None and last_team is not None and team != last_team:
                        switches += 1
                        last_team = team
                    elif team is not None:
                        last_team = team
            team_switch_counts.append(switches)
        summary["team_avg_switches_per_track"] = (
            float(np.mean(team_switch_counts)) if team_switch_counts else 0.0
        )
        summary["team_max_switches_per_track"] = (
            int(max(team_switch_counts)) if team_switch_counts else 0
        )
        # team balance
        team1 = 0
        team2 = 0
        for tid, lst in per_players.items():
            # majority assignment
            teams = [
                attrs.get("team_id")
                for _, attrs in lst
                if attrs.get("team_id") is not None
            ]
            if teams:
                counts = {1: teams.count(1), 2: teams.count(2)}
                if counts[1] >= counts[2]:
                    team1 += 1
                else:
                    team2 += 1
        summary["team_balance_team1_tracks"] = team1
        summary["team_balance_team2_tracks"] = team2

        # write summary CSV
        summary_csv = out_dir / "summary_metrics.csv"
        with summary_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value"])
            for k, v in summary.items():
                writer.writerow([k, v])

        return SummaryMetrics(variant=variant_name, metrics=summary)
