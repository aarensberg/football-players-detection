"""
Consolidate and analyze metrics from all experiment variants.

Reads summary_metrics.csv from each variant and builds a unified table
for easy comparison and reporting.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class VariantMetrics:
    variant: str
    # Ball metrics
    ball_frames_detected: int
    ball_frames_observed: int
    ball_frames_interpolated: int
    ball_max_interp_gap: int
    # Player metrics
    player_frames_detected: int
    player_avg_detections_per_frame: float
    player_total_tracks: int
    player_avg_track_length: float
    player_max_track_length: int
    # Goalkeeper metrics
    goalkeeper_frames_detected: int
    goalkeeper_total_tracks: int
    goalkeeper_avg_track_length: float
    # Referee metrics
    referee_frames_detected: int
    referee_total_tracks: int
    # Team metrics
    team_avg_switches_per_track: Optional[float]
    team_max_switches_per_track: Optional[int]
    team_balance_team1: Optional[int]
    team_balance_team2: Optional[int]
    # Speed metrics
    player_unrealistic_speed_events: int

    @property
    def ball_coverage_percentage(self, total_frames: int = 750) -> float:
        """Ball visible as percentage of total frames."""
        return (self.ball_frames_detected / total_frames) * 100.0

    @property
    def player_track_fragmentation(self) -> float:
        """Ratio of tracks to average track length (higher = more fragmented)."""
        if self.player_avg_track_length > 0:
            return self.player_total_tracks / self.player_avg_track_length
        return 0.0


def load_variant_metrics(variant_dir: Path) -> VariantMetrics:
    csv_path = variant_dir / "summary_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No metrics CSV found at {csv_path}")

    metrics_dict: Dict[str, str] = {}
    with csv_path.open("r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            metrics_dict[row["metric"]] = row["value"]

    def get_float(key: str) -> float:
        val = metrics_dict.get(key, "0")
        return float(val) if val else 0.0

    def get_int(key: str) -> int:
        val = metrics_dict.get(key, "0")
        return int(float(val)) if val else 0

    def get_optional_float(key: str) -> Optional[float]:
        val = metrics_dict.get(key)
        if val and val.strip():
            try:
                return float(val)
            except ValueError:
                return None
        return None

    def get_optional_int(key: str) -> Optional[int]:
        val = metrics_dict.get(key)
        if val and val.strip():
            try:
                return int(float(val))
            except ValueError:
                return None
        return None

    variant_name = metrics_dict.get("variant", variant_dir.name)
    return VariantMetrics(
        variant=variant_name,
        ball_frames_detected=get_int("ball_frames_with_detection"),
        ball_frames_observed=get_int("ball_frames_with_observed"),
        ball_frames_interpolated=get_int("ball_frames_with_interpolated"),
        ball_max_interp_gap=get_int("ball_max_interp_gap"),
        player_frames_detected=get_int("player_frames_with_detection"),
        player_avg_detections_per_frame=get_float("player_avg_detections_per_frame"),
        player_total_tracks=get_int("player_total_tracks"),
        player_avg_track_length=get_float("player_avg_track_length_frames"),
        player_max_track_length=get_int("player_max_track_length_frames"),
        goalkeeper_frames_detected=get_int("goalkeeper_frames_with_detection"),
        goalkeeper_total_tracks=get_int("goalkeeper_total_tracks"),
        goalkeeper_avg_track_length=get_float("goalkeeper_avg_track_length_frames"),
        referee_frames_detected=get_int("referee_frames_with_detection"),
        referee_total_tracks=get_int("referee_total_tracks"),
        team_avg_switches_per_track=get_optional_float("team_avg_switches_per_track"),
        team_max_switches_per_track=get_optional_int("team_max_switches_per_track"),
        team_balance_team1=get_optional_int("team_balance_team1_tracks"),
        team_balance_team2=get_optional_int("team_balance_team2_tracks"),
        player_unrealistic_speed_events=get_int("player_unrealistic_speed_events"),
    )


def main() -> None:
    exp_dir = Path("output/experiments")
    variant_names = [
        "baseline",
        "finetuned_full",
        "finetuned_no_interp",
        "finetuned_no_cam",
        "finetuned_no_team",
    ]

    variants: List[VariantMetrics] = []
    for vname in variant_names:
        vdir = exp_dir / vname
        if vdir.exists():
            try:
                vm = load_variant_metrics(vdir)
                variants.append(vm)
                print(f"✓ Loaded {vname}")
            except Exception as e:
                print(f"✗ Failed to load {vname}: {e}")
        else:
            print(f"✗ Directory not found: {vdir}")

    if not variants:
        print("No variants loaded.")
        return

    total_frames = 750
    print("\n" + "=" * 120)
    print("CONSOLIDATED METRICS TABLE")
    print("=" * 120)
    print(
        f"{'Variant':<25} | {'Ball Cov %':<12} | {'Ball Interp':<12} | "
        f"{'Player Trks':<12} | {'Avg Trk Len':<12} | {'Team Sw Avg':<12} | {'Unrealistic Spd':<15}"
    )
    print("-" * 120)
    for vm in variants:
        ball_cov = (vm.ball_frames_detected / total_frames) * 100.0
        team_sw = (
            f"{vm.team_avg_switches_per_track:.2f}"
            if vm.team_avg_switches_per_track is not None
            else "N/A"
        )
        print(
            f"{vm.variant:<25} | {ball_cov:>10.1f}% | {vm.ball_frames_interpolated:>10} | "
            f"{vm.player_total_tracks:>10} | {vm.player_avg_track_length:>10.1f} | {team_sw:>10} | {vm.player_unrealistic_speed_events:>13}"
        )

    print("\n" + "=" * 120)
    print("KEY METRICS COMPARISON: FINE-TUNED FULL vs BASELINE")
    print("=" * 120)
    baseline = next((v for v in variants if v.variant == "baseline"), None)
    finetuned_full = next((v for v in variants if v.variant == "finetuned_full"), None)

    if baseline and finetuned_full:
        ball_cov_bl = (baseline.ball_frames_detected / total_frames) * 100.0
        ball_cov_ft = (finetuned_full.ball_frames_detected / total_frames) * 100.0
        print(
            f"Ball coverage: {ball_cov_bl:.1f}% → {ball_cov_ft:.1f}% (+{ball_cov_ft - ball_cov_bl:.1f}pp)"
        )
        print(
            f"Player tracks: {baseline.player_total_tracks} → {finetuned_full.player_total_tracks} ({baseline.player_total_tracks / finetuned_full.player_total_tracks:.1f}x more consolidated)"
        )
        print(
            f"Avg track length: {baseline.player_avg_track_length:.1f} → {finetuned_full.player_avg_track_length:.1f} frames (+{finetuned_full.player_avg_track_length - baseline.player_avg_track_length:.1f})"
        )
        print(
            f"Referee detection: {baseline.referee_frames_detected} → {finetuned_full.referee_frames_detected} frames"
        )

    print("\n" + "=" * 120)
    print("ABLATION ANALYSIS")
    print("=" * 120)
    if finetuned_full:
        no_interp = next(
            (v for v in variants if v.variant == "finetuned_no_interp"), None
        )
        no_cam = next((v for v in variants if v.variant == "finetuned_no_cam"), None)
        no_team = next((v for v in variants if v.variant == "finetuned_no_team"), None)

        if no_interp:
            print(f"Ball Interpolation Impact:")
            print(
                f"  - WITH interpolation: {finetuned_full.ball_frames_detected} frames ({finetuned_full.ball_frames_interpolated} interpolated)"
            )
            print(f"  - WITHOUT interpolation: {no_interp.ball_frames_detected} frames")
            print(
                f"  - Difference: {finetuned_full.ball_frames_detected - no_interp.ball_frames_detected} frames gained from interpolation"
            )

        if no_cam:
            print(f"\nCamera Motion Compensation Impact:")
            print(
                f"  - WITH compensation: ball {finetuned_full.ball_frames_detected} frames, team switches {finetuned_full.team_avg_switches_per_track:.2f}"
            )
            print(
                f"  - WITHOUT compensation: ball {no_cam.ball_frames_detected} frames, team switches {no_cam.team_avg_switches_per_track:.2f}"
            )

        if no_team:
            print(f"\nTeam Assignment Impact:")
            print(
                f"  - WITH team assignment: avg switches {finetuned_full.team_avg_switches_per_track:.2f}, max {finetuned_full.team_max_switches_per_track}"
            )
            print(
                f"  - WITHOUT team assignment: team metrics N/A (assignment disabled)"
            )

    print("\n" + "=" * 120)
    print("DETAILED METRICS BY VARIANT")
    print("=" * 120)
    for vm in variants:
        print(f"\n{vm.variant}:")
        print(f"  Ball:")
        print(
            f"    - Frames with detection: {vm.ball_frames_detected} / {total_frames} ({(vm.ball_frames_detected/total_frames)*100:.1f}%)"
        )
        print(
            f"    - Observed: {vm.ball_frames_observed}, Interpolated: {vm.ball_frames_interpolated}, Max gap: {vm.ball_max_interp_gap}"
        )
        print(f"  Players:")
        print(
            f"    - Frames with detection: {vm.player_frames_detected} / {total_frames}"
        )
        print(
            f"    - Total tracks: {vm.player_total_tracks}, Avg length: {vm.player_avg_track_length:.1f}, Max: {vm.player_max_track_length}"
        )
        print(f"  Goalkeepers:")
        print(
            f"    - Frames with detection: {vm.goalkeeper_frames_detected}, Total tracks: {vm.goalkeeper_total_tracks}"
        )
        print(f"  Referees:")
        print(
            f"    - Frames with detection: {vm.referee_frames_detected}, Total tracks: {vm.referee_total_tracks}"
        )
        if vm.team_avg_switches_per_track is not None:
            print(f"  Team Assignment:")
            print(
                f"    - Avg switches per track: {vm.team_avg_switches_per_track:.2f}, Max: {vm.team_max_switches_per_track}"
            )
            print(
                f"    - Balance: Team1={vm.team_balance_team1} tracks, Team2={vm.team_balance_team2} tracks"
            )
        else:
            print(f"  Team Assignment: Disabled")


if __name__ == "__main__":
    main()
