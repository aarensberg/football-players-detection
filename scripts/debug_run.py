from src.config import parse_args
from src.pipeline import run_pipeline

cfg = parse_args()
output_path, detector = run_pipeline(cfg, return_detector=True)
print("Frames with geometry:", len(detector.frame_geometry))
for idx in sorted(detector.frame_geometry.keys())[:10]:
    geom = detector.frame_geometry[idx]
    cam = geom.get("camera_motion", {})
    print(idx, cam.get("dx_px"), cam.get("dy_px"), cam.get("num_points"))
print("\nInspecting goalkeeper appearances in early frames...")
players = detector.tracks.get("players", {})
goalkeeper_found = False
for frame_idx in sorted(players.keys())[:200]:
    frame_tracks = players.get(frame_idx, {})
    for tid, attrs in frame_tracks.items():
        if str(attrs.get("object_type", "")).lower() == "goalkeeper":
            print(
                f'Frame {frame_idx}: track {tid} object_type=goalkeeper team_id={attrs.get("team_id")}'
            )
            goalkeeper_found = True
            break
        model_cls = str(attrs.get("model_class", "")).lower()
        if "goalkeeper" in model_cls:
            print(
                f"Frame {frame_idx}: track {tid} model_class indicates goalkeeper -> model_class={attrs.get('model_class')} team_id={attrs.get('team_id')}"
            )
            goalkeeper_found = True
            break
    if goalkeeper_found:
        break
if not goalkeeper_found:
    print("No goalkeeper object_type found in first 200 frames of player tracks")
print("Done")
