# Football Match Analysis with YOLOv8

This project builds a football video analysis pipeline around YOLOv8, multi-object tracking, team assignment from jersey colors, camera-motion compensation, ball interpolation, and metric extraction. The main goal is to turn broadcast footage into an annotated output video and experiment logs that can be used in the report and slides.

## What the project does

- Detects players, goalkeepers, referees, and the ball in each frame.
- Tracks objects over time with persistent IDs.
- Assigns players to teams using jersey-colour clustering.
- Estimates ball possession from the ball/player geometry.
- Compensates for camera motion with optical flow.
- Interpolates short ball gaps and writes summary metrics to CSV.
- Produces annotated videos in `output/`.

## Repository Layout

- `main.py`: main entry point for a single video.
- `src/`: implementation of the pipeline.
- `scripts/`: experiment runners, analysis helpers, and debug utilities.
- `models/`: saved detector weights.
- `football-players-detection.v1i.yolov8/`: dataset in YOLO format.
- `output/`: generated videos, metrics, and experiment artifacts.
- `report/`: LaTeX report.
- `slides/`: Beamer presentation.

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If you are on macOS and use a Python installation managed by Homebrew, the local virtual environment is the safest option.

## Run the Main Pipeline

The default workflow processes a video and writes an annotated output video to `output/`.

```bash
source .venv/bin/activate
python3 main.py 08fd33_4.mp4
```

You can override the main options exposed by `src/config.py`, for example:

```bash
python3 main.py 08fd33_4.mp4 --max-frames 750 --imgsz 1920 --device mps
```

Useful flags include:

- `--detector-weights-mode generic|football_finetuned`
- `--detector-weights-path models/colab_v3_51e_1920_b2-best.pt`
- `--enable-camera-motion/--no-enable-camera-motion`
- `--enable-ball-interpolation/--no-enable-ball-interpolation`
- `--enable-team-assignment/--no-enable-team-assignment`

## Run Experiments

`scripts/run_experiments.py` runs the main pipeline across several variants and stores each run in `output/experiments/<variant>/`.

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python3 scripts/run_experiments.py \
	--video 08fd33_4.mp4 \
	--variants baseline finetuned_full finetuned_no_interp finetuned_no_cam finetuned_no_team \
	--max-frames 750 \
	--imgsz 1920
```

Each variant produces:

- an annotated video,
- `frame_metrics.csv`,
- `summary_metrics.csv`,
- a screenshot preview.

The summary CSVs are used to update the numbers reported in `report/report.tex` and `slides/slides.tex`.

## Analysis and Debug Scripts

The `scripts/` directory also contains helper scripts for debugging and analysis. The most useful ones are:

- `scripts/analyze_experiments.py`: load the per-variant `summary_metrics.csv` files and print a consolidated comparison.
- `scripts/analyze_consolidation.py`: inspect the consolidated metrics and goalkeeper tracking behaviour.
- `scripts/train_football_detector.py`: manual training entry point for a football-specific YOLOv8 detector.
- `scripts/test_goalkeepers.py`, `scripts/analyze_goalkeeper_stability.py`, `scripts/detailed_goalkeeper_trace.py`: goalkeeper-specific debugging and stability checks.
- `scripts/check_early_tracks.py` and `scripts/trace_first_keeper_debug.py`: track-level debugging for early-frame fragmentation.
- `scripts/debug_run.py` and `scripts/debug_ball_fusion.py`: low-level debugging for the detector and ball fusion logic.

Some of these scripts are exploratory and may reflect earlier debugging workflows; the supported end-to-end path is `main.py` and `scripts/run_experiments.py`.

## Training

If you want to fine-tune a detector on the bundled dataset, use:

```bash
source .venv/bin/activate
python3 scripts/train_football_detector.py \
	--model yolov8n.pt \
	--data football-players-detection.v1i.yolov8/data.yaml \
	--epochs 50 \
	--batch 16 \
	--imgsz 960 \
	--device 0
```

The script saves the best weights to `models/football_yolov8_best.pt` by default.

## Outputs

- `output/<run>.mp4`: annotated video.
- `output/experiments/<variant>/summary_metrics.csv`: aggregated metrics for the report.
- `output/experiments/<variant>/frame_metrics.csv`: per-frame metrics.
- `runs/detect/`: Ultralytics training artifacts.

## Notes

- The pipeline is designed for the provided football clip and dataset, but the configuration can be changed through command-line arguments.
- The report and slides are generated with LaTeX and reflect the latest experiment outputs.
- The project focuses on a modular computer-vision pipeline rather than a single end-to-end model.
