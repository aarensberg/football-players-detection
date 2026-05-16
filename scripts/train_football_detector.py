"""
Manual training entrypoint for a football-specific YOLOv8 detector.

IMPORTANT:
- This script is NOT run automatically by Copilot, imports, main.py, or pipeline code.
- Run it manually on a GPU machine when you explicitly want to train.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on football player detection data")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLOv8 model or weights")
    parser.add_argument(
        "--data",
        type=str,
        default="football-players-detection.v1i.yolov8/data.yaml",
        help="Dataset YAML path",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Device for training (e.g. "0", "0,1", "cpu", "mps")',
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="football_yolov8_finetune",
        help="Run name inside output project directory",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Output project directory used by Ultralytics",
    )
    parser.add_argument(
        "--save-best-to",
        type=str,
        default="models/football_yolov8_best.pt",
        help="Deterministic destination for best weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.run_name,
        exist_ok=True,
    )

    best_from_run = Path(args.project) / args.run_name / "weights" / "best.pt"
    if not best_from_run.exists():
        raise FileNotFoundError(f"Training completed but best weights were not found at: {best_from_run}")

    target = Path(args.save_best_to)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_from_run, target)
    print(f"Saved best weights to: {target}")


if __name__ == "__main__":
    main()
