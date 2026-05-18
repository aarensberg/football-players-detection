#!/usr/bin/env python3
"""Calculate IoU between track 99 and track 116 detections at frame 138."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection_tracking import DetectionTracker
import numpy as np

# Boxes from the debug output
box_99 = [486.5, 100.3, 546.1, 244.2]  # Approximate from earlier frames
box_116 = [489.1, 101.1, 548.7, 245.9]  # At frame 138, distance 2.9px center

bbox_99 = np.array(box_99, dtype=np.float32)
bbox_116 = np.array(box_116, dtype=np.float32)

iou = DetectionTracker._bbox_iou(bbox_99, bbox_116)
print(f"IoU between id=99 and id=116: {iou:.4f}")
print(f"Dedup threshold: 0.4")
print(f"Should be deduped: {iou > 0.4}")

# More conservative estimates
box_99_estimate = [450, 80, 550, 250]
box_116_estimate = [450, 80, 550, 250]
iou2 = DetectionTracker._bbox_iou(np.array(box_99_estimate), np.array(box_116_estimate))
print(f"\nWith exact same boxes:")
print(f"IoU: {iou2:.4f}")
