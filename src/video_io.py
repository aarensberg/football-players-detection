from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


class VideoReader:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.meta = VideoMeta(
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        frame_idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self) -> None:
        self.cap.release()


class VideoWriter:
    def __init__(self, output_path: Path, meta: VideoMeta) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(
            str(output_path), fourcc, meta.fps, (meta.width, meta.height)
        )
        if not self.writer.isOpened():
            raise ValueError(f"Could not open video writer: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
