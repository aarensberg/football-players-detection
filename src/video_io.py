from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import cv2
import numpy as np


def sanitize_path_component(value: str | Path) -> str:
    component = Path(value).stem if isinstance(value, Path) else Path(str(value)).stem
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", component).strip("._-")
    return component or "unknown"


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = path.suffix
    stem = path.stem
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def build_output_video_path(
    output_dir: Path,
    input_video_path: Path,
    model_path: str | Path,
    extension: str = "mp4",
) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M")
    input_name = sanitize_path_component(input_video_path.stem)
    model_name = sanitize_path_component(model_path)
    ext = extension.lstrip(".").lower() or "mp4"
    candidate = output_dir / f"{input_name}-{model_name}-{timestamp}.{ext}"
    return ensure_unique_path(candidate)


def _video_fourcc_candidates(suffix: str) -> Iterable[str]:
    suffix = suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        # Prefer MP4 on macOS; AVI is only used if all MP4-capable codecs fail.
        return ("mp4v", "avc1")
    return ("XVID", "MJPG")


def _open_writer(path: Path, meta: "VideoMeta") -> cv2.VideoWriter:
    for fourcc_name in _video_fourcc_candidates(path.suffix):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            meta.fps,
            (meta.width, meta.height),
        )
        if writer.isOpened():
            return writer
    raise ValueError(f"Could not open video writer: {path}")


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
        self.output_path = output_path
        try:
            self.writer = _open_writer(output_path, meta)
        except ValueError:
            if output_path.suffix.lower() == ".mp4":
                fallback_path = ensure_unique_path(output_path.with_suffix(".avi"))
                self.writer = _open_writer(fallback_path, meta)
                self.output_path = fallback_path
                print(
                    f"[WARN] MP4 writer unavailable; falling back to AVI: {self.output_path}"
                )
            else:
                raise

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
