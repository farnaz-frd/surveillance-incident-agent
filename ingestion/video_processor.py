"""
Video ingestion: takes a raw video file (simulating a camera buffer dump)
and samples it down to a small set of representative frames on disk.

Why sample instead of feeding every frame to the VLM: a 30fps, 60-second
clip is 1800 frames. Running a vision model on every frame is wasteful and,
on constrained hardware, simply not viable. Fixed-interval sampling is the
right default for a surveillance context — anomalies (a person entering a
restricted zone, an abandoned bag) persist across multiple seconds, so we
don't need frame-level granularity to catch them. This is a real design
trade-off worth calling out in your README: sampling rate vs. detection
latency vs. compute cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Raised when a video file can't be opened or read."""


@dataclass
class SampledFrame:
    frame_path: Path
    frame_index: int
    timestamp_s: float
    camera_id: str


class VideoProcessor:
    def __init__(self, output_dir: Path, sample_rate_s: float = 2.0, max_frames: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate_s = sample_rate_s
        self.max_frames = max_frames

    def sample_frames(self, video_path: str | Path, camera_id: str) -> list[SampledFrame]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"OpenCV could not open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                raise VideoProcessingError(f"Could not determine FPS for: {video_path}")

            frame_interval = max(1, int(round(fps * self.sample_rate_s)))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            sampled: list[SampledFrame] = []
            frame_idx = 0

            while cap.isOpened() and len(sampled) < self.max_frames:
                # Seeking by index is more reliable across codecs than
                # reading sequentially and discarding — avoids drift on
                # variable-frame-rate files.
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_s = frame_idx / fps
                out_path = self.output_dir / f"{camera_id}_frame{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)

                sampled.append(
                    SampledFrame(
                        frame_path=out_path,
                        frame_index=frame_idx,
                        timestamp_s=timestamp_s,
                        camera_id=camera_id,
                    )
                )

                frame_idx += frame_interval
                if frame_idx >= total_frames:
                    break

            if not sampled:
                raise VideoProcessingError(f"No frames could be extracted from: {video_path}")

            logger.info("Sampled %d frames from %s", len(sampled), video_path.name)
            return sampled

        finally:
            cap.release()


# if __name__ == "__main__":
#     # Quick smoke test — point this at a sample clip once you have one in
#     # data/sample_videos/. This block is what you'd run manually to sanity
#     # check the module in isolation, before wiring it into main.py.
#     logging.basicConfig(level=logging.INFO)
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python video_processor.py <path_to_video>")
#         sys.exit(1)

#     processor = VideoProcessor(output_dir=Path("./.frame_cache"))
#     frames = processor.sample_frames(sys.argv[1], camera_id="CAM_01")
#     for f in frames:
#         print(f)

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     processor = VideoProcessor(output_dir=Path("./.frame_cache"))

#     video_path = Path("./data/sample_videos/test.mp4") #./data/sample_videos/synthetic_test.mp4

#     frames = processor.sample_frames(
#         video_path=video_path,
#         camera_id="CAM_01"
#     )

#     for frame in frames:
#         print(frame)
