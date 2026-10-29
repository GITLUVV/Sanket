"""Standardized image and video preprocessing pipeline for Sanket.

Provides uniform temporal frame subsampling, aspect-ratio-preserving
letterbox resizing, coordinate unletterboxing, and batch video decoding.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable metadata descriptor for a video file."""

    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Extract metadata from a video file using OpenCV.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata containing frame count, fps, dimensions, and duration.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video cannot be opened or decoded.
    """
    path_obj = Path(video_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Unable to open or decode video file: {video_path}")

    try:
        raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Fallback for streams or formats with missing header metadata
        if raw_count <= 0:
            count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
            frame_count = count
        else:
            frame_count = raw_count

        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Video file has no readable frames: {video_path}")

        duration = frame_count / fps if fps > 0.0 else 0.0

        return VideoMetadata(
            path=str(path_obj),
            frame_count=frame_count,
            fps=max(fps, 0.0),
            width=width,
            height=height,
            duration_seconds=duration,
        )
    finally:
        cap.release()


def compute_temporal_indices(total_frames: int, target_frames: int = 30) -> list[int]:
    """Compute uniformly spaced temporal frame indices using linear interpolation.

    For videos where total_frames >= target_frames, downsamples frames evenly.
    For videos where total_frames < target_frames, upsamples by repeating frames
    evenly across the temporal sequence.

    Args:
        total_frames: Total number of frames in the source video (>= 1).
        target_frames: Desired uniform output length (default 30).

    Returns:
        A list of integer indices of length target_frames, monotonic non-decreasing,
        with index 0 mapped to the first frame and target_frames - 1 mapped to the last frame.

    Raises:
        ValueError: If total_frames < 1 or target_frames < 1.
    """
    if total_frames < 1:
        raise ValueError(f"total_frames must be >= 1, got {total_frames}")
    if target_frames < 1:
        raise ValueError(f"target_frames must be >= 1, got {target_frames}")

    if total_frames == 1:
        return [0] * target_frames

    # Compute evenly spaced indices across [0, total_frames - 1]
    raw_indices = np.linspace(0, total_frames - 1, num=target_frames)
    rounded = np.round(raw_indices).astype(int)

    # Ensure bounds safety
    clamped = np.clip(rounded, 0, total_frames - 1)
    return clamped.tolist()


def letterbox_resize(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 480),
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resize an image preserving aspect ratio with symmetric letterbox padding.

    Args:
        image: Source image array of shape (H, W, C) or (H, W).
        target_size: Desired output (width, height) tuple. Default (640, 480).
        pad_color: RGB/BGR tuple for padding borders. Default black (0, 0, 0).

    Returns:
        A tuple of (padded_image, transform_metadata).
        padded_image has shape (target_height, target_width, C) or (target_height, target_width).
        transform_metadata contains scale, padding offsets, and original shape.

    Raises:
        ValueError: If image is empty or invalid.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot resize empty or None image array")

    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size

    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Target size must be positive, got {target_size}")

    # Compute aspect-preserving scale factor
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # Use INTER_AREA when downscaling, INTER_LINEAR when upscaling
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Calculate symmetric padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    metadata: dict[str, Any] = {
        "scale": float(scale),
        "pad_left": int(pad_left),
        "pad_right": int(pad_right),
        "pad_top": int(pad_top),
        "pad_bottom": int(pad_bottom),
        "original_shape": (int(orig_h), int(orig_w)),
        "resized_shape": (int(new_h), int(new_w)),
        "target_size": (int(target_w), int(target_h)),
    }

    return padded, metadata


def unletterbox_coordinates(
    x: float,
    y: float,
    metadata: dict[str, Any],
) -> tuple[float, float]:
    """Invert letterbox transformation to map coordinates back to the original image space.

    Args:
        x: X-coordinate in the letterboxed image space (pixels).
        y: Y-coordinate in the letterboxed image space (pixels).
        metadata: Transformation dictionary returned by `letterbox_resize`.

    Returns:
        Tuple of (orig_x, orig_y) in original image pixel coordinates.
    """
    scale = metadata["scale"]
    pad_left = metadata["pad_left"]
    pad_top = metadata["pad_top"]
    orig_h, orig_w = metadata["original_shape"]

    orig_x = (x - pad_left) / scale
    orig_y = (y - pad_top) / scale

    # Clamp to valid image bounds
    clamped_x = max(0.0, min(float(orig_w - 1), orig_x))
    clamped_y = max(0.0, min(float(orig_h - 1), orig_y))

    return clamped_x, clamped_y


def read_video_frames(
    video_path: str | Path,
    target_indices: list[int] | None = None,
    to_rgb: bool = True,
) -> list[np.ndarray]:
    """Read specific or all frames from a video file.

    Args:
        video_path: Path to the video file.
        target_indices: Optional list of frame indices to extract. If None,
            reads all sequential frames.
        to_rgb: If True, converts OpenCV BGR frames to RGB format.

    Returns:
        List of frame arrays (H, W, 3) in uint8 format.

    Raises:
        FileNotFoundError: If the video file is missing.
        ValueError: If video decoding fails or frame indices cannot be satisfied.
    """
    path_obj = Path(video_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frames: list[np.ndarray] = []
    try:
        if target_indices is None:
            # Sequential extraction of all frames
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            if not frames:
                raise ValueError(f"No readable frames found in {video_path}")
            return frames

        # Extract targeted frame indices
        unique_indices = sorted(set(target_indices))
        cache: dict[int, np.ndarray] = {}
        curr_pos = 0

        for idx in unique_indices:
            if curr_pos != idx:
                # Use grab() for short forward skips, set() for seeks
                if 0 < (idx - curr_pos) < 5:
                    while curr_pos < idx:
                        cap.grab()
                        curr_pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    curr_pos = idx

            ret, frame = cap.read()
            curr_pos += 1

            if not ret or frame is None:
                # If EOF was reached near edge, replicate last successfully decoded frame
                if cache:
                    last_valid = list(cache.values())[-1]
                    frame = last_valid.copy()
                else:
                    raise ValueError(f"Failed to read initial frame {idx} from {video_path}")
            elif to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cache[idx] = frame

        # Construct final frame sequence preserving target_indices ordering and duplicates
        frames = [cache[idx].copy() for idx in target_indices]
        return frames
    finally:
        cap.release()


def sample_video_frames(
    video_path: str | Path,
    target_frames: int = 30,
    target_size: tuple[int, int] | None = (640, 480),
    letterbox: bool = True,
    to_rgb: bool = True,
) -> np.ndarray:
    """Preprocess a video clip into a fixed-length normalized tensor.

    Applies uniform temporal subsampling and optional spatial resizing.

    Args:
        video_path: Path to the video file.
        target_frames: Target sequence length T (default 30).
        target_size: Target (width, height) for spatial resizing. If None,
            original resolution is preserved.
        letterbox: If True, uses aspect-ratio preserving letterboxing. If False,
            uses direct linear resizing.
        to_rgb: If True, converts BGR to RGB format.

    Returns:
        Numpy array of shape (T, H, W, 3) and dtype uint8.
    """
    meta = get_video_metadata(video_path)
    indices = compute_temporal_indices(meta.frame_count, target_frames=target_frames)
    raw_frames = read_video_frames(video_path, target_indices=indices, to_rgb=to_rgb)

    processed: list[np.ndarray] = []
    for f in raw_frames:
        if target_size is not None:
            if letterbox:
                resized_f, _ = letterbox_resize(f, target_size=target_size)
            else:
                resized_f = cv2.resize(f, target_size, interpolation=cv2.INTER_LINEAR)
            processed.append(resized_f)
        else:
            processed.append(f)

    return np.stack(processed, axis=0)


class VideoPreprocessor:
    """Configurable video preprocessing engine for model training and inference."""

    def __init__(
        self,
        target_frames: int = 30,
        target_size: tuple[int, int] = (640, 480),
        letterbox: bool = True,
        to_rgb: bool = True,
        normalize_pixels: bool = False,
    ) -> None:
        """Initialize the video preprocessor.

        Args:
            target_frames: Fixed sequence length T per clip (default 30).
            target_size: Fixed spatial resolution (width, height). Default (640, 480).
            letterbox: Whether to preserve aspect ratio via letterboxing. Default True.
            to_rgb: Whether to convert OpenCV BGR to RGB. Default True.
            normalize_pixels: If True, casts to float32 and scales pixel intensities
                to [0.0, 1.0]. Default False.
        """
        self.target_frames = target_frames
        self.target_size = target_size
        self.letterbox = letterbox
        self.to_rgb = to_rgb
        self.normalize_pixels = normalize_pixels

    def preprocess(self, video_path: str | Path) -> np.ndarray:
        """Preprocess a video file into a standardized tensor.

        Args:
            video_path: Path to the video file.

        Returns:
            Numpy array of shape (T, H, W, 3). If normalize_pixels is True,
            dtype is float32 with values in [0.0, 1.0]. Otherwise, uint8 [0, 255].
        """
        tensor = sample_video_frames(
            video_path=video_path,
            target_frames=self.target_frames,
            target_size=self.target_size,
            letterbox=self.letterbox,
            to_rgb=self.to_rgb,
        )

        if self.normalize_pixels:
            return tensor.astype(np.float32) / 255.0
        return tensor

    def preprocess_clip(self, frames: list[np.ndarray]) -> np.ndarray:
        """Preprocess an in-memory list of raw video frames.

        Useful for live camera feeds or memory-buffered clip segments.

        Args:
            frames: Non-empty list of numpy frame arrays.

        Returns:
            Numpy array of shape (T, H, W, 3).
        """
        if not frames:
            raise ValueError("Cannot preprocess empty frames list")

        n_frames = len(frames)
        indices = compute_temporal_indices(n_frames, target_frames=self.target_frames)
        sampled_frames = [frames[i] for i in indices]

        processed: list[np.ndarray] = []
        for f in sampled_frames:
            cur = f
            if self.to_rgb and cur.ndim == 3 and cur.shape[2] == 3:
                # If input frames are BGR, convert to RGB
                cur = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)

            if self.target_size is not None:
                if self.letterbox:
                    cur, _ = letterbox_resize(cur, target_size=self.target_size)
                else:
                    cur = cv2.resize(cur, self.target_size, interpolation=cv2.INTER_LINEAR)
            processed.append(cur)

        tensor = np.stack(processed, axis=0)
        if self.normalize_pixels:
            return tensor.astype(np.float32) / 255.0
        return tensor

    def save_preprocessed(self, video_path: str | Path, output_path: str | Path) -> Path:
        """Preprocess a video and serialize the resulting tensor to a .npy file.

        Args:
            video_path: Source video path.
            output_path: Destination .npy path.

        Returns:
            Path object to the created .npy file.
        """
        tensor = self.preprocess(video_path)
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(dest), tensor)
        logger.info("Saved preprocessed tensor of shape %s to %s", tensor.shape, dest)
        return dest
