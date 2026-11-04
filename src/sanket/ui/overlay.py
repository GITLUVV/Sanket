"""Real-time camera visual feedback and prediction HUD overlay (SANKET-046).

Renders high-contrast, accessible visual HUD elements directly onto video frames:
1. Hand skeletal connections and keypoint landmarks with high-contrast color themes.
2. Active predicted sign gloss banner with clear typography and confidence indicator.
3. Top-k candidate prediction bars for diagnostic inspection.
4. Operational telemetry: FPS, inference latency, tracking status, and rejection state.
5. High efficiency: Designed for < 2.0 ms per-frame render overhead.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from sanket.realtime.engine import InferenceResult
from sanket.realtime.smoothing import SmoothedPrediction
from sanket.realtime.thresholding import CalibratedPrediction
from sanket.vision.landmarks import FrameLandmarks
from sanket.vision.streaming_landmarks import NormalizedFrameLandmarks

logger = logging.getLogger(__name__)

# Standard 21-landmark hand skeletal connections
HAND_SKELETON_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Palm base / Wrist to MCP joints
    (0, 1),
    (0, 5),
    (0, 9),
    (0, 13),
    (0, 17),
    # Thumb
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky finger
    (17, 18),
    (18, 19),
    (19, 20),
    # Palm knuckles
    (5, 9),
    (9, 13),
    (13, 17),
)


class ColorTheme(str, enum.Enum):
    """High-contrast color themes for accessible visualization."""

    HIGH_CONTRAST = "high_contrast"
    CYAN_AMBER = "cyan_amber"
    MONOCHROME = "monochrome"


@dataclass(frozen=True)
class Palette:
    """Color palette specifications (BGR tuples for OpenCV)."""

    left_hand_joint: tuple[int, int, int]
    left_hand_bone: tuple[int, int, int]
    right_hand_joint: tuple[int, int, int]
    right_hand_bone: tuple[int, int, int]
    banner_bg: tuple[int, int, int]
    banner_text: tuple[int, int, int]
    gauge_fill: tuple[int, int, int]
    gauge_bg: tuple[int, int, int]
    telemetry_bg: tuple[int, int, int]
    telemetry_text: tuple[int, int, int]
    candidate_bar: tuple[int, int, int]


THEMES: dict[ColorTheme, Palette] = {
    ColorTheme.HIGH_CONTRAST: Palette(
        left_hand_joint=(0, 255, 255),  # Yellow
        left_hand_bone=(0, 200, 255),
        right_hand_joint=(255, 255, 0),  # Cyan
        right_hand_bone=(255, 200, 0),
        banner_bg=(24, 24, 24),  # Dark Grey
        banner_text=(255, 255, 255),  # Pure White
        gauge_fill=(0, 255, 128),  # Vibrant Green
        gauge_bg=(60, 60, 60),
        telemetry_bg=(20, 20, 20),
        telemetry_text=(220, 220, 220),
        candidate_bar=(255, 165, 0),  # Bright Blue
    ),
    ColorTheme.CYAN_AMBER: Palette(
        left_hand_joint=(0, 215, 255),  # Gold/Amber
        left_hand_bone=(0, 165, 255),
        right_hand_joint=(255, 255, 0),  # Cyan
        right_hand_bone=(200, 200, 0),
        banner_bg=(30, 25, 25),
        banner_text=(250, 250, 250),
        gauge_fill=(0, 220, 100),
        gauge_bg=(50, 50, 50),
        telemetry_bg=(25, 20, 20),
        telemetry_text=(200, 240, 255),
        candidate_bar=(230, 180, 50),
    ),
    ColorTheme.MONOCHROME: Palette(
        left_hand_joint=(255, 255, 255),
        left_hand_bone=(180, 180, 180),
        right_hand_joint=(255, 255, 255),
        right_hand_bone=(180, 180, 180),
        banner_bg=(0, 0, 0),
        banner_text=(255, 255, 255),
        gauge_fill=(255, 255, 255),
        gauge_bg=(80, 80, 80),
        telemetry_bg=(0, 0, 0),
        telemetry_text=(255, 255, 255),
        candidate_bar=(200, 200, 200),
    ),
}


@dataclass
class OverlayConfig:
    """Configuration settings for the camera HUD overlay."""

    show_skeleton: bool = True
    show_gloss_banner: bool = True
    show_candidates: bool = True
    show_telemetry: bool = True
    show_confidence_gauge: bool = True
    theme: ColorTheme = ColorTheme.HIGH_CONTRAST
    banner_alpha: float = 0.65
    top_k_candidates: int = 3
    joint_radius: int = 4
    bone_thickness: int = 2
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6
    font_thickness: int = 1


@dataclass
class TelemetryData:
    """Operational metrics to display on the visual HUD."""

    capture_fps: float = 0.0
    inference_fps: float = 0.0
    latency_ms: float = 0.0
    hands_detected: str = "None"
    dropped_frames: int = 0
    status_message: str = "Active"
    metadata: dict[str, Any] = field(default_factory=dict)


class VisualHUDOverlay:
    """Composes high-performance HUD visual layers directly onto OpenCV BGR image frames."""

    def __init__(self, config: OverlayConfig | None = None) -> None:
        """Initialize VisualHUDOverlay with configuration.

        Args:
            config: Optional OverlayConfig instance.
        """
        self.config = config or OverlayConfig()
        self.palette = THEMES.get(self.config.theme, THEMES[ColorTheme.HIGH_CONTRAST])

    def set_theme(self, theme: ColorTheme | str) -> None:
        """Update color palette theme."""
        if isinstance(theme, str):
            theme = ColorTheme(theme.lower())
        self.config.theme = theme
        self.palette = THEMES.get(theme, THEMES[ColorTheme.HIGH_CONTRAST])

    def render(
        self,
        frame: np.ndarray,
        prediction: InferenceResult | SmoothedPrediction | CalibratedPrediction | None = None,
        landmarks: FrameLandmarks | NormalizedFrameLandmarks | None = None,
        telemetry: TelemetryData | None = None,
    ) -> np.ndarray:
        """Compose HUD elements onto the given frame in-place or on a copy.

        Args:
            frame: Input video frame (BGR format, shape [H, W, 3]).
            prediction: Optional prediction result container.
            landmarks: Optional hand landmark detections.
            telemetry: Optional telemetry metrics.

        Returns:
            Annotated frame with visual HUD elements rendered.
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        height, width = canvas.shape[:2]

        # 1. Render Skeletal Hand Connections
        if self.config.show_skeleton and landmarks is not None:
            self._render_skeletons(canvas, landmarks, width, height)

        # 2. Render Top-Left Telemetry Box
        if self.config.show_telemetry and telemetry is not None:
            self._render_telemetry(canvas, telemetry, width, height)

        # 3. Render Top-Right / Side Top-K Candidate Distribution
        if self.config.show_candidates and prediction is not None:
            self._render_candidates(canvas, prediction, width, height)

        # 4. Render Bottom Active Gloss Banner
        if self.config.show_gloss_banner and prediction is not None:
            self._render_gloss_banner(canvas, prediction, width, height)

        return canvas

    def _render_skeletons(
        self,
        canvas: np.ndarray,
        landmarks: FrameLandmarks | NormalizedFrameLandmarks,
        width: int,
        height: int,
    ) -> None:
        """Draw left and right hand skeletal points and bones."""
        raw_lms = (
            landmarks.raw_landmarks
            if isinstance(landmarks, NormalizedFrameLandmarks)
            else landmarks
        )

        # Helper to draw a single hand
        def draw_hand(
            coords: np.ndarray,
            detected: bool,
            joint_color: tuple[int, int, int],
            bone_color: tuple[int, int, int],
        ) -> None:
            if not detected or coords is None or len(coords) < 21:
                return

            pts_2d: list[tuple[int, int]] = []
            for pt in coords[:21]:
                # Hand landmarks are normalized [0.0, 1.0]
                px = int(np.clip(pt[0] * width, 0, width - 1))
                py = int(np.clip(pt[1] * height, 0, height - 1))
                pts_2d.append((px, py))

            # Draw bones
            for u, v in HAND_SKELETON_CONNECTIONS:
                if u < len(pts_2d) and v < len(pts_2d):
                    cv2.line(
                        canvas,
                        pts_2d[u],
                        pts_2d[v],
                        bone_color,
                        self.config.bone_thickness,
                        lineType=cv2.LINE_AA,
                    )

            # Draw joints
            for px, py in pts_2d:
                cv2.circle(
                    canvas,
                    (px, py),
                    self.config.joint_radius,
                    joint_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        # Left hand
        draw_hand(
            raw_lms.left_hand,
            raw_lms.left_detected,
            self.palette.left_hand_joint,
            self.palette.left_hand_bone,
        )

        # Right hand
        draw_hand(
            raw_lms.right_hand,
            raw_lms.right_detected,
            self.palette.right_hand_joint,
            self.palette.right_hand_bone,
        )

    def _render_telemetry(
        self,
        canvas: np.ndarray,
        telemetry: TelemetryData,
        width: int,
        height: int,
    ) -> None:
        """Render high-contrast operational diagnostics in the top-left corner."""
        lines = [
            f"FPS: {telemetry.capture_fps:.1f} (Inference: {telemetry.inference_fps:.1f})",
            f"Latency: {telemetry.latency_ms:.1f} ms",
            f"Tracking: {telemetry.hands_detected}",
            f"Status: {telemetry.status_message}",
        ]
        if telemetry.dropped_frames > 0:
            lines.append(f"Dropped: {telemetry.dropped_frames}")

        box_w = 260
        line_height = 20
        box_h = 16 + len(lines) * line_height
        x0, y0 = 10, 10

        self._draw_translucent_rect(canvas, x0, y0, box_w, box_h, self.palette.telemetry_bg)

        y_text = y0 + 18
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (x0 + 10, y_text),
                self.config.font_face,
                0.45,
                self.palette.telemetry_text,
                1,
                lineType=cv2.LINE_AA,
            )
            y_text += line_height

    def _render_candidates(
        self,
        canvas: np.ndarray,
        prediction: InferenceResult | SmoothedPrediction | CalibratedPrediction,
        width: int,
        height: int,
    ) -> None:
        """Render top-k class probability bars in the top-right corner."""
        if isinstance(prediction, CalibratedPrediction):
            probs = prediction.probabilities
        elif isinstance(prediction, SmoothedPrediction):
            probs = prediction.smoothed_probabilities
        elif isinstance(prediction, InferenceResult):
            probs = prediction.probabilities
        else:
            return

        if not probs:
            return

        # Sort top-k
        sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[
            : self.config.top_k_candidates
        ]

        box_w = 210
        line_height = 24
        box_h = 16 + len(sorted_probs) * line_height
        x0 = width - box_w - 10
        y0 = 10

        self._draw_translucent_rect(canvas, x0, y0, box_w, box_h, self.palette.telemetry_bg)

        bar_max_w = 70
        y_pos = y0 + 18
        for gloss, p in sorted_probs:
            label = f"{gloss[:10]:<10}"
            cv2.putText(
                canvas,
                label,
                (x0 + 8, y_pos),
                self.config.font_face,
                0.42,
                self.palette.banner_text,
                1,
                lineType=cv2.LINE_AA,
            )

            # Draw bar background and fill
            bar_x = x0 + 80
            bar_y = y_pos - 10
            bar_h = 12
            cv2.rectangle(
                canvas,
                (bar_x, bar_y),
                (bar_x + bar_max_w, bar_y + bar_h),
                self.palette.gauge_bg,
                -1,
            )
            fill_w = int(np.clip(p * bar_max_w, 0, bar_max_w))
            if fill_w > 0:
                cv2.rectangle(
                    canvas,
                    (bar_x, bar_y),
                    (bar_x + fill_w, bar_y + bar_h),
                    self.palette.candidate_bar,
                    -1,
                )

            # Percentage label
            pct_text = f"{p * 100:.0f}%"
            cv2.putText(
                canvas,
                pct_text,
                (bar_x + bar_max_w + 6, y_pos),
                self.config.font_face,
                0.40,
                self.palette.telemetry_text,
                1,
                lineType=cv2.LINE_AA,
            )

            y_pos += line_height

    def _render_gloss_banner(
        self,
        canvas: np.ndarray,
        prediction: InferenceResult | SmoothedPrediction | CalibratedPrediction,
        width: int,
        height: int,
    ) -> None:
        """Render prominent bottom banner displaying the active recognized sign gloss."""
        if isinstance(prediction, CalibratedPrediction):
            gloss = prediction.gloss
            confidence = prediction.confidence
            is_valid = prediction.is_valid_sign
            reason = prediction.rejection_reason
        elif isinstance(prediction, SmoothedPrediction):
            gloss = prediction.active_gloss
            confidence = prediction.confidence
            is_valid = True
            reason = None
        elif isinstance(prediction, InferenceResult):
            gloss = prediction.predicted_gloss
            confidence = prediction.confidence
            is_valid = True
            reason = None
        else:
            return

        banner_h = 60
        y0 = height - banner_h - 12
        x0 = 12
        banner_w = width - 24

        self._draw_translucent_rect(canvas, x0, y0, banner_w, banner_h, self.palette.banner_bg)

        # Format display text
        display_gloss = gloss.upper()
        if not is_valid and reason:
            sub_label = f"({reason.replace('_', ' ')})"
        else:
            sub_label = f"Confidence: {confidence * 100:.1f}%"

        # Main gloss text
        text_size, _ = cv2.getTextSize(display_gloss, self.config.font_face, 1.1, 2)
        text_x = x0 + 20
        text_y = y0 + 38

        # Color highlight: yellow/amber if idle/no_sign, white if valid
        gloss_color = (
            self.palette.banner_text if is_valid else (100, 180, 255)
        )

        cv2.putText(
            canvas,
            display_gloss,
            (text_x, text_y),
            self.config.font_face,
            1.1,
            gloss_color,
            2,
            lineType=cv2.LINE_AA,
        )

        # Secondary label
        cv2.putText(
            canvas,
            sub_label,
            (text_x + text_size[0] + 16, text_y - 4),
            self.config.font_face,
            0.48,
            self.palette.telemetry_text,
            1,
            lineType=cv2.LINE_AA,
        )

        # Confidence gauge bar on the right side of the banner
        if self.config.show_confidence_gauge and is_valid:
            gauge_w = 120
            gauge_h = 16
            gauge_x = x0 + banner_w - gauge_w - 20
            gauge_y = y0 + 22

            # Background gauge
            cv2.rectangle(
                canvas,
                (gauge_x, gauge_y),
                (gauge_x + gauge_w, gauge_y + gauge_h),
                self.palette.gauge_bg,
                -1,
            )
            # Filled gauge
            fill_w = int(np.clip(confidence * gauge_w, 0, gauge_w))
            cv2.rectangle(
                canvas,
                (gauge_x, gauge_y),
                (gauge_x + fill_w, gauge_y + gauge_h),
                self.palette.gauge_fill,
                -1,
            )
            # Gauge outline
            cv2.rectangle(
                canvas,
                (gauge_x, gauge_y),
                (gauge_x + gauge_w, gauge_y + gauge_h),
                (180, 180, 180),
                1,
            )

    def _draw_translucent_rect(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple[int, int, int],
    ) -> None:
        """Fast alpha-blended rectangle for backdrop readability."""
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return

        alpha = self.config.banner_alpha
        sub_img = canvas[y1:y2, x1:x2]
        color_rect = np.full_like(sub_img, color, dtype=np.uint8)
        cv2.addWeighted(color_rect, alpha, sub_img, 1.0 - alpha, 0, sub_img)
