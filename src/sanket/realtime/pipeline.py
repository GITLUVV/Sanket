"""End-to-end real-time Indian Sign Language recognition pipeline (SANKET-048).

Integrates all Phase 2 subsystem components into a cohesive streaming pipeline:
1. Video Capture: Asynchronous camera capture or headless mock video.
2. Preprocessing: Real-time smoothing, contrast enhancement, mirroring, resizing.
3. Feature Extraction: MediaPipe streaming landmarks and palm normalization.
4. Inference: Low-latency sign classification with pre-warmed baseline model.
5. Temporal Smoothing: Exponential Moving Average and hysteresis filtering.
6. Gating & Calibration: Dynamic confidence cutoff and resting-hand detection.
7. User Interface: Real-time camera HUD overlay and accessible subtitles.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sanket.realtime.engine import InferenceResult, RealtimeInferenceEngine
from sanket.realtime.smoothing import SmoothedPrediction, TemporalPredictionSmoother
from sanket.realtime.thresholding import (
    CalibratedPrediction,
    ConfidenceCalibrator,
    SensitivityProfile,
)
from sanket.ui.overlay import OverlayConfig, TelemetryData, VisualHUDOverlay
from sanket.ui.text_display import AccessibleTextDisplay, TextDisplayConfig
from sanket.vision.capture import BaseVideoCapture, CameraCapture, MockVideoCapture
from sanket.vision.preprocessing import RealtimePreprocessor
from sanket.vision.streaming_landmarks import NormalizedFrameLandmarks, StreamingHandLandmarker

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Master configuration for the end-to-end real-time recognition pipeline."""

    # Video Source
    camera_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True
    use_mock: bool = False
    mock_pattern: str = "moving_box"
    mock_file: str | Path | None = None

    # Model & Inference
    model_path: str | Path | None = None
    label_map_path: str | Path | None = None
    min_confidence: float = 0.65
    sensitivity_profile: SensitivityProfile | str = SensitivityProfile.BALANCED

    # Smoothing & Hysteresis
    hysteresis_frames: int = 3
    smoothing_alpha: float = 0.35
    min_dwell_ms: float = 200.0

    # Display & HUD
    show_hud: bool = True
    show_skeleton: bool = True
    show_candidates: bool = True
    headless: bool = False
    window_name: str = "SANKET - Real-Time ISL MVP"


@dataclass
class PipelineTelemetry:
    """Real-time operational statistics for pipeline monitoring."""

    processed_frames: int = 0
    dropped_frames: int = 0
    capture_fps: float = 0.0
    pipeline_fps: float = 0.0
    total_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    active_gloss: str = "IDLE"
    confidence: float = 0.0
    hands_detected: str = "None"
    metadata: dict[str, Any] = field(default_factory=dict)


class RealtimePipeline:
    """End-to-end real-time sign recognition pipeline orchestrator."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        model: Any | None = None,
        label_map: dict[int, str] | None = None,
    ) -> None:
        """Initialize RealtimePipeline and its constituent subsystems.

        Args:
            config: Pipeline configuration options.
            model: Optional pre-loaded classifier model instance.
            label_map: Optional class ID to sign gloss mapping dictionary.
        """
        self.config = config or PipelineConfig()

        # 1. Video Capture subsystem
        if self.config.use_mock or self.config.mock_file is not None:
            self.capture: BaseVideoCapture = MockVideoCapture(
                video_path=self.config.mock_file,
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
                synthetic_pattern=self.config.mock_pattern,
            )
        else:
            self.capture = CameraCapture(
                device_id=self.config.camera_id,
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
            )

        # 2. Preprocessor
        self.preprocessor = RealtimePreprocessor(
            target_size=(self.config.width, self.config.height),
            mirror=self.config.mirror,
            to_rgb=False,  # Keep BGR for display and let landmarker handle RGB conversion
        )

        # 3. Streaming Landmark Extractor
        self.landmarker = StreamingHandLandmarker()

        # 4. Inference Engine (instantiate with candidate models or defaults)
        try:
            self.engine = RealtimeInferenceEngine(
                model=model,
                model_path=self.config.model_path,
                min_confidence_threshold=self.config.min_confidence,
            )
        except Exception as e:
            logger.warning("Could not initialize RealtimeInferenceEngine: %s", e)
            self.engine = None

        # 5. Temporal Smoother
        self.smoother = TemporalPredictionSmoother(
            ema_alpha=self.config.smoothing_alpha,
            hysteresis_threshold=self.config.hysteresis_frames,
            min_dwell_ms=self.config.min_dwell_ms,
        )

        # 6. Confidence Calibrator & OOV Gater
        self.calibrator = ConfidenceCalibrator(
            min_confidence=self.config.min_confidence,
            profile=self.config.sensitivity_profile,
        )

        # 7. Visual HUD Overlay
        self.overlay = VisualHUDOverlay(
            config=OverlayConfig(
                show_skeleton=self.config.show_skeleton,
                show_candidates=self.config.show_candidates,
            )
        )

        # 8. Accessible Text Display
        self.text_display = AccessibleTextDisplay(
            config=TextDisplayConfig(
                include_hindi=True,
                suppress_duplicates=True,
            )
        )

        # Operational state
        self._is_running = False
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._fps_counter = 0
        self._current_fps = 0.0
        self.telemetry = PipelineTelemetry()

    def start(self) -> None:
        """Start underlying capture and landmark workers."""
        logger.info("Starting SANKET Real-Time Pipeline...")
        self.capture.start()
        self._is_running = True
        self._last_fps_time = time.perf_counter()
        self._frame_count = 0

    def stop(self) -> None:
        """Gracefully terminate all streaming components and release hardware."""
        logger.info("Stopping SANKET Real-Time Pipeline...")
        self._is_running = False
        try:
            self.capture.stop()
        except Exception as e:
            logger.warning("Error stopping capture: %s", e)
        try:
            self.landmarker.close()
        except Exception as e:
            logger.warning("Error closing landmarker: %s", e)
        if not self.config.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def __enter__(self) -> RealtimePipeline:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def step(self, timeout_s: float = 1.0) -> tuple[np.ndarray | None, CalibratedPrediction | None]:
        """Execute one complete forward step of the real-time recognition pipeline.

        Args:
            timeout_s: Maximum seconds to wait for next camera frame.

        Returns:
            Tuple of (rendered_frame, calibrated_prediction).
        """
        step_start_time = time.perf_counter()

        # 1. Acquire Frame
        captured = self.capture.read(timeout=timeout_s)
        if captured is None:
            return None, None

        ts_ms = captured.timestamp_ms

        # 2. Preprocess Frame
        active_frame, _ = self.preprocessor.process_frame(captured.frame)

        # 3. Extract Streaming Landmarks
        norm_landmarks: NormalizedFrameLandmarks | None = None
        try:
            norm_landmarks = self.landmarker.process_frame(active_frame, timestamp_ms=ts_ms)
        except Exception as e:
            logger.debug("Landmark extraction error: %s", e)

        # Tracking state
        hands_state = "None"
        if norm_landmarks is not None:
            if norm_landmarks.left_detected and norm_landmarks.right_detected:
                hands_state = "Both"
            elif norm_landmarks.left_detected:
                hands_state = "Left"
            elif norm_landmarks.right_detected:
                hands_state = "Right"

        # 4. Feature Extraction & Model Inference
        raw_pred: InferenceResult | None = None
        has_hands = (
            norm_landmarks is not None
            and (norm_landmarks.left_detected or norm_landmarks.right_detected)
        )
        if self.engine is not None and has_hands and norm_landmarks is not None:
            try:
                # Use normalized flat features
                raw_pred = self.engine.predict(norm_landmarks, timestamp_ms=ts_ms)
            except Exception as e:
                logger.debug("Inference step error: %s", e)

        # 5. Temporal Prediction Smoothing
        smoothed_pred: SmoothedPrediction | None = None
        if raw_pred is not None:
            smoothed_pred = self.smoother.update(raw_pred)

        # 6. Confidence Calibration & OOV Gating
        calibrated_pred: CalibratedPrediction | None = None
        if smoothed_pred is not None:
            calibrated_pred = self.calibrator.evaluate(
                smoothed_pred,
                landmarks=norm_landmarks,
                timestamp_ms=ts_ms,
            )
        elif raw_pred is not None:
            calibrated_pred = self.calibrator.evaluate(
                raw_pred,
                landmarks=norm_landmarks,
                timestamp_ms=ts_ms,
            )

        # 7. Update Text Subtitle History
        if calibrated_pred is not None:
            self.text_display.update(calibrated_pred)

        # Update Telemetry & Latencies
        step_latency_ms = (time.perf_counter() - step_start_time) * 1000.0
        self._update_telemetry(ts_ms, step_latency_ms, calibrated_pred, hands_state)

        # 8. Render Visual HUD Overlay
        output_frame = active_frame
        if self.config.show_hud and not self.config.headless:
            is_valid = bool(calibrated_pred and calibrated_pred.is_valid_sign)
            status_text = "Recognizing" if is_valid else "Searching"
            hud_telemetry = TelemetryData(
                capture_fps=self.capture.fps,
                inference_fps=self._current_fps,
                latency_ms=step_latency_ms,
                hands_detected=hands_state,
                dropped_frames=captured.dropped_frames,
                status_message=status_text,
            )
            output_frame = self.overlay.render(
                active_frame,
                prediction=calibrated_pred or smoothed_pred or raw_pred,
                landmarks=norm_landmarks,
                telemetry=hud_telemetry,
            )

        return output_frame, calibrated_pred

    def run_loop(
        self,
        max_frames: int | None = None,
        callback: Callable[[np.ndarray, CalibratedPrediction | None], None] | None = None,
    ) -> None:
        """Run the streaming pipeline loop until user exit ('q' key) or max_frames reached."""
        self.start()
        frames_run = 0
        try:
            while self._is_running:
                frame, pred = self.step()
                if frame is None:
                    continue

                frames_run += 1
                if callback is not None:
                    callback(frame, pred)

                # GUI Window handling
                if not self.config.headless:
                    cv2.imshow(self.config.window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' or ESC
                        logger.info("User requested exit.")
                        break

                if max_frames is not None and frames_run >= max_frames:
                    break
        finally:
            self.stop()

    def _update_telemetry(
        self,
        timestamp_ms: float,
        latency_ms: float,
        prediction: CalibratedPrediction | None,
        hands_state: str,
    ) -> None:
        """Compute rolling FPS and update current telemetry structure."""
        self._frame_count += 1
        self._fps_counter += 1

        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._current_fps = float(self._fps_counter / elapsed)
            self._fps_counter = 0
            self._last_fps_time = now

        active_gloss = prediction.gloss if prediction else "IDLE"
        conf = prediction.confidence if prediction else 0.0

        self.telemetry = PipelineTelemetry(
            processed_frames=self._frame_count,
            dropped_frames=getattr(self.capture, "dropped_frames", 0),
            capture_fps=getattr(self.capture, "fps", 30.0),
            pipeline_fps=self._current_fps,
            total_latency_ms=latency_ms,
            active_gloss=active_gloss,
            confidence=conf,
            hands_detected=hands_state,
        )
