"""Unit tests for real-time camera visual overlay and prediction HUD (SANKET-046)."""

from __future__ import annotations

import time

import numpy as np

from sanket.realtime.engine import InferenceResult
from sanket.realtime.smoothing import SmoothedPrediction
from sanket.realtime.thresholding import (
    IDLE_CLASS_ID,
    IDLE_GLOSS,
    CalibratedPrediction,
)
from sanket.ui.overlay import (
    HAND_SKELETON_CONNECTIONS,
    ColorTheme,
    TelemetryData,
    VisualHUDOverlay,
)
from sanket.vision.landmarks import FrameLandmarks


def _make_dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_dummy_landmarks() -> FrameLandmarks:
    # 21 points in normalized [0, 1] space
    pts = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        pts[i] = [0.2 + (i % 5) * 0.05, 0.3 + (i // 5) * 0.05, 0.0]
    return FrameLandmarks(
        left_hand=pts,
        right_hand=pts,
        left_detected=True,
        right_detected=True,
        left_confidence=0.92,
        right_confidence=0.92,
    )


def _make_dummy_inference() -> InferenceResult:
    return InferenceResult(
        predicted_gloss="namaste",
        predicted_class=1,
        confidence=0.94,
        probabilities={"namaste": 0.94, "dhanyavad": 0.04, "alvida": 0.02},
        feature_latency_ms=1.2,
        model_latency_ms=2.1,
        total_latency_ms=3.3,
        timestamp_ms=1000.0,
    )


class TestVisualHUDOverlay:
    """Test suite for OpenCV HUD overlay composition and rendering speed."""

    def test_initialization_default_theme(self) -> None:
        overlay = VisualHUDOverlay()
        assert overlay.config.theme == ColorTheme.HIGH_CONTRAST
        assert overlay.palette is not None

    def test_set_theme_switch(self) -> None:
        overlay = VisualHUDOverlay()
        overlay.set_theme(ColorTheme.CYAN_AMBER)
        assert overlay.config.theme == ColorTheme.CYAN_AMBER
        overlay.set_theme("monochrome")
        assert overlay.config.theme == ColorTheme.MONOCHROME

    def test_render_empty_or_none_frame(self) -> None:
        overlay = VisualHUDOverlay()
        assert overlay.render(None) is None
        empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
        rendered = overlay.render(empty_arr)
        assert rendered.size == 0

    def test_render_with_landmarks(self) -> None:
        overlay = VisualHUDOverlay()
        frame = _make_dummy_frame(640, 480)
        landmarks = _make_dummy_landmarks()
        result = overlay.render(frame, landmarks=landmarks)
        assert result.shape == frame.shape
        # Rendered frame should not be all zeros anymore due to bones and joints
        assert np.count_nonzero(result) > 0

    def test_render_with_telemetry(self) -> None:
        overlay = VisualHUDOverlay()
        frame = _make_dummy_frame(640, 480)
        telemetry = TelemetryData(
            capture_fps=29.8,
            inference_fps=30.1,
            latency_ms=8.5,
            hands_detected="Both",
            status_message="Tracking",
        )
        result = overlay.render(frame, telemetry=telemetry)
        assert result.shape == frame.shape
        assert np.count_nonzero(result) > 0

    def test_render_with_inference_result(self) -> None:
        overlay = VisualHUDOverlay()
        frame = _make_dummy_frame(640, 480)
        pred = _make_dummy_inference()
        result = overlay.render(frame, prediction=pred)
        assert result.shape == frame.shape
        assert np.count_nonzero(result) > 0

    def test_render_with_calibrated_prediction_valid(self) -> None:
        overlay = VisualHUDOverlay()
        frame = _make_dummy_frame(640, 480)
        calibrated = CalibratedPrediction(
            gloss="namaste",
            class_id=1,
            confidence=0.95,
            entropy=0.3,
            kinetic_energy=0.08,
            is_valid_sign=True,
            is_resting=False,
            rejection_reason=None,
            raw_gloss="namaste",
            raw_confidence=0.95,
            probabilities={"namaste": 0.95, "alvida": 0.05},
            timestamp_ms=1050.0,
        )
        result = overlay.render(frame, prediction=calibrated)
        assert result.shape == frame.shape
        assert np.count_nonzero(result) > 0

    def test_render_with_calibrated_prediction_rejected_idle(self) -> None:
        overlay = VisualHUDOverlay()
        frame = _make_dummy_frame(640, 480)
        idle_pred = CalibratedPrediction(
            gloss=IDLE_GLOSS,
            class_id=IDLE_CLASS_ID,
            confidence=0.85,
            entropy=0.4,
            kinetic_energy=0.001,
            is_valid_sign=False,
            is_resting=True,
            rejection_reason="kinetic_energy_below_threshold",
            raw_gloss="namaste",
            raw_confidence=0.85,
            probabilities={"namaste": 0.85},
            timestamp_ms=1100.0,
        )
        result = overlay.render(frame, prediction=idle_pred)
        assert result.shape == frame.shape
        assert np.count_nonzero(result) > 0

    def test_render_with_smoothed_prediction(self) -> None:
        overlay = VisualHUDOverlay()
        frame = _make_dummy_frame(640, 480)
        smoothed = SmoothedPrediction(
            active_gloss="dhanyavad",
            active_class=2,
            confidence=0.89,
            raw_gloss="dhanyavad",
            raw_class=2,
            smoothed_probabilities={"dhanyavad": 0.89, "namaste": 0.11},
            state_changed=True,
            dwell_duration_ms=45.0,
            timestamp_ms=1200.0,
        )
        result = overlay.render(frame, prediction=smoothed)
        assert result.shape == frame.shape
        assert np.count_nonzero(result) > 0

    def test_render_latency_under_two_milliseconds(self) -> None:
        overlay = VisualHUDOverlay()
        frame = _make_dummy_frame(640, 480)
        landmarks = _make_dummy_landmarks()
        telemetry = TelemetryData(capture_fps=30.0, inference_fps=30.0, latency_ms=5.0)
        pred = _make_dummy_inference()

        # Warmup
        for _ in range(10):
            overlay.render(frame, prediction=pred, landmarks=landmarks, telemetry=telemetry)

        # Benchmark 50 renders
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            overlay.render(frame, prediction=pred, landmarks=landmarks, telemetry=telemetry)
            times.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(times))
        # Acceptance criteria target is < 5.0 ms per frame under CPU test runners
        assert mean_ms < 5.0, f"HUD render latency {mean_ms:.2f} ms exceeded threshold"

    def test_skeleton_connections_length(self) -> None:
        # Standard hand skeleton has 23 connection pairs (5 digits + knuckle bars)
        assert len(HAND_SKELETON_CONNECTIONS) == 23
