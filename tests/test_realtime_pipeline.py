"""Unit tests for the end-to-end real-time recognition pipeline (SANKET-048)."""

from __future__ import annotations

import numpy as np

from sanket.models.random_forest import RandomForestSignClassifier
from sanket.realtime.pipeline import PipelineConfig, RealtimePipeline
from sanket.realtime.thresholding import CalibratedPrediction


def _create_mock_model() -> tuple[RandomForestSignClassifier, dict[int, str]]:
    """Create a lightweight deterministic dummy model with matching feature dimension."""
    rng = np.random.RandomState(42)
    features = rng.randn(20, 645).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=int)
    clf = RandomForestSignClassifier(n_estimators=5, random_state=42)
    clf.fit(features, y)
    label_map = {0: "hello", 1: "thank_you"}
    return clf, label_map


class TestRealtimePipelineIntegration:
    """Test suite verifying end-to-end real-time pipeline integration."""

    def test_pipeline_initialization_with_mock_source(self) -> None:
        model, label_map = _create_mock_model()
        config = PipelineConfig(
            use_mock=True,
            mock_pattern="solid",
            headless=True,
            width=320,
            height=240,
        )
        pipeline = RealtimePipeline(config=config, model=model, label_map=label_map)
        assert pipeline.capture is not None
        assert pipeline.engine is not None
        assert pipeline.engine.model.is_fitted is True
        assert pipeline.telemetry.processed_frames == 0

    def test_single_pipeline_step_execution(self) -> None:
        model, label_map = _create_mock_model()
        config = PipelineConfig(
            use_mock=True,
            mock_pattern="moving_box",
            headless=True,
            width=320,
            height=240,
        )
        pipeline = RealtimePipeline(config=config, model=model, label_map=label_map)

        with pipeline:
            frame, pred = pipeline.step(timeout_s=1.0)
            assert frame is not None
            assert frame.shape == (240, 320, 3)
            # Pred can be None or CalibratedPrediction based on synthetic detections
            if pred is not None:
                assert isinstance(pred, CalibratedPrediction)

            assert pipeline.telemetry.processed_frames == 1

    def test_multi_frame_run_loop_with_callback(self) -> None:
        model, label_map = _create_mock_model()
        config = PipelineConfig(
            use_mock=True,
            mock_pattern="solid",
            headless=True,
            width=320,
            height=240,
        )
        pipeline = RealtimePipeline(config=config, model=model, label_map=label_map)

        frames_collected = []

        def frame_cb(frame: np.ndarray, pred: CalibratedPrediction | None) -> None:
            frames_collected.append(frame)

        pipeline.run_loop(max_frames=5, callback=frame_cb)

        assert len(frames_collected) == 5
        assert pipeline.telemetry.processed_frames == 5

    def test_pipeline_text_display_integration(self) -> None:
        model, label_map = _create_mock_model()
        config = PipelineConfig(
            use_mock=True,
            mock_pattern="solid",
            headless=True,
            width=320,
            height=240,
        )
        pipeline = RealtimePipeline(config=config, model=model, label_map=label_map)

        # Ingest a manual sign into pipeline's text display
        entry = pipeline.text_display.update("namaste", confidence=0.95, timestamp_ms=1000.0)
        assert entry is not None
        assert entry.english_text == "Hello"
        assert pipeline.text_display.get_full_transcript() == "Hello"

    def test_pipeline_context_manager_cleanup(self) -> None:
        config = PipelineConfig(use_mock=True, headless=True)
        pipeline = RealtimePipeline(config=config)

        assert pipeline._is_running is False
        with pipeline:
            assert pipeline._is_running is True
        assert pipeline._is_running is False

    def test_pipeline_step_with_detected_hands_yields_prediction(self) -> None:
        """Verify that when hand landmarks are detected, pipeline produces prediction."""
        from unittest.mock import MagicMock

        from sanket.vision.landmarks import FrameLandmarks
        from sanket.vision.streaming_landmarks import NormalizedFrameLandmarks

        model, label_map = _create_mock_model()
        config = PipelineConfig(
            use_mock=True,
            mock_pattern="solid",
            headless=True,
            width=320,
            height=240,
        )
        pipeline = RealtimePipeline(config=config, model=model, label_map=label_map)

        # Mock landmarker to return detected hand landmarks
        raw_lm = FrameLandmarks(
            left_hand=np.zeros((21, 3), dtype=np.float32),
            right_hand=np.zeros((21, 3), dtype=np.float32),
            left_detected=True,
            right_detected=False,
            left_confidence=0.95,
            right_confidence=0.0,
        )
        norm_lm = NormalizedFrameLandmarks(
            raw_landmarks=raw_lm,
            left_hand_normalized=np.random.randn(21, 3).astype(np.float32),
            right_hand_normalized=np.zeros((21, 3), dtype=np.float32),
            left_metadata={},
            right_metadata={},
            timestamp_ms=100.0,
        )

        pipeline.landmarker.process_frame = MagicMock(return_value=norm_lm)

        with pipeline:
            frame, pred = pipeline.step(timeout_s=1.0)
            assert frame is not None
            # Prediction must be produced without being swallowed by an exception
            assert pred is not None
            assert isinstance(pred, CalibratedPrediction)
            assert hasattr(pred, "gloss")
            assert hasattr(pred, "is_valid_sign")
            assert pipeline.telemetry.processed_frames == 1
