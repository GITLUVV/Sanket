"""Real-time inference engine and asynchronous model execution worker (SANKET-043).

Provides:
1. `RealtimeInferenceEngine`: Synchronous model lifecycle management, pre-warming,
   dynamic 645-dimensional feature extraction, and low-latency prediction.
2. `RealtimeInferenceWorker`: Asynchronous threaded queue worker decoupling camera
   frame acquisition (>= 30 FPS) from model inference.
3. `InferenceResult`: Structured dataclass capturing predicted gloss, class probabilities,
   latency profile, and timestamps.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sanket.features.extractor import FeatureExtractor
from sanket.models.base import BaseSignClassifier
from sanket.models.random_forest import RandomForestSignClassifier
from sanket.vision.landmarks import SequenceLandmarks
from sanket.vision.streaming_landmarks import NormalizedFrameLandmarks

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when an inference model cannot be loaded or initialized."""


@dataclass(frozen=True)
class InferenceResult:
    """Structured container capturing a single real-time sign recognition result."""

    predicted_gloss: str
    predicted_class: int
    confidence: float
    probabilities: dict[str, float]
    feature_latency_ms: float
    model_latency_ms: float
    total_latency_ms: float
    timestamp_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        """Check if top prediction confidence is non-zero."""
        return self.confidence > 0.0


class RealtimeInferenceEngine:
    """Synchronous model execution and feature extraction engine.

    Extracts 645-dimensional tabular feature representations (normalized coordinates,
    joint angles, pairwise distances, and temporal moments) and executes inference
    using the validated Random Forest baseline (ADR-0003).
    """

    def __init__(
        self,
        model: BaseSignClassifier | None = None,
        model_path: str | Path | None = None,
        feature_extractor: FeatureExtractor | None = None,
        min_confidence_threshold: float = 0.0,
    ) -> None:
        """Initialize the real-time inference engine.

        Args:
            model: Optional pre-loaded classifier implementing BaseSignClassifier.
            model_path: Optional path to serialized model artifact (.joblib).
            feature_extractor: Optional configured FeatureExtractor instance.
            min_confidence_threshold: Minimum confidence threshold to return a valid gloss.

        Raises:
            ModelLoadError: If neither model nor model_path is provided, or model fails to load.
        """
        self.min_confidence_threshold = min_confidence_threshold

        if feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                include_coordinates=True,
                include_angles=True,
                include_distances=True,
                include_inter_hand=True,
                normalize_input=False,  # Input is already normalized in streaming pipeline
                pooling_strategy="mean_std_delta",
                temporal_stats=("mean", "std", "delta"),
            )
        else:
            self.feature_extractor = feature_extractor

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model_from_disk(model_path)
        else:
            # Default candidates
            candidates = [
                Path("models/random_forest_baseline.joblib"),
                Path("models/baseline_rf.joblib"),
                Path.home() / ".cache" / "sanket" / "models" / "random_forest_baseline.joblib",
            ]
            loaded = None
            for cand in candidates:
                if cand.is_file():
                    try:
                        loaded = self._load_model_from_disk(cand)
                        break
                    except Exception:
                        continue
            if loaded is None:
                raise ModelLoadError(
                    "No model provided and default baseline artifacts not found in models/."
                )
            self.model = loaded

        if not self.model.is_fitted:
            raise ModelLoadError("Model instance must be fitted prior to engine initialization.")

        self.classes: list[str] = (
            self.model.classes
            if self.model.classes is not None
            else [f"class_{i}" for i in range(16)]
        )

        # Pre-warm model to avoid cold-start JIT or allocation latency during live streaming
        self.pre_warm()

    def _load_model_from_disk(self, path: str | Path) -> BaseSignClassifier:
        """Load model artifact from disk."""
        p = Path(path)
        if not p.is_file():
            raise ModelLoadError(f"Model artifact not found at: {path}")
        try:
            return RandomForestSignClassifier.load(p)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model from {path}: {exc}") from exc

    def pre_warm(self) -> None:
        """Pre-warm the inference engine with a dummy 645-dimensional feature evaluation."""
        dummy_dim = self.feature_extractor.output_dimension
        dummy_x = np.zeros((1, dummy_dim), dtype=np.float32)
        try:
            self.model.predict(dummy_x)
            self.model.predict_proba(dummy_x)
            logger.debug("RealtimeInferenceEngine successfully pre-warmed.")
        except Exception as exc:
            logger.warning("Pre-warm execution encountered warning: %s", exc)

    def extract_features(
        self,
        frames: list[NormalizedFrameLandmarks] | NormalizedFrameLandmarks | SequenceLandmarks,
    ) -> np.ndarray:
        """Extract a single 645-dimensional feature row from a sequence or single frame.

        Args:
            frames: Single NormalizedFrameLandmarks, list of NormalizedFrameLandmarks,
                or SequenceLandmarks object.

        Returns:
            2D numpy array of shape (1, 645) and dtype float32.

        Raises:
            ValueError: If frames sequence is empty.
        """
        if isinstance(frames, SequenceLandmarks):
            if frames.num_frames == 0:
                raise ValueError("Cannot extract features from empty frame sequence.")
            seq = frames
        else:
            if isinstance(frames, NormalizedFrameLandmarks):
                frame_list = [frames]
            else:
                frame_list = list(frames)

            if len(frame_list) == 0:
                raise ValueError("Cannot extract features from empty frame sequence.")
            t = len(frame_list)
            left_hands = np.zeros((t, 21, 3), dtype=np.float32)
            right_hands = np.zeros((t, 21, 3), dtype=np.float32)
            left_det = np.zeros(t, dtype=bool)
            right_det = np.zeros(t, dtype=bool)
            left_conf = np.zeros(t, dtype=np.float32)
            right_conf = np.zeros(t, dtype=np.float32)

            for i, f in enumerate(frame_list):
                left_hands[i] = f.left_hand_normalized
                right_hands[i] = f.right_hand_normalized
                left_det[i] = f.left_detected
                right_det[i] = f.right_detected
                left_conf[i] = f.raw_landmarks.left_confidence
                right_conf[i] = f.raw_landmarks.right_confidence

            seq = SequenceLandmarks(
                left_hand=left_hands,
                right_hand=right_hands,
                left_detected=left_det,
                right_detected=right_det,
                left_confidence=left_conf,
                right_confidence=right_conf,
            )

        feat_vector = self.feature_extractor.extract_sequence_features(seq, normalize=False)
        return feat_vector.features.reshape(1, -1).astype(np.float32)

    def predict(
        self,
        frames: (
            list[NormalizedFrameLandmarks]
            | NormalizedFrameLandmarks
            | SequenceLandmarks
            | np.ndarray
        ),
        timestamp_ms: float | None = None,
    ) -> InferenceResult:
        """Execute end-to-end feature extraction and model prediction.

        Args:
            frames: Sequence of normalized frames, single NormalizedFrameLandmarks,
                SequenceLandmarks, or precomputed (1, 645) array.
            timestamp_ms: Optional millisecond timestamp. If None, derived from time.perf_counter().

        Returns:
            InferenceResult container with predicted gloss, probabilities, and latency.
        """
        if timestamp_ms is None:
            timestamp_ms = time.perf_counter() * 1000.0

        # 1. Feature extraction
        t0 = time.perf_counter()
        if isinstance(frames, np.ndarray):
            x = frames if frames.ndim == 2 else frames.reshape(1, -1)
            feature_ms = 0.0
        else:
            x = self.extract_features(frames)
            feature_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Model inference (optimized to evaluate probabilities and derive argmax)
        t1 = time.perf_counter()
        probas_matrix = self.model.predict_proba(x)[0]
        pred_class_idx = int(np.argmax(probas_matrix))
        model_ms = (time.perf_counter() - t1) * 1000.0
        total_ms = feature_ms + model_ms

        top_confidence = float(probas_matrix[pred_class_idx])

        # Map class index to gloss
        if 0 <= pred_class_idx < len(self.classes):
            gloss = self.classes[pred_class_idx]
        else:
            gloss = f"class_{pred_class_idx}"

        if top_confidence < self.min_confidence_threshold:
            gloss = "UNCERTAIN"

        probabilities_dict = {
            self.classes[i] if i < len(self.classes) else f"class_{i}": float(probas_matrix[i])
            for i in range(len(probas_matrix))
        }

        return InferenceResult(
            predicted_gloss=gloss,
            predicted_class=pred_class_idx,
            confidence=top_confidence,
            probabilities=probabilities_dict,
            feature_latency_ms=feature_ms,
            model_latency_ms=model_ms,
            total_latency_ms=total_ms,
            timestamp_ms=timestamp_ms,
        )


class RealtimeInferenceWorker:
    """Asynchronous threaded inference worker.

    Consumes sliding windows of normalized frames from an incoming thread-safe queue,
    runs inference in a dedicated background worker thread, and publishes structured
    InferenceResult objects to an output queue without dropping camera acquisition frames.
    """

    def __init__(
        self,
        engine: RealtimeInferenceEngine,
        queue_size: int = 10,
    ) -> None:
        """Initialize the threaded inference worker.

        Args:
            engine: Configured RealtimeInferenceEngine instance.
            queue_size: Max size of pending input inference tasks.
        """
        self.engine = engine
        self._in_queue: queue.Queue[
            tuple[list[NormalizedFrameLandmarks] | SequenceLandmarks | np.ndarray, float]
        ] = queue.Queue(maxsize=queue_size)
        self._out_queue: queue.Queue[InferenceResult] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._latest_result: InferenceResult | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="RealtimeInferenceWorker",
        )
        self._worker_thread.start()
        logger.info("RealtimeInferenceWorker started.")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the background worker thread gracefully."""
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        self._worker_thread = None
        logger.info("RealtimeInferenceWorker stopped.")

    def submit_async(
        self,
        frames: list[NormalizedFrameLandmarks] | SequenceLandmarks | np.ndarray,
        timestamp_ms: float | None = None,
    ) -> bool:
        """Submit a frame sequence for asynchronous inference without blocking.

        If the queue is full (worker running slower than caller), the oldest task
        is dropped to guarantee zero pipeline lag.

        Args:
            frames: Frame sequence or feature vector to evaluate.
            timestamp_ms: Monotonic timestamp in ms.

        Returns:
            True if submitted successfully, False if dropped due to contention.
        """
        if timestamp_ms is None:
            timestamp_ms = time.perf_counter() * 1000.0

        try:
            self._in_queue.put_nowait((frames, timestamp_ms))
            return True
        except queue.Full:
            # Drop oldest to prevent backpressure latency accumulation
            try:
                _ = self._in_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._in_queue.put_nowait((frames, timestamp_ms))
                return True
            except queue.Full:
                return False

    def get_latest_result(self) -> InferenceResult | None:
        """Get the most recent InferenceResult atomically."""
        with self._lock:
            return self._latest_result

    def poll_result(self, timeout: float | None = None) -> InferenceResult | None:
        """Poll the next available InferenceResult from the output queue."""
        try:
            res = self._out_queue.get(timeout=timeout)
            with self._lock:
                self._latest_result = res
            return res
        except queue.Empty:
            return None

    def _run_loop(self) -> None:
        """Main inference worker loop."""
        while not self._stop_event.is_set():
            try:
                item = self._in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            frames, ts = item
            try:
                result = self.engine.predict(frames, timestamp_ms=ts)
                with self._lock:
                    self._latest_result = result

                # Push to out queue, dropping oldest if full
                if self._out_queue.full():
                    try:
                        _ = self._out_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._out_queue.put_nowait(result)
            except Exception as exc:
                logger.error("Inference worker execution failed: %s", exc)
            finally:
                self._in_queue.task_done()
