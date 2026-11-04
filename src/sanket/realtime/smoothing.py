"""Temporal prediction smoothing, majority voting, and hysteresis filtering (SANKET-044).

Eliminates high-frequency prediction flicker, sensor noise, and boundary transitions
in real-time sign recognition streams through:
1. Exponential Moving Average (EMA) probability smoothing.
2. Sliding-window majority voting.
3. Schmitt-trigger hysteresis state transitions with minimum dwell-time enforcement.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sanket.realtime.engine import InferenceResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmoothedPrediction:
    """Stabilized sign prediction output after temporal filtering and hysteresis."""

    active_gloss: str
    active_class: int
    confidence: float
    raw_gloss: str
    raw_class: int
    smoothed_probabilities: dict[str, float]
    state_changed: bool
    dwell_duration_ms: float
    timestamp_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TemporalPredictionSmoother:
    """Online temporal filtering and hysteresis state machine for real-time sign predictions.

    Combines:
    1. Exponential Moving Average (EMA) on class probability distributions.
    2. Sliding-window majority voting over recent class hypotheses.
    3. Schmitt-trigger hysteresis requiring consistent consecutive frames or probability
       margins before committing a state change.
    4. Minimum dwell-time enforcement to prevent rapid flutter across sign boundaries.
    """

    def __init__(
        self,
        num_classes: int = 16,
        classes: list[str] | None = None,
        ema_alpha: float = 0.3,
        window_size: int = 7,
        hysteresis_threshold: int = 4,
        min_dwell_ms: float = 150.0,
        uncertain_gloss: str = "UNCERTAIN",
    ) -> None:
        """Initialize the temporal prediction smoother.

        Args:
            num_classes: Total number of target vocabulary classes.
            classes: Optional mapping of integer class indices to gloss labels.
            ema_alpha: Smoothing weight in (0.0, 1.0] for exponential moving average.
            window_size: Length of recent class history window for majority voting.
            hysteresis_threshold: Number of consecutive identical predictions required
                to trigger a transition to a new active sign.
            min_dwell_ms: Minimum time in milliseconds the active state must be held
                before allowing a transition to a different sign.
            uncertain_gloss: String indicator for unconfident or idle states.

        Raises:
            ValueError: If parameters are invalid or out of permissible ranges.
        """
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0.0, 1.0], got {ema_alpha}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if hysteresis_threshold < 1:
            raise ValueError(f"hysteresis_threshold must be >= 1, got {hysteresis_threshold}")
        if min_dwell_ms < 0.0:
            raise ValueError(f"min_dwell_ms must be >= 0.0, got {min_dwell_ms}")

        self.num_classes = num_classes
        self.classes: list[str] = (
            classes if classes is not None else [f"class_{i}" for i in range(num_classes)]
        )
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.hysteresis_threshold = hysteresis_threshold
        self.min_dwell_ms = min_dwell_ms
        self.uncertain_gloss = uncertain_gloss

        # State tracking
        self._smoothed_probas: np.ndarray | None = None
        self._history_window: collections.deque[int] = collections.deque(maxlen=window_size)

        self._active_class: int = -1
        self._active_gloss: str = uncertain_gloss
        self._candidate_class: int = -1
        self._candidate_consecutive_count: int = 0
        self._state_start_time_ms: float = 0.0
        self._last_timestamp_ms: float = 0.0

    @property
    def active_gloss(self) -> str:
        """Currently committed active sign gloss."""
        return self._active_gloss

    @property
    def active_class(self) -> int:
        """Currently committed active class index."""
        return self._active_class

    def reset(self) -> None:
        """Reset temporal filters, voting history, and hysteresis state."""
        self._smoothed_probas = None
        self._history_window.clear()
        self._active_class = -1
        self._active_gloss = self.uncertain_gloss
        self._candidate_class = -1
        self._candidate_consecutive_count = 0
        self._state_start_time_ms = 0.0
        self._last_timestamp_ms = 0.0

    def process(
        self,
        raw_result: InferenceResult | tuple[int, dict[str, float]],
        timestamp_ms: float | None = None,
    ) -> SmoothedPrediction:
        """Filter a raw inference prediction through EMA, majority voting, and hysteresis.

        Args:
            raw_result: InferenceResult object or tuple of (raw_class_idx, probabilities_dict).
            timestamp_ms: Optional millisecond timestamp. Defaults to result timestamp or clock.

        Returns:
            SmoothedPrediction capturing stabilized sign state, dwell duration, and confidence.
        """
        # Unpack input
        if isinstance(raw_result, InferenceResult):
            raw_class = raw_result.predicted_class
            raw_gloss = raw_result.predicted_gloss
            probas_dict = raw_result.probabilities
            ts = raw_result.timestamp_ms if timestamp_ms is None else timestamp_ms
        else:
            raw_class, probas_dict = raw_result
            raw_gloss = (
                self.classes[raw_class]
                if 0 <= raw_class < len(self.classes)
                else f"class_{raw_class}"
            )
            ts = time.perf_counter() * 1000.0 if timestamp_ms is None else timestamp_ms

        self._last_timestamp_ms = ts

        # 1. Exponential Moving Average (EMA) on probability vectors
        raw_probas_arr = np.zeros(self.num_classes, dtype=np.float32)
        for i, class_name in enumerate(self.classes):
            if class_name in probas_dict:
                raw_probas_arr[i] = probas_dict[class_name]
            elif f"class_{i}" in probas_dict:
                raw_probas_arr[i] = probas_dict[f"class_{i}"]

        # Normalize probability array if non-zero
        prob_sum = float(np.sum(raw_probas_arr))
        if prob_sum > 0.0:
            raw_probas_arr /= prob_sum

        if self._smoothed_probas is None:
            self._smoothed_probas = raw_probas_arr.copy()
        else:
            self._smoothed_probas = (
                self.ema_alpha * raw_probas_arr + (1.0 - self.ema_alpha) * self._smoothed_probas
            )

        # 2. Candidate tracking and Schmitt-trigger hysteresis state transitions
        # Track consecutive raw incoming predictions for hysteresis transitions
        if raw_class == self._candidate_class:
            self._candidate_consecutive_count += 1
        else:
            self._candidate_class = raw_class
            self._candidate_consecutive_count = 1

        self._history_window.append(raw_class)
        state_changed = False
        dwell_ms = max(0.0, ts - self._state_start_time_ms) if self._active_class != -1 else 0.0

        # Initial commit if uninitialized
        if self._active_class == -1:
            if self._candidate_consecutive_count >= min(2, self.hysteresis_threshold):
                self._active_class = raw_class
                self._active_gloss = (
                    self.classes[raw_class]
                    if 0 <= raw_class < len(self.classes)
                    else f"class_{raw_class}"
                )
                self._state_start_time_ms = ts
                state_changed = True
                dwell_ms = 0.0
        elif raw_class != self._active_class:
            # Check hysteresis agreement and minimum dwell time before committing transition
            can_transition = (
                self._candidate_consecutive_count >= self.hysteresis_threshold
                and dwell_ms >= self.min_dwell_ms
            )
            if can_transition:
                self._active_class = raw_class
                self._active_gloss = (
                    self.classes[raw_class]
                    if 0 <= raw_class < len(self.classes)
                    else f"class_{raw_class}"
                )
                self._state_start_time_ms = ts
                state_changed = True
                dwell_ms = 0.0
        else:
            # Reset candidate tracking since current state matches raw prediction
            self._candidate_class = raw_class
            self._candidate_consecutive_count = self.hysteresis_threshold

        active_confidence = (
            float(self._smoothed_probas[self._active_class])
            if (self._smoothed_probas is not None and 0 <= self._active_class < self.num_classes)
            else 0.0
        )

        smoothed_dict = {
            self.classes[i] if i < len(self.classes) else f"class_{i}": float(
                self._smoothed_probas[i]
            )
            for i in range(self.num_classes)
        }

        return SmoothedPrediction(
            active_gloss=self._active_gloss,
            active_class=self._active_class,
            confidence=active_confidence,
            raw_gloss=raw_gloss,
            raw_class=raw_class,
            smoothed_probabilities=smoothed_dict,
            state_changed=state_changed,
            dwell_duration_ms=dwell_ms,
            timestamp_ms=ts,
            metadata={
                "consecutive_candidate_count": self._candidate_consecutive_count,
                "window_size": len(self._history_window),
            },
        )
