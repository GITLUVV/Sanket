"""Real-time sign language recognition subsystem.

Provides asynchronous, low-latency model inference workers, online feature
pipeline integration, temporal smoothing, confidence/OOV thresholding, and end-to-end orchestration.
"""

from sanket.realtime.engine import (
    InferenceResult,
    ModelLoadError,
    RealtimeInferenceEngine,
    RealtimeInferenceWorker,
)
from sanket.realtime.pipeline import (
    PipelineConfig,
    PipelineTelemetry,
    RealtimePipeline,
)
from sanket.realtime.smoothing import (
    SmoothedPrediction,
    TemporalPredictionSmoother,
)
from sanket.realtime.thresholding import (
    IDLE_CLASS_ID,
    IDLE_GLOSS,
    NO_SIGN_CLASS_ID,
    NO_SIGN_GLOSS,
    CalibratedPrediction,
    ConfidenceCalibrator,
    SensitivityProfile,
    compute_kinetic_energy,
    compute_shannon_entropy,
)

__all__ = [
    "IDLE_CLASS_ID",
    "IDLE_GLOSS",
    "NO_SIGN_CLASS_ID",
    "NO_SIGN_GLOSS",
    "CalibratedPrediction",
    "ConfidenceCalibrator",
    "InferenceResult",
    "ModelLoadError",
    "PipelineConfig",
    "PipelineTelemetry",
    "RealtimeInferenceEngine",
    "RealtimeInferenceWorker",
    "RealtimePipeline",
    "SensitivityProfile",
    "SmoothedPrediction",
    "TemporalPredictionSmoother",
    "compute_kinetic_energy",
    "compute_shannon_entropy",
]
