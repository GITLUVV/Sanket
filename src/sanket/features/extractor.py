"""Comprehensive feature engineering pipeline for Indian Sign Language recognition.

Extracts fixed-dimension, discriminative tabular feature vectors from normalized
skeletal hand and pose landmarks, combining:
1. 3D landmark coordinates (wrist-centered, palm-scaled)
2. Joint flexion and abduction angles
3. Anatomical pairwise inter-joint distances
4. Inter-hand spatial interaction metrics
5. Upper-body reference anchor distances (optional)
6. Temporal summary statistics (mean, standard deviation, net displacement)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sanket.features.geometry import (
    compute_joint_angles,
    compute_pairwise_distances,
    get_hand_angle_names,
    get_hand_distance_names,
)
from sanket.features.temporal import aggregate_sequence_features
from sanket.vision.landmarks import (
    NUM_HAND_LANDMARKS,
    FrameLandmarks,
    SequenceLandmarks,
)
from sanket.vision.normalization import LandmarkNormalizer

INTER_HAND_DISTANCE_PAIRS: list[tuple[str, int, int]] = [
    ("inter_wrist_dist", 0, 0),  # Left wrist to Right wrist
    ("inter_index_tip_dist", 8, 8),  # Left index tip to Right index tip
    ("inter_thumb_tip_dist", 4, 4),  # Left thumb tip to Right thumb tip
]


@dataclass
class FeatureVector:
    """Structured container for extracted tabular feature vectors."""

    features: np.ndarray
    feature_names: list[str]
    dimension: int = field(init=False)
    pooling_strategy: str = "mean_std_delta"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize and validate dimension."""
        if self.features.ndim == 1:
            self.dimension = int(self.features.size)
            if len(self.feature_names) != self.dimension:
                msg = (
                    f"Feature vector size ({self.dimension}) does not match "
                    f"length of feature_names ({len(self.feature_names)})"
                )
                raise ValueError(msg)
        else:
            self.dimension = int(self.features.shape[-1])
            if len(self.feature_names) != self.dimension:
                msg = (
                    f"Feature columns ({self.dimension}) does not match "
                    f"length of feature_names ({len(self.feature_names)})"
                )
                raise ValueError(msg)

    def to_dict(self) -> dict[str, float]:
        """Convert feature vector to a name-to-value dictionary."""
        if self.features.ndim != 1:
            msg = "to_dict() is only supported for 1D feature vectors"
            raise ValueError(msg)
        flat = self.features.ravel()
        return {name: float(val) for name, val in zip(self.feature_names, flat, strict=True)}


class FeatureExtractor:
    """Master feature extractor converting landmarks into ML-ready feature vectors."""

    def __init__(
        self,
        include_coordinates: bool = True,
        include_angles: bool = True,
        include_distances: bool = True,
        include_inter_hand: bool = True,
        normalize_input: bool = True,
        normalization_method: str = "palm",
        align_rotation: bool = False,
        pooling_strategy: str = "mean_std_delta",
        temporal_stats: Sequence[str] = ("mean", "std", "delta"),
    ) -> None:
        """Initialize FeatureExtractor configuration.

        Args:
            include_coordinates: Include 3D coordinates (126 dims for two hands).
            include_angles: Include flexion/abduction angles (38 dims for two hands).
            include_distances: Include anatomical pairwise distances (48 dims).
            include_inter_hand: Include inter-hand Euclidean distances (3 dims).
            normalize_input: Run landmark normalization before feature calculation.
            normalization_method: Scale normalization metric ('palm', 'max_span', 'bbox').
            align_rotation: Align 2D hand orientation vertically during normalization.
            pooling_strategy: Temporal aggregation ('mean_std_delta', 'mean_std', 'mean', 'none').
            temporal_stats: List of statistics to aggregate when pooling.
        """
        self.include_coordinates = include_coordinates
        self.include_angles = include_angles
        self.include_distances = include_distances
        self.include_inter_hand = include_inter_hand
        self.normalize_input = normalize_input
        self.normalization_method = normalization_method
        self.align_rotation = align_rotation
        self.pooling_strategy = pooling_strategy
        self.temporal_stats = list(temporal_stats)

        self._normalizer = LandmarkNormalizer(
            method=normalization_method,
            align_rotation=align_rotation,
        )

    def get_frame_feature_names(self) -> list[str]:
        """Generate deterministic column names for a single frame feature vector."""
        names: list[str] = []

        # 1. 3D Coordinates
        if self.include_coordinates:
            for hand_prefix in ("left", "right"):
                for i in range(NUM_HAND_LANDMARKS):
                    for axis in ("x", "y", "z"):
                        names.append(f"{hand_prefix}_joint_{i}_{axis}")

        # 2. Joint Angles
        if self.include_angles:
            names.extend(get_hand_angle_names(prefix="left"))
            names.extend(get_hand_angle_names(prefix="right"))

        # 3. Pairwise Distances
        if self.include_distances:
            names.extend(get_hand_distance_names(prefix="left"))
            names.extend(get_hand_distance_names(prefix="right"))

        # 4. Inter-hand Distances
        if self.include_inter_hand:
            for pair_name, _, _ in INTER_HAND_DISTANCE_PAIRS:
                names.append(pair_name)

        return names

    def get_feature_names(self) -> list[str]:
        """Generate deterministic column names for the aggregated feature vector."""
        frame_names = self.get_frame_feature_names()
        if self.pooling_strategy == "none":
            return frame_names

        # When pooled across temporal stats (e.g. mean, std, delta)
        pooled_names: list[str] = []
        for stat in self.temporal_stats:
            for fname in frame_names:
                pooled_names.append(f"{fname}_{stat}")

        return pooled_names

    @property
    def frame_dimension(self) -> int:
        """Return dimensionality of a single frame feature vector."""
        return len(self.get_frame_feature_names())

    @property
    def output_dimension(self) -> int:
        """Return dimensionality of the final aggregated feature vector."""
        return len(self.get_feature_names())

    def extract_frame_features(
        self,
        frame: FrameLandmarks,
        normalize: bool | None = None,
    ) -> np.ndarray:
        """Extract flat 1D feature vector for a single frame.

        Args:
            frame: FrameLandmarks instance containing left and right hand landmarks.
            normalize: Override self.normalize_input if explicitly provided.

        Returns:
            1D float32 array of shape (frame_dimension,).
        """
        should_normalize = self.normalize_input if normalize is None else normalize

        left_coords = frame.left_hand
        right_coords = frame.right_hand

        if should_normalize:
            norm_frame = self._normalizer.normalize_frame(frame)
            left_coords = norm_frame.left_hand
            right_coords = norm_frame.right_hand

        features_list: list[np.ndarray] = []

        # 1. 3D Coordinates
        if self.include_coordinates:
            features_list.append(left_coords.ravel().astype(np.float32))
            features_list.append(right_coords.ravel().astype(np.float32))

        # 2. Joint Angles
        if self.include_angles:
            features_list.append(compute_joint_angles(left_coords))
            features_list.append(compute_joint_angles(right_coords))

        # 3. Pairwise Distances
        if self.include_distances:
            features_list.append(compute_pairwise_distances(left_coords))
            features_list.append(compute_pairwise_distances(right_coords))

        # 4. Inter-hand Distances
        if self.include_inter_hand:
            inter_hand = np.zeros(len(INTER_HAND_DISTANCE_PAIRS), dtype=np.float32)
            # Only compute inter-hand distance if both hands are detected / non-zero
            both_present = np.any(left_coords != 0.0) and np.any(right_coords != 0.0)
            if both_present:
                for idx, (_, l_idx, r_idx) in enumerate(INTER_HAND_DISTANCE_PAIRS):
                    diff = left_coords[l_idx] - right_coords[r_idx]
                    inter_hand[idx] = float(np.linalg.norm(diff))
            features_list.append(inter_hand)

        if not features_list:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(features_list).astype(np.float32)

    def extract_sequence_features(
        self,
        sequence: SequenceLandmarks,
        normalize: bool | None = None,
    ) -> FeatureVector:
        """Extract aggregated feature vector from a temporal SequenceLandmarks object.

        Args:
            sequence: SequenceLandmarks instance.
            normalize: Override self.normalize_input if provided.

        Returns:
            FeatureVector instance containing aggregated features and metadata.
        """
        t = sequence.num_frames
        if t == 0:
            names = self.get_feature_names()
            return FeatureVector(
                features=np.zeros(len(names), dtype=np.float32),
                feature_names=names,
                pooling_strategy=self.pooling_strategy,
                metadata={"num_frames": 0},
            )

        should_normalize = self.normalize_input if normalize is None else normalize
        seq_to_process = (
            self._normalizer.normalize_sequence(sequence) if should_normalize else sequence
        )

        # Extract per-frame features: shape (T, D_frame)
        frame_dim = self.frame_dimension
        matrix = np.zeros((t, frame_dim), dtype=np.float32)

        for i in range(t):
            frame = FrameLandmarks(
                left_hand=seq_to_process.left_hand[i],
                right_hand=seq_to_process.right_hand[i],
                left_detected=bool(seq_to_process.left_detected[i]),
                right_detected=bool(seq_to_process.right_detected[i]),
                left_confidence=float(seq_to_process.left_confidence[i]),
                right_confidence=float(seq_to_process.right_confidence[i]),
            )
            matrix[i] = self.extract_frame_features(frame, normalize=False)

        if self.pooling_strategy == "none":
            return FeatureVector(
                features=matrix,
                feature_names=self.get_frame_feature_names(),
                pooling_strategy="none",
                metadata={"num_frames": t, "frame_dim": frame_dim},
            )

        # Boolean mask: frame is valid if at least one hand was detected
        mask = sequence.left_detected | sequence.right_detected
        pooled_features = aggregate_sequence_features(
            matrix,
            stats=self.temporal_stats,
            mask=mask if np.any(mask) else None,
        )

        names = self.get_feature_names()
        return FeatureVector(
            features=pooled_features,
            feature_names=names,
            pooling_strategy=self.pooling_strategy,
            metadata={
                "num_frames": t,
                "frame_dimension": frame_dim,
                "stats": self.temporal_stats,
                "valid_frames": int(np.sum(mask)),
            },
        )
