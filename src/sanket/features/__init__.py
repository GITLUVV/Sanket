"""Feature engineering package for Indian Sign Language recognition."""

from sanket.features.extractor import (
    INTER_HAND_DISTANCE_PAIRS,
    FeatureExtractor,
    FeatureVector,
)
from sanket.features.geometry import (
    ABDUCTION_ANGLE_PAIRS,
    FLEXION_ANGLE_TRIPLETS,
    NUM_HAND_ANGLES,
    NUM_HAND_DISTANCES,
    PAIRWISE_DISTANCE_PAIRS,
    compute_angle_between_vectors,
    compute_joint_angles,
    compute_pairwise_distances,
    get_hand_angle_names,
    get_hand_distance_names,
)
from sanket.features.temporal import (
    aggregate_sequence_features,
    compute_frame_velocities,
    compute_temporal_statistics,
)

__all__ = [
    "ABDUCTION_ANGLE_PAIRS",
    "FLEXION_ANGLE_TRIPLETS",
    "INTER_HAND_DISTANCE_PAIRS",
    "NUM_HAND_ANGLES",
    "NUM_HAND_DISTANCES",
    "PAIRWISE_DISTANCE_PAIRS",
    "FeatureExtractor",
    "FeatureVector",
    "aggregate_sequence_features",
    "compute_angle_between_vectors",
    "compute_frame_velocities",
    "compute_joint_angles",
    "compute_pairwise_distances",
    "compute_temporal_statistics",
    "get_hand_angle_names",
    "get_hand_distance_names",
]
