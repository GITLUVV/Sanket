"""Multidimensional landmark sequence tensor representations and batch generation (SANKET-061).

Constructs structured (batch_size, sequence_length, feature_dim) float32 tensors
from variable-length or fixed-length SequenceLandmarks objects.

Supports:
- Normalized 3D skeletal landmark coordinates (126 dims: 21 left + 21 right hands).
- Geometric augmentation (joint angles and inter-finger distances).
- Temporal padding, truncation, and binary attention/padding masks (1 = valid, 0 = padded).
- Batch tensor collator utilities for NumPy and PyTorch sequence model training.
- Compressed serialization and deserialization for temporal sequence datasets (.npz).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sanket.features.geometry import compute_joint_angles, compute_pairwise_distances
from sanket.vision.landmarks import COORDINATE_DIM, NUM_HAND_LANDMARKS, SequenceLandmarks

logger = logging.getLogger(__name__)

# Dimension Constants
HAND_COORDINATE_DIM = NUM_HAND_LANDMARKS * COORDINATE_DIM  # 63
DUAL_HAND_COORDINATE_DIM = HAND_COORDINATE_DIM * 2  # 126
NUM_ANGLES_PER_HAND = 19  # 15 flexion + 4 abduction
NUM_DISTANCES_PER_HAND = 24  # 10 inter-fingertip + 10 tip-to-wrist + 4 mcp-to-wrist
NUM_GEOMETRIC_PER_HAND = NUM_ANGLES_PER_HAND + NUM_DISTANCES_PER_HAND  # 43
GEOMETRIC_FEATURE_DIM = NUM_GEOMETRIC_PER_HAND * 2  # 86 (left + right)
AUGMENTED_FEATURE_DIM = DUAL_HAND_COORDINATE_DIM + GEOMETRIC_FEATURE_DIM  # 212


@dataclass(frozen=True)
class SequenceTensor:
    """Multidimensional tensor representation for a single sign gesture sequence.

    Attributes:
        data: Feature tensor of shape (sequence_length, feature_dim) and dtype float32.
        mask: Boolean validity mask of shape (sequence_length,) where True represents
            valid temporal frames and False represents padded frames.
        original_length: Actual unpadded frame count prior to padding or truncation.
        gloss: Optional string class label / gloss.
        class_id: Optional integer class index.
        signer_id: Optional signer identifier string.
        metadata: Optional dictionary with recording or feature parameters.
    """

    data: np.ndarray  # Shape: (T, D), dtype: float32
    mask: np.ndarray  # Shape: (T,), dtype: bool
    original_length: int
    gloss: str | None = None
    class_id: int | None = None
    signer_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"SequenceTensor data must be 2D (T, D), got shape {self.data.shape}")
        if self.mask.ndim != 1:
            raise ValueError(f"SequenceTensor mask must be 1D (T,), got shape {self.mask.shape}")
        if self.data.shape[0] != self.mask.shape[0]:
            raise ValueError(
                f"Data length {self.data.shape[0]} does not match mask length {self.mask.shape[0]}"
            )

    @property
    def sequence_length(self) -> int:
        """Temporal length (frames) of the sequence tensor."""
        return int(self.data.shape[0])

    @property
    def feature_dim(self) -> int:
        """Feature dimensionality (channels) per temporal step."""
        return int(self.data.shape[1])


@dataclass(frozen=True)
class BatchSequenceTensor:
    """Batched sequence tensor formatted for recurrent or temporal network training.

    Attributes:
        data: Batch tensor of shape (batch_size, max_sequence_length, feature_dim), float32.
        mask: Boolean mask of shape (batch_size, max_sequence_length), bool.
        lengths: Actual unpadded frame lengths of shape (batch_size,), int32.
        labels: Optional class index array of shape (batch_size,), int64.
        glosses: Optional list of string gloss names of length batch_size.
    """

    data: np.ndarray  # Shape: (B, T, D), dtype: float32
    mask: np.ndarray  # Shape: (B, T), dtype: bool
    lengths: np.ndarray  # Shape: (B,), dtype: int32
    labels: np.ndarray | None = None  # Shape: (B,), dtype: int64
    glosses: list[str] | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"Batch data must be 3D (B, T, D), got shape {self.data.shape}")
        if self.mask.ndim != 2:
            raise ValueError(f"Batch mask must be 2D (B, T), got shape {self.mask.shape}")
        if self.data.shape[0] != self.mask.shape[0] or self.data.shape[1] != self.mask.shape[1]:
            raise ValueError(
                f"Batch data shape {self.data.shape} incompatible with mask shape {self.mask.shape}"
            )

    @property
    def batch_size(self) -> int:
        """Number of sequences in this batch."""
        return int(self.data.shape[0])

    @property
    def sequence_length(self) -> int:
        """Maximum temporal sequence length in this batch."""
        return int(self.data.shape[1])

    @property
    def feature_dim(self) -> int:
        """Dimensionality of each temporal feature slice."""
        return int(self.data.shape[2])


class SequenceTensorBuilder:
    """Builder for transforming SequenceLandmarks into structured temporal sequence tensors.

    Supports coordinate feature extraction, geometric joint angle augmentation,
    temporal sequence truncation, and zero-padding with boolean validity masking.
    """

    def __init__(
        self,
        target_length: int = 30,
        include_geometry: bool = False,
        padding_mode: str = "post",
        truncation_mode: str = "post",
    ) -> None:
        """Initialize the SequenceTensorBuilder.

        Args:
            target_length: Fixed number of temporal frames T (default: 30 frames).
            include_geometry: If True, augments coordinates with joint angles and
                inter-finger distances (176 dims instead of 126 dims).
            padding_mode: 'post' (pad end of sequence) or 'pre' (pad start of sequence).
            truncation_mode: 'post' (keep first T frames) or 'uniform' (subsample T frames).

        Raises:
            ValueError: If target_length <= 0 or invalid padding/truncation modes.
        """
        if target_length <= 0:
            raise ValueError(f"target_length must be positive, got {target_length}")
        if padding_mode not in {"post", "pre"}:
            raise ValueError(f"Invalid padding_mode '{padding_mode}', expected 'post' or 'pre'")
        if truncation_mode not in {"post", "uniform"}:
            raise ValueError(
                f"Invalid truncation_mode '{truncation_mode}', expected 'post' or 'uniform'"
            )

        self.target_length = target_length
        self.include_geometry = include_geometry
        self.padding_mode = padding_mode
        self.truncation_mode = truncation_mode

    @property
    def feature_dim(self) -> int:
        """Expected feature dimension per time step."""
        return AUGMENTED_FEATURE_DIM if self.include_geometry else DUAL_HAND_COORDINATE_DIM

    def _extract_frame_features(self, sequence: SequenceLandmarks) -> np.ndarray:
        """Extract unpadded (T_orig, feature_dim) feature array from SequenceLandmarks."""
        t_orig = sequence.num_frames
        if t_orig == 0:
            raise ValueError("Cannot build tensor from empty SequenceLandmarks (0 frames).")

        # 1. Coordinate features: shape (T, 126)
        coord_features = sequence.to_flat_matrix()

        if not self.include_geometry:
            return coord_features.astype(np.float32)

        # 2. Geometric features: shape (T, 86)
        geom_features = np.zeros((t_orig, GEOMETRIC_FEATURE_DIM), dtype=np.float32)
        for i in range(t_orig):
            # Left hand geometry
            if sequence.left_detected[i]:
                left_angles = compute_joint_angles(sequence.left_hand[i])
                left_dists = compute_pairwise_distances(sequence.left_hand[i])
                geom_features[i, :NUM_GEOMETRIC_PER_HAND] = np.concatenate(
                    [left_angles, left_dists]
                )

            # Right hand geometry
            if sequence.right_detected[i]:
                right_angles = compute_joint_angles(sequence.right_hand[i])
                right_dists = compute_pairwise_distances(sequence.right_hand[i])
                geom_features[i, NUM_GEOMETRIC_PER_HAND:] = np.concatenate(
                    [right_angles, right_dists]
                )

        # Concatenate: shape (T, 212)
        return np.concatenate([coord_features, geom_features], axis=1).astype(np.float32)

    def build_tensor(
        self,
        sequence: SequenceLandmarks,
        gloss: str | None = None,
        class_id: int | None = None,
        signer_id: str | None = None,
    ) -> SequenceTensor:
        """Transform a SequenceLandmarks object into a padded SequenceTensor of fixed length.

        Args:
            sequence: Input SequenceLandmarks.
            gloss: Optional sign gloss name.
            class_id: Optional integer class index.
            signer_id: Optional signer identifier.

        Returns:
            SequenceTensor of shape (target_length, feature_dim) with boolean mask.
        """
        raw_features = self._extract_frame_features(sequence)
        t_orig, feat_dim = raw_features.shape

        # Case 1: Sequence matches target length exactly
        if t_orig == self.target_length:
            data = raw_features
            mask = np.ones(self.target_length, dtype=bool)

        # Case 2: Sequence longer than target -> Truncate or uniformly sample
        elif t_orig > self.target_length:
            if self.truncation_mode == "uniform":
                indices = np.linspace(0, t_orig - 1, self.target_length).round().astype(int)
                data = raw_features[indices]
            else:  # 'post'
                data = raw_features[: self.target_length]
            mask = np.ones(self.target_length, dtype=bool)

        # Case 3: Sequence shorter than target -> Zero pad
        else:
            pad_len = self.target_length - t_orig
            pad_block = np.zeros((pad_len, feat_dim), dtype=np.float32)

            if self.padding_mode == "post":
                data = np.concatenate([raw_features, pad_block], axis=0)
                mask = np.concatenate(
                    [np.ones(t_orig, dtype=bool), np.zeros(pad_len, dtype=bool)]
                )
            else:  # 'pre'
                data = np.concatenate([pad_block, raw_features], axis=0)
                mask = np.concatenate(
                    [np.zeros(pad_len, dtype=bool), np.ones(t_orig, dtype=bool)]
                )

        return SequenceTensor(
            data=data.astype(np.float32),
            mask=mask,
            original_length=t_orig,
            gloss=gloss,
            class_id=class_id,
            signer_id=signer_id,
        )


def collate_sequence_tensors(
    tensors: list[SequenceTensor],
    pad_to_batch_max: bool = False,
) -> BatchSequenceTensor:
    """Collate a list of SequenceTensor objects into a BatchSequenceTensor.

    Args:
        tensors: Non-empty list of SequenceTensor objects.
        pad_to_batch_max: If True, dynamically pads to the maximum sequence length
            found in the batch. If False, all tensors must already share target_length.

    Returns:
        BatchSequenceTensor with shape (batch_size, sequence_length, feature_dim).

    Raises:
        ValueError: If tensors list is empty, feature dimensions mismatch, or
            sequence lengths mismatch when pad_to_batch_max is False.
    """
    if not tensors:
        raise ValueError("Cannot collate empty list of sequence tensors.")

    batch_size = len(tensors)
    feat_dim = tensors[0].feature_dim

    # Verify uniform feature dimensionality
    for i, t in enumerate(tensors):
        if t.feature_dim != feat_dim:
            raise ValueError(
                f"Tensor at index {i} has feature_dim {t.feature_dim}, expected {feat_dim}"
            )

    if pad_to_batch_max:
        max_len = max(t.sequence_length for t in tensors)
    else:
        max_len = tensors[0].sequence_length
        for i, t in enumerate(tensors):
            if t.sequence_length != max_len:
                raise ValueError(
                    f"Tensor {i} has length {t.sequence_length} != {max_len}. "
                    "Set pad_to_batch_max=True for variable-length batches."
                )

    batch_data = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
    batch_mask = np.zeros((batch_size, max_len), dtype=bool)
    lengths = np.zeros(batch_size, dtype=np.int32)

    has_labels = tensors[0].class_id is not None
    labels = np.zeros(batch_size, dtype=np.int64) if has_labels else None
    glosses: list[str] | None = [t.gloss or "" for t in tensors] if tensors[0].gloss else None

    for i, t in enumerate(tensors):
        curr_len = t.sequence_length
        batch_data[i, :curr_len] = t.data
        batch_mask[i, :curr_len] = t.mask
        lengths[i] = t.original_length

        if has_labels and labels is not None and t.class_id is not None:
            labels[i] = t.class_id

    return BatchSequenceTensor(
        data=batch_data,
        mask=batch_mask,
        lengths=lengths,
        labels=labels,
        glosses=glosses,
    )


def save_sequence_dataset_npz(
    filepath: str | Path,
    batch: BatchSequenceTensor,
    label_map: dict[int, str] | None = None,
) -> Path:
    """Save a batched sequence tensor dataset to a compressed .npz archive.

    Args:
        filepath: Target output file path.
        batch: BatchSequenceTensor containing data, mask, lengths, and optional labels.
        label_map: Optional dictionary mapping class indices to string glosses.

    Returns:
        Path of the written .npz file.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, Any] = {
        "data": batch.data,
        "mask": batch.mask,
        "lengths": batch.lengths,
    }
    if batch.labels is not None:
        save_dict["labels"] = batch.labels
    if batch.glosses is not None:
        save_dict["glosses"] = np.array(batch.glosses)
    if label_map is not None:
        import json

        save_dict["label_map_json"] = json.dumps(label_map)

    np.savez_compressed(out_path, **save_dict)
    logger.info(
        "Saved compressed sequence dataset to %s (Batch size: %d)",
        out_path,
        batch.batch_size,
    )
    return out_path


def load_sequence_dataset_npz(filepath: str | Path) -> BatchSequenceTensor:
    """Load a BatchSequenceTensor from a compressed .npz archive.

    Args:
        filepath: Path to the .npz archive.

    Returns:
        Reconstructed BatchSequenceTensor.

    Raises:
        FileNotFoundError: If filepath does not exist.
        KeyError: If required tensor fields are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Sequence dataset file not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        data = archive["data"].astype(np.float32)
        mask = archive["mask"].astype(bool)
        lengths = archive["lengths"].astype(np.int32)
        labels = archive["labels"].astype(np.int64) if "labels" in archive else None
        glosses = list(archive["glosses"].astype(str)) if "glosses" in archive else None

    return BatchSequenceTensor(
        data=data,
        mask=mask,
        lengths=lengths,
        labels=labels,
        glosses=glosses,
    )
