"""Unit tests for sequence data representation and tensor generation (SANKET-061)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sanket.data.sequence import (
    AUGMENTED_FEATURE_DIM,
    DUAL_HAND_COORDINATE_DIM,
    BatchSequenceTensor,
    SequenceTensor,
    SequenceTensorBuilder,
    collate_sequence_tensors,
    load_sequence_dataset_npz,
    save_sequence_dataset_npz,
)
from sanket.vision.landmarks import SequenceLandmarks


def _create_mock_sequence(num_frames: int = 25) -> SequenceLandmarks:
    """Helper creating a deterministic SequenceLandmarks object."""
    rng = np.random.RandomState(42)
    left_hand = rng.randn(num_frames, 21, 3).astype(np.float32)
    right_hand = rng.randn(num_frames, 21, 3).astype(np.float32)
    left_det = np.ones(num_frames, dtype=bool)
    right_det = np.ones(num_frames, dtype=bool)
    left_conf = np.full(num_frames, 0.9, dtype=np.float32)
    right_conf = np.full(num_frames, 0.9, dtype=np.float32)

    return SequenceLandmarks(
        left_hand=left_hand,
        right_hand=right_hand,
        left_detected=left_det,
        right_detected=right_det,
        left_confidence=left_conf,
        right_confidence=right_conf,
    )


class TestSequenceTensorRepresentation:
    """Test suite validating SequenceTensor and SequenceTensorBuilder."""

    def test_sequence_tensor_builder_defaults(self) -> None:
        builder = SequenceTensorBuilder(target_length=30, include_geometry=False)
        assert builder.target_length == 30
        assert builder.feature_dim == DUAL_HAND_COORDINATE_DIM  # 126

    def test_tensor_padding_shorter_sequence(self) -> None:
        builder = SequenceTensorBuilder(target_length=30, padding_mode="post")
        seq = _create_mock_sequence(num_frames=20)
        tensor = builder.build_tensor(seq, gloss="HELLO", class_id=0)

        assert isinstance(tensor, SequenceTensor)
        assert tensor.sequence_length == 30
        assert tensor.feature_dim == 126
        assert tensor.original_length == 20
        assert tensor.gloss == "HELLO"
        assert tensor.class_id == 0

        # Verify mask: first 20 True, last 10 False
        assert np.all(tensor.mask[:20])
        assert not np.any(tensor.mask[20:])
        # Verify padding data is zeroed out
        assert np.all(tensor.data[20:] == 0.0)

    def test_tensor_padding_pre_mode(self) -> None:
        builder = SequenceTensorBuilder(target_length=30, padding_mode="pre")
        seq = _create_mock_sequence(num_frames=20)
        tensor = builder.build_tensor(seq)

        # In pre mode, first 10 frames are zero padding
        assert not np.any(tensor.mask[:10])
        assert np.all(tensor.mask[10:])
        assert np.all(tensor.data[:10] == 0.0)

    def test_tensor_truncation_post_mode(self) -> None:
        builder = SequenceTensorBuilder(target_length=30, truncation_mode="post")
        seq = _create_mock_sequence(num_frames=45)
        tensor = builder.build_tensor(seq)

        assert tensor.sequence_length == 30
        assert tensor.original_length == 45
        assert np.all(tensor.mask)  # All 30 frames are valid

    def test_tensor_truncation_uniform_mode(self) -> None:
        builder = SequenceTensorBuilder(target_length=30, truncation_mode="uniform")
        seq = _create_mock_sequence(num_frames=50)
        tensor = builder.build_tensor(seq)

        assert tensor.sequence_length == 30
        assert np.all(tensor.mask)

    def test_augmented_geometry_features(self) -> None:
        builder = SequenceTensorBuilder(target_length=30, include_geometry=True)
        assert builder.feature_dim == AUGMENTED_FEATURE_DIM  # 212

        seq = _create_mock_sequence(num_frames=30)
        tensor = builder.build_tensor(seq)
        assert tensor.data.shape == (30, 212)


class TestBatchSequenceTensor:
    """Test suite validating batch collation and serialization."""

    def test_collate_sequence_tensors_uniform_length(self) -> None:
        builder = SequenceTensorBuilder(target_length=30)
        t1 = builder.build_tensor(_create_mock_sequence(20), gloss="HELLO", class_id=0)
        t2 = builder.build_tensor(_create_mock_sequence(25), gloss="THANK_YOU", class_id=1)

        batch = collate_sequence_tensors([t1, t2])
        assert isinstance(batch, BatchSequenceTensor)
        assert batch.batch_size == 2
        assert batch.sequence_length == 30
        assert batch.feature_dim == 126
        assert batch.data.shape == (2, 30, 126)
        assert batch.mask.shape == (2, 30)
        assert np.array_equal(batch.lengths, np.array([20, 25], dtype=np.int32))
        assert batch.labels is not None and np.array_equal(batch.labels, np.array([0, 1]))
        assert batch.glosses == ["HELLO", "THANK_YOU"]

    def test_collate_empty_list_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot collate empty list"):
            collate_sequence_tensors([])

    def test_save_and_load_sequence_dataset_npz(self, tmp_path: Path) -> None:
        builder = SequenceTensorBuilder(target_length=30)
        t1 = builder.build_tensor(_create_mock_sequence(20), gloss="HELLO", class_id=0)
        t2 = builder.build_tensor(_create_mock_sequence(30), gloss="YES", class_id=1)
        batch = collate_sequence_tensors([t1, t2])

        npz_file = tmp_path / "test_dataset.npz"
        out = save_sequence_dataset_npz(npz_file, batch, label_map={0: "HELLO", 1: "YES"})
        assert out.exists()

        loaded_batch = load_sequence_dataset_npz(npz_file)
        assert loaded_batch.batch_size == 2
        assert loaded_batch.sequence_length == 30
        assert loaded_batch.feature_dim == 126
        assert np.allclose(loaded_batch.data, batch.data)
        assert np.array_equal(loaded_batch.mask, batch.mask)
        assert np.array_equal(loaded_batch.lengths, batch.lengths)
        assert np.array_equal(loaded_batch.labels, batch.labels)
        assert loaded_batch.glosses == ["HELLO", "YES"]
