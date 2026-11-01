"""Structured experiment manifest generation and serialization.

Implements the formal benchmark artifact schema required by
docs/research/EVALUATION_METHODOLOGY.md Section 6.2.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def get_git_provenance() -> tuple[str, bool]:
    """Retrieve current Git commit SHA and dirty working tree status.

    Returns:
        Tuple of (commit_sha, is_dirty).
    """
    try:
        # Get HEAD commit SHA
        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        commit_sha = sha_proc.stdout.strip() if sha_proc.returncode == 0 else "unknown"

        # Check for uncommitted modifications
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        is_dirty = bool(status_proc.stdout.strip()) if status_proc.returncode == 0 else False
        return commit_sha, is_dirty
    except Exception:
        return "unknown", False


def compute_data_checksum(data: np.ndarray | bytes | str | Path) -> str:
    """Compute deterministic SHA-256 checksum of data or file contents.

    Args:
        data: Numpy array, raw bytes, string path, or Path object.

    Returns:
        Hexadecimal SHA-256 checksum string.
    """
    hasher = hashlib.sha256()

    if isinstance(data, (str, Path)):
        p = Path(data)
        if p.is_file():
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        hasher.update(str(data).encode("utf-8"))
        return hasher.hexdigest()

    if isinstance(data, np.ndarray):
        hasher.update(data.tobytes())
        return hasher.hexdigest()

    if isinstance(data, bytes):
        hasher.update(data)
        return hasher.hexdigest()

    hasher.update(repr(data).encode("utf-8"))
    return hasher.hexdigest()


def get_sanitized_environment_info() -> dict[str, Any]:
    """Capture runtime platform and dependency environment metadata without private paths.

    Returns:
        Sanitized environment dictionary suitable for public reproducibility logging.
    """
    import sklearn

    env_info: dict[str, Any] = {
        "os_name": os.name,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    try:
        import joblib

        env_info["joblib_version"] = joblib.__version__
    except ImportError:
        pass

    try:
        import cv2

        env_info["opencv_version"] = cv2.__version__
    except ImportError:
        pass

    try:
        import mediapipe

        env_info["mediapipe_version"] = getattr(mediapipe, "__version__", "unknown")
    except ImportError:
        pass

    return env_info


@dataclass
class ExperimentManifest:
    """Structured, reproducible artifact manifest for sign language model experiments."""

    experiment_id: str
    timestamp_utc: str
    git_commit_sha: str
    git_dirty: bool
    dataset_checksum: str
    active_vocabulary: list[str]
    signer_partitions: dict[str, list[str]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to a JSON-compatible dictionary."""
        return {
            "schema_version": "1.0.0",
            "experiment_id": self.experiment_id,
            "timestamp_utc": self.timestamp_utc,
            "git_commit_sha": self.git_commit_sha,
            "git_dirty": self.git_dirty,
            "dataset_checksum": self.dataset_checksum,
            "active_vocabulary": self.active_vocabulary,
            "signer_partitions": self.signer_partitions,
            "metrics": self.metrics,
            "runtime": self.runtime,
            "environment": self.environment,
            "model_metadata": self.model_metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, output_path: str | Path) -> Path:
        """Save manifest to disk at the specified path.

        Args:
            output_path: Path where JSON manifest will be written.

        Returns:
            Resolved Path object.
        """
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=2))
        return p

    @classmethod
    def load(cls, manifest_path: str | Path) -> ExperimentManifest:
        """Load manifest from a JSON file.

        Args:
            manifest_path: Path to existing manifest JSON file.

        Returns:
            ExperimentManifest instance.
        """
        p = Path(manifest_path).resolve()
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            experiment_id=data.get("experiment_id", "unknown"),
            timestamp_utc=data.get("timestamp_utc", ""),
            git_commit_sha=data.get("git_commit_sha", "unknown"),
            git_dirty=data.get("git_dirty", False),
            dataset_checksum=data.get("dataset_checksum", ""),
            active_vocabulary=data.get("active_vocabulary", []),
            signer_partitions=data.get("signer_partitions", {}),
            metrics=data.get("metrics", {}),
            runtime=data.get("runtime", {}),
            environment=data.get("environment", {}),
            model_metadata=data.get("model_metadata", {}),
        )

    @classmethod
    def create(
        cls,
        experiment_id: str,
        active_vocabulary: list[str],
        metrics: dict[str, Any],
        runtime: dict[str, Any],
        dataset_checksum: str = "synthetic-seed-42",
        signer_partitions: dict[str, list[str]] | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> ExperimentManifest:
        """Convenience constructor populating Git provenance and environment metadata automatically.

        Args:
            experiment_id: Unique identifier for the experiment run.
            active_vocabulary: List of vocabulary sign names/glosses.
            metrics: Dictionary of evaluated classification metrics.
            runtime: Dictionary of latency and throughput benchmarks.
            dataset_checksum: Checksum of evaluation feature data.
            signer_partitions: Optional mapping of split names to signer IDs.
            model_metadata: Optional dictionary of model configuration/hyperparameters.

        Returns:
            Fully populated ExperimentManifest instance.
        """
        commit_sha, is_dirty = get_git_provenance()
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        env = get_sanitized_environment_info()

        return cls(
            experiment_id=experiment_id,
            timestamp_utc=now_utc,
            git_commit_sha=commit_sha,
            git_dirty=is_dirty,
            dataset_checksum=dataset_checksum,
            active_vocabulary=list(active_vocabulary),
            signer_partitions=signer_partitions or {},
            metrics=metrics,
            runtime=runtime,
            environment=env,
            model_metadata=model_metadata or {},
        )
