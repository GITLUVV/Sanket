"""Foundational unit tests verifying package imports, versioning, and configs."""

from pathlib import Path
import pytest
import sanket
from sanket.core.config import AppConfig, load_config


def test_package_metadata():
    """Verify package version, author, and license metadata."""
    assert sanket.__version__ == "0.1.0"
    assert sanket.__author__ == "Anveshna"
    assert sanket.__license__ == "Apache-2.0"


def test_default_config_instantiation():
    """Verify that default AppConfig instantiates with valid parameters."""
    config = AppConfig()
    assert config.camera.width == 640
    assert config.camera.height == 480
    assert config.camera.fps == 30
    assert config.landmarks.max_num_hands == 2
    assert config.landmarks.min_detection_confidence == 0.7
    assert config.model.architecture == "random_forest"
    assert config.model.confidence_threshold == 0.75


def test_yaml_config_loading(tmp_path):
    """Verify that YAML configuration file loads and overrides defaults."""
    custom_yaml = tmp_path / "custom_config.yaml"
    custom_yaml.write_text(
        """
camera:
  device_id: 1
  width: 1280
  height: 720
  fps: 60
landmarks:
  max_num_hands: 1
  min_detection_confidence: 0.85
model:
  architecture: "svm"
  confidence_threshold: 0.80
"""
    )
    loaded = load_config(custom_yaml)
    assert loaded.camera.device_id == 1
    assert loaded.camera.width == 1280
    assert loaded.camera.height == 720
    assert loaded.camera.fps == 60
    assert loaded.landmarks.max_num_hands == 1
    assert loaded.landmarks.min_detection_confidence == 0.85
    assert loaded.model.architecture == "svm"
    assert loaded.model.confidence_threshold == 0.80


def test_missing_config_falls_back():
    """Verify that a nonexistent file path cleanly falls back to defaults."""
    nonexistent = Path("/nonexistent/path/config.yaml")
    config = load_config(nonexistent)
    assert isinstance(config, AppConfig)
    assert config.camera.fps == 30
