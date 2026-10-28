"""Configuration management models using Pydantic."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    import yaml

    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


class CameraConfig(BaseModel):
    """Configuration for video input and camera device capture."""

    device_id: int = Field(default=0, description="Video capture device index or camera ID")
    width: int = Field(default=640, description="Capture frame width")
    height: int = Field(default=480, description="Capture frame height")
    fps: int = Field(default=30, description="Target capture frames per second")


class LandmarkConfig(BaseModel):
    """Configuration for MediaPipe landmark extraction and tracking."""

    max_num_hands: int = Field(default=2, description="Maximum number of hands to detect")
    min_detection_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    normalize_coordinates: bool = Field(
        default=True, description="Enable scale and origin normalization"
    )


class ModelConfig(BaseModel):
    """Configuration for machine learning recognition models."""

    architecture: str = Field(
        default="random_forest", description="Baseline model architecture name"
    )
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    smoothing_window_size: int = Field(
        default=5, ge=1, description="Frames for rolling prediction smoothing"
    )
    weights_path: str | None = Field(default=None, description="Path to serialised model weights")


class AppConfig(BaseModel):
    """Master application configuration."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    landmarks: LandmarkConfig = Field(default_factory=LandmarkConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Lightweight fallback parser for basic nested key-value YAML files."""
    result: dict[str, Any] = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#")[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            result[current_section] = {}
        elif line.startswith("  ") and ":" in line and current_section:
            key, val_str = line.split(":", 1)
            key = key.strip()
            val_str = val_str.strip()
            val: Any
            if val_str in ("true", "True"):
                val = True
            elif val_str in ("false", "False"):
                val = False
            elif val_str in ("null", "None", "~", ""):
                val = None
            elif (val_str.startswith('"') and val_str.endswith('"')) or (
                val_str.startswith("'") and val_str.endswith("'")
            ):
                val = val_str[1:-1]
            else:
                try:
                    val = int(val_str) if "." not in val_str else float(val_str)
                except ValueError:
                    val = val_str
            result[current_section][key] = val

    return result


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from a YAML file or return default configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.
    """
    if config_path is None or not Path(config_path).exists():
        return AppConfig()

    text = Path(config_path).read_text(encoding="utf-8")
    if HAVE_YAML:
        data: dict[str, Any] = yaml.safe_load(text) or {}
    else:
        data = _parse_simple_yaml(text)

    return AppConfig(**data)
