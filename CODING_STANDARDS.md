# SANKET (संकेत) — Engineering & Coding Standards

> **Authoritative Python Style, Typing, and Software Craftsmanship Guidelines**  
> *Enforced across all modules, tests, scripts, and documentation in SANKET.*

---

## 1. Core Principles

1. **Explicit Over Implicit**: Explicit type annotations, parameter defaults, and variable naming.
2. **Defensive & Deterministic**: Enforce seed locking (`random_state=42`) and validate input shapes, coordinate bounds, and file paths.
3. **Pure Function Preference**: Isolate mathematical transformations (`geometry.py`, `normalization.py`) as pure functions without global state.
4. **Vectorized Computation**: Utilize NumPy array slicing, matrix multiplication, and broadcasting instead of nested Python loops.
5. **Zero-Biometric Persistence**: Video frames and biometric media must remain in volatile memory and never touch disk storage.

---

## 2. Python 3.10+ Language & Typing Standards

### Modern Type Annotations
All function signatures and class attributes must be explicitly typed using standard modern Python 3.10+ typing syntax:

```python
# DO: Use native union syntax and built-in collection types
def normalize_landmarks(
    landmarks: np.ndarray,
    scale_factor: float | None = None,
) -> tuple[np.ndarray, float]:
    """Normalize 3D coordinates relative to wrist origin and palm scale."""
    ...

# DON'T: Use deprecated typing module imports for standard containers
from typing import List, Optional, Tuple  # Avoid in new code
```

### Static Type Safety
- Never use untyped `*args` or `**kwargs` when signature parameters can be explicitly enumerated.
- Avoid casting to `Any` unless wrapping dynamic third-party library internals (e.g., MediaPipe protobuf wrappers).

---

## 3. Documentation & Docstring Standards

All public classes, methods, and functions must have Google-style docstrings:

```python
class RandomForestSignClassifier(BaseSignClassifier):
    """Random Forest classifier for isolated sign language recognition.

    Implements the BaseSignClassifier contract using an ensemble of decision trees
    evaluated on 645-dimensional geometric and temporal landmark features.

    Attributes:
        n_estimators: Number of decision trees in the forest.
        max_depth: Maximum tree depth to constrain overfitting.
        random_state: Pseudo-random seed for deterministic training.
        is_fitted: Boolean flag indicating if the model has completed training.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestSignClassifier":
        """Train the Random Forest estimator on extracted feature vectors.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Target gloss labels of shape (n_samples,).

        Returns:
            Fitted instance of self for method chaining.

        Raises:
            ValueError: If X and y have mismatched sample dimensions or if
                feature dimensions do not match the expected 645-dim contract.
        """
```

---

## 4. Error Handling & Custom Exceptions

- Never write bare `except:` clauses or suppress exceptions with `pass`.
- Use custom domain exceptions inheriting from standard Python base exceptions:

```python
class SanketError(Exception):
    """Base exception for all SANKET runtime and validation errors."""

class LandmarkExtractionError(SanketError):
    """Raised when MediaPipe fails to detect required anatomical keypoints."""

class IncompatibleFeatureShapeError(SanketError):
    """Raised when feature vector dimensions do not match the expected 645-dim topology."""
```

---

## 5. Path Manipulation & Cross-Platform File I/O

- Always use `pathlib.Path` for file system paths. Never perform raw string concatenation (`path + "/" + file` or `path + "\\" + file`).
- Open files with explicit UTF-8 encoding: `open(path, "r", encoding="utf-8")`.
- **Absolute Privacy Rule**: Never hardcode or print personal machine paths (e.g., `C:\Users\...`). Use repository-relative paths:

```python
from pathlib import Path

# DO: Portable, relative path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"
```

---

## 6. Linter & Formatter Configuration

- **Ruff**: Configured in `pyproject.toml` (`line-length = 100`, target `py310`).
  - Active rules: `E` (Pycodestyle errors), `F` (Pyflakes), `W` (Warnings), `I` (isort import ordering), `N` (naming conventions), `UP` (pyupgrade), `B` (flake8-bugbear).
- Run `ruff check .` before staging any commit.

