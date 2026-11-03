# Coding Standards & Python Guidelines

## Language & Environment
- Target: Python >= 3.10.
- Use explicit type hints on all public functions, arguments, and return types.
- Prefer Python 3.10+ native typing: `str | None` instead of `Optional[str]`, `list[int]` instead of `List[int]`, `tuple[int, ...]` instead of `Tuple[int, ...]`.
- Use `pathlib.Path` for all file path operations. Never concatenate path strings manually.
- Always open files with explicit `encoding="utf-8"`.

## Function & Class Design
- Prefer small, pure functions for mathematical and coordinate transformations.
- Maintain immutability where feasible; avoid modifying input arrays in-place unless explicitly requested and documented.
- All classifiers must inherit from `sanket.models.base.BaseSignClassifier`.
- Configuration schemas must inherit from `pydantic.BaseModel`.

## Formatting & Linting
- Maximum line length: 100 characters.
- Formatting must strictly pass `ruff check .` with zero errors or warnings.
- Clean import ordering: standard library first, third-party libraries second, internal `sanket.*` modules last.

