"""Helpers for creating non-destructive, repository-scoped run directories."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT / "output").resolve()


def _canonical_path(path: Path) -> Path:
    """Canonicalize a path, while tolerating inaccessible Windows temp paths."""
    try:
        return path.resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(os.fspath(path)))


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository, independent of cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _reject_protected_path(path: Path) -> Path:
    resolved = _canonical_path(path)
    protected = {_canonical_path(PROJECT_ROOT), OUTPUT_ROOT}
    if resolved in protected:
        raise ValueError(
            f"Refusing to use protected directory as a run directory: {resolved}. "
            "Pass a child directory such as output/GME/<experiment>/<run_id>."
        )
    return resolved


def prepare_explicit_run_dir(path: str | Path) -> Path:
    """Create an explicitly requested run directory without overwriting data.

    A pre-existing non-empty directory is rejected. This catches accidental
    reuse of ``output`` or an earlier completed run before any artifacts are
    written.
    """
    resolved = _reject_protected_path(resolve_project_path(path))
    if resolved.exists():
        if not resolved.is_dir():
            raise NotADirectoryError(f"Run path exists but is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty run directory: {resolved}. "
                "Choose a new run directory."
            )
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.mkdir()
    return resolved


def allocate_run_dir(base_dir: str | Path, experiment_name: str) -> Path:
    """Allocate a unique timestamped run directory atomically."""
    base = _canonical_path(resolve_project_path(base_dir))
    # The base output directory is allowed; only the allocated run directory
    # is protected from reuse.
    base.mkdir(parents=True, exist_ok=True)

    experiment = str(experiment_name).strip()
    if not experiment or experiment in {".", ".."}:
        raise ValueError(f"Invalid experiment name: {experiment_name!r}")
    parent = base / experiment
    parent.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = parent / stamp
    suffix = 0
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = parent / f"{stamp}_{suffix}"
