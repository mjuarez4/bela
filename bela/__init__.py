from pathlib import Path

from . import common  # noqa: F401
from .batchspec import DEFAULT_SPEC, SIMPLE_SPEC, BatchSpec

ROOT = Path(__file__).resolve().parent.parent

__all__ = ["BatchSpec", "DEFAULT_SPEC", "SIMPLE_SPEC"]
