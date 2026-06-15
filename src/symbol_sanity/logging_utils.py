"""Shared progress logging for symbol-sanity experiments."""

from __future__ import annotations

import os


def log(message: str) -> None:
    """Print a prefixed progress line unless ``SYMBOL_SANITY_QUIET=1``."""

    if os.environ.get("SYMBOL_SANITY_QUIET") == "1":
        return
    print(f"[symbol_sanity] {message}", flush=True)