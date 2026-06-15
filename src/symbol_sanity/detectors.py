"""Concept detector implementations for swap-protocol tests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from symbol_sanity.checkpoints import load_checkpoint, save_detector_checkpoint
from symbol_sanity.io import require_concept_names


def create_oracle_detector(path: Path, *, name: str, concept_names: list[str]) -> None:
    save_detector_checkpoint(
        path,
        name=name,
        concept_names=concept_names,
        detector_type="oracle_metadata",
    )


def create_noisy_detector(
    path: Path,
    *,
    name: str,
    concept_names: list[str],
    flip_probability: float,
    seed: int,
) -> None:
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError("flip_probability must be between 0 and 1")
    save_detector_checkpoint(
        path,
        name=name,
        concept_names=concept_names,
        detector_type="noisy_oracle_metadata",
        payload={
            "flip_probability": flip_probability,
            "seed": seed,
        },
    )


def predict_concepts(
    detector_checkpoint: dict[str, Any],
    row: dict[str, Any],
    *,
    row_index: int,
) -> list[int]:
    concept_names = list(detector_checkpoint["concept_names"])
    require_concept_names(list(row["concept_names"]), concept_names)
    concept_vector = list(row["concept_vector"])

    detector_type = detector_checkpoint["detector_type"]
    if detector_type == "oracle_metadata":
        return concept_vector

    if detector_type == "noisy_oracle_metadata":
        payload = detector_checkpoint["payload"]
        rng = random.Random(f"{payload['seed']}:{row_index}:{row['image_path']}")
        flip_probability = float(payload["flip_probability"])
        return [
            1 - value if rng.random() < flip_probability else value
            for value in concept_vector
        ]

    raise ValueError(f"Unsupported detector type: {detector_type!r}")


def load_detector(path: Path) -> dict[str, Any]:
    return load_checkpoint(path, expected_component="detector")

