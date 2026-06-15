"""Checkpoint formats for detector/head swap experiments.

These JSON checkpoints are deliberately simple. They define the detector-head
interface now and can later point to neural model artifacts from the official CBM
implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from symbol_sanity.io import read_json, write_json


CHECKPOINT_VERSION = 1


def save_detector_checkpoint(
    path: Path,
    *,
    name: str,
    concept_names: list[str],
    detector_type: str,
    payload: dict[str, Any] | None = None,
) -> None:
    write_json(
        path,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "component": "detector",
            "name": name,
            "detector_type": detector_type,
            "concept_names": concept_names,
            "interface": "concept_vector_binary",
            "payload": payload or {},
        },
    )


def save_head_checkpoint(
    path: Path,
    *,
    name: str,
    task_name: str,
    concept_names: list[str],
    num_classes: int,
    model_type: str,
    payload: dict[str, Any],
) -> None:
    write_json(
        path,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "component": "head",
            "name": name,
            "task_name": task_name,
            "num_classes": num_classes,
            "model_type": model_type,
            "concept_names": concept_names,
            "interface": "concept_vector_binary",
            "payload": payload,
        },
    )


def load_checkpoint(path: Path, expected_component: str) -> dict[str, Any]:
    checkpoint = read_json(path)
    component = checkpoint.get("component")
    if component != expected_component:
        raise ValueError(
            f"Expected {expected_component!r} checkpoint at {path}, got {component!r}"
        )
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {checkpoint.get('checkpoint_version')!r}"
        )
    if checkpoint.get("interface") != "concept_vector_binary":
        raise ValueError(f"Unsupported interface: {checkpoint.get('interface')!r}")
    return checkpoint

