"""Dataset IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = read_json(dataset_dir / "schema.json")
    rows = read_jsonl(dataset_dir / "metadata.jsonl")
    return schema, rows


def require_concept_names(actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        raise ValueError(
            "Concept schema mismatch. "
            f"Expected {expected!r}, got {actual!r}"
        )

