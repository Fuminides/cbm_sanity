"""Concept-head models."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from symbol_sanity.checkpoints import load_checkpoint, save_head_checkpoint
from symbol_sanity.io import require_concept_names


def _concept_key(vector: list[int]) -> str:
    return "".join(str(value) for value in vector)


def train_lookup_head(
    path: Path,
    *,
    name: str,
    task_name: str,
    concept_names: list[str],
    num_classes: int,
    rows: list[dict[str, Any]],
) -> None:
    """Train a majority-label lookup head over binary concept vectors."""

    counts: dict[str, Counter[int]] = defaultdict(Counter)
    global_counts: Counter[int] = Counter()

    for row in rows:
        require_concept_names(list(row["concept_names"]), concept_names)
        vector = list(row["concept_vector"])
        label = int(row["task_labels"][task_name])
        counts[_concept_key(vector)][label] += 1
        global_counts[label] += 1

    if not global_counts:
        raise ValueError("Cannot train head on an empty dataset")

    default_label = _most_common_label(global_counts)
    table = {
        key: _most_common_label(label_counts)
        for key, label_counts in counts.items()
    }
    save_head_checkpoint(
        path,
        name=name,
        task_name=task_name,
        concept_names=concept_names,
        num_classes=num_classes,
        model_type="majority_lookup",
        payload={
            "default_label": default_label,
            "table": table,
        },
    )


def predict_label(head_checkpoint: dict[str, Any], concept_vector: list[int]) -> int:
    if len(concept_vector) != len(head_checkpoint["concept_names"]):
        raise ValueError(
            "Concept vector length mismatch. "
            f"Expected {len(head_checkpoint['concept_names'])}, got {len(concept_vector)}"
        )
    if head_checkpoint["model_type"] != "majority_lookup":
        raise ValueError(f"Unsupported head model: {head_checkpoint['model_type']!r}")

    payload = head_checkpoint["payload"]
    key = _concept_key(concept_vector)
    return int(payload["table"].get(key, payload["default_label"]))


def load_head(path: Path) -> dict[str, Any]:
    return load_checkpoint(path, expected_component="head")


def _most_common_label(counts: Counter[int]) -> int:
    max_count = max(counts.values())
    return min(label for label, count in counts.items() if count == max_count)

