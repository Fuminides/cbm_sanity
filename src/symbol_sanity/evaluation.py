"""Evaluation routines for original and swapped CBM components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from symbol_sanity.detectors import load_detector, predict_concepts
from symbol_sanity.heads import load_head, predict_label
from symbol_sanity.io import load_dataset, require_concept_names, write_json
from symbol_sanity.metrics import accuracy, concept_agreement, macro_f1


def evaluate_detector_head(
    dataset_dir: Path,
    detector_path: Path,
    head_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    schema, rows = load_dataset(dataset_dir)
    detector = load_detector(detector_path)
    head = load_head(head_path)

    concept_names = list(schema["concept_names"])
    require_concept_names(list(detector["concept_names"]), concept_names)
    require_concept_names(list(head["concept_names"]), concept_names)

    task_name = head["task_name"]
    if task_name not in schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in dataset schema")

    y_true: list[int] = []
    y_pred: list[int] = []
    predicted_concepts: list[list[int]] = []
    oracle_concepts: list[list[int]] = []

    for row_index, row in enumerate(rows):
        concepts = predict_concepts(detector, row, row_index=row_index)
        prediction = predict_label(head, concepts)
        y_true.append(int(row["task_labels"][task_name]))
        y_pred.append(prediction)
        predicted_concepts.append(concepts)
        oracle_concepts.append(list(row["concept_vector"]))

    result = {
        "dataset_dir": str(dataset_dir),
        "detector": detector["name"],
        "detector_type": detector["detector_type"],
        "head": head["name"],
        "task_name": task_name,
        "num_examples": len(rows),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, int(head["num_classes"])),
        "concept_agreement_with_oracle": concept_agreement(predicted_concepts, oracle_concepts),
    }

    if output_path is not None:
        write_json(output_path, result)
    return result


def evaluate_swap(
    dataset_dir: Path,
    original_detector_path: Path,
    swap_detector_path: Path,
    head_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    original = evaluate_detector_head(dataset_dir, original_detector_path, head_path)
    swapped = evaluate_detector_head(dataset_dir, swap_detector_path, head_path)

    result = {
        "dataset_dir": str(dataset_dir),
        "head": original["head"],
        "task_name": original["task_name"],
        "original_detector": original["detector"],
        "swap_detector": swapped["detector"],
        "num_examples": original["num_examples"],
        "original_accuracy": original["accuracy"],
        "swapped_accuracy": swapped["accuracy"],
        "swap_drop": original["accuracy"] - swapped["accuracy"],
        "relative_retention": (
            0.0 if original["accuracy"] == 0 else swapped["accuracy"] / original["accuracy"]
        ),
        "original_concept_agreement_with_oracle": original["concept_agreement_with_oracle"],
        "swap_concept_agreement_with_oracle": swapped["concept_agreement_with_oracle"],
    }

    if output_path is not None:
        write_json(output_path, result)
    return result

