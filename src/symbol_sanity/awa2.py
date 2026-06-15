"""Animals with Attributes 2 manifest builder for CBM swap experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Awa2BuildResult:
    output_dir: str
    awa2_root: str
    num_attributes: int
    num_classes: int
    class_ids: list[int]
    train_rows: int
    val_rows: int
    test_rows: int


def build_awa2_manifest(
    *,
    awa2_root: Path,
    output_dir: Path,
    class_ids: list[int] | None = None,
    num_classes: int | None = None,
    class_start: int = 1,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
    attribute_kind: str = "binary",
    continuous_threshold: float = 50.0,
) -> dict[str, Any]:
    """Build train/val/test metadata manifests from an AwA2 folder.

    AwA2 attributes are class-level predicates. The same concept vector is
    assigned to every image of a class, which is appropriate for class-split
    transfer experiments but weaker than CUB's image-level annotations.
    """

    _validate_awa2_root(awa2_root)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")
    if attribute_kind not in {"binary", "continuous-threshold"}:
        raise ValueError(
            "attribute_kind must be 'binary' or 'continuous-threshold'"
        )

    classes = _read_awa2_classes(awa2_root)
    predicates = _read_awa2_predicates(awa2_root)
    selected_class_ids = _select_classes(
        all_class_ids=sorted(classes),
        class_ids=class_ids,
        num_classes=num_classes,
        class_start=class_start,
    )
    class_to_label = {
        class_id: compact_label
        for compact_label, class_id in enumerate(selected_class_ids)
    }
    class_attribute_vectors = _read_class_attribute_vectors(
        awa2_root=awa2_root,
        class_ids=selected_class_ids,
        attribute_kind=attribute_kind,
        continuous_threshold=continuous_threshold,
    )
    concept_names = [
        _sanitize_concept_name(predicates[predicate_id])
        for predicate_id in sorted(predicates)
    ]

    image_paths_by_class = {
        class_id: _image_paths_for_class(awa2_root, classes[class_id])
        for class_id in selected_class_ids
    }
    empty_classes = [
        classes[class_id]
        for class_id, image_paths in image_paths_by_class.items()
        if not image_paths
    ]
    if empty_classes:
        raise ValueError(f"Selected AwA2 classes contain no images: {empty_classes}")

    rng = random.Random(seed)
    split_ids: dict[str, list[tuple[int, Path]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for class_id, image_paths in image_paths_by_class.items():
        shuffled = sorted(image_paths)
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * test_fraction))
        n_val = int(round(len(shuffled) * val_fraction))
        split_ids["test"].extend((class_id, path) for path in shuffled[:n_test])
        split_ids["val"].extend(
            (class_id, path) for path in shuffled[n_test : n_test + n_val]
        )
        split_ids["train"].extend(
            (class_id, path) for path in shuffled[n_test + n_val :]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "dataset": "AwA2",
        "awa2_root": str(awa2_root),
        "concept_names": concept_names,
        "concept_attribute_ids": sorted(predicates),
        "concept_encoding": f"{attribute_kind}_class_level_attributes",
        "attribute_policy": attribute_kind,
        "attribute_policy_details": {
            "labels": "class_level_attributes_replicated_to_images",
            "source": "AwA2 predicate matrix",
            "continuous_threshold": continuous_threshold
            if attribute_kind == "continuous-threshold"
            else None,
        },
        "detector_head_interface": "concept_vector_binary",
        "classes": {
            str(class_id): classes[class_id]
            for class_id in selected_class_ids
        },
        "class_to_label": {
            str(class_id): label
            for class_id, label in class_to_label.items()
        },
        "tasks": {
            "species": {
                "num_classes": len(selected_class_ids),
                "description": "AwA2 animal species classification over selected classes.",
            }
        },
    }

    for split_name, items in split_ids.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        _write_json(split_dir / "schema.json", schema)
        rows = [
            _row_for_image(
                image_index=index,
                image_path=image_path,
                class_id=class_id,
                class_name=classes[class_id],
                class_to_label=class_to_label,
                concept_names=concept_names,
                concept_vector=class_attribute_vectors[class_id],
            )
            for index, (class_id, image_path) in enumerate(sorted(items), start=1)
        ]
        _write_jsonl(split_dir / "metadata.jsonl", rows)

    result = Awa2BuildResult(
        output_dir=str(output_dir),
        awa2_root=str(awa2_root),
        num_attributes=len(concept_names),
        num_classes=len(selected_class_ids),
        class_ids=selected_class_ids,
        train_rows=len(split_ids["train"]),
        val_rows=len(split_ids["val"]),
        test_rows=len(split_ids["test"]),
    )
    _write_json(output_dir / "manifest_summary.json", result.__dict__)
    return result.__dict__


def _row_for_image(
    *,
    image_index: int,
    image_path: Path,
    class_id: int,
    class_name: str,
    class_to_label: dict[int, int],
    concept_names: list[str],
    concept_vector: list[int],
) -> dict[str, Any]:
    return {
        "image_id": image_index,
        "image_path": str(image_path),
        "class_id": class_id,
        "class_name": class_name,
        "concept_names": concept_names,
        "concept_vector": concept_vector,
        "task_labels": {
            "species": class_to_label[class_id],
        },
    }


def _select_classes(
    *,
    all_class_ids: list[int],
    class_ids: list[int] | None,
    num_classes: int | None,
    class_start: int,
) -> list[int]:
    if class_ids is not None:
        selected = sorted(class_ids)
    elif num_classes is not None:
        selected = list(range(class_start, class_start + num_classes))
    else:
        selected = list(all_class_ids)
    missing = sorted(set(selected) - set(all_class_ids))
    if missing:
        raise ValueError(f"Unknown AwA2 class ids: {missing}")
    if not selected:
        raise ValueError("Selected AwA2 class subset is empty")
    return selected


def _read_class_attribute_vectors(
    *,
    awa2_root: Path,
    class_ids: list[int],
    attribute_kind: str,
    continuous_threshold: float,
) -> dict[int, list[int]]:
    if attribute_kind == "binary":
        matrix_path = _awa2_matrix_path(awa2_root, preferred="binary")
    else:
        matrix_path = _awa2_matrix_path(awa2_root, preferred="continuous")
    rows = _read_numeric_matrix(matrix_path)
    binary_fallback_from_continuous = (
        attribute_kind == "binary"
        and matrix_path.name == "predicate-matrix-continuous.txt"
    )
    continuous_fallback_from_binary = (
        attribute_kind == "continuous-threshold"
        and matrix_path.name == "predicate-matrix-binary.txt"
    )
    class_vectors = {}
    for class_id in class_ids:
        row = rows[class_id - 1]
        if attribute_kind == "binary":
            if binary_fallback_from_continuous:
                class_vectors[class_id] = [
                    1 if value >= continuous_threshold else 0 for value in row
                ]
            else:
                class_vectors[class_id] = [int(round(value)) for value in row]
        else:
            if continuous_fallback_from_binary:
                class_vectors[class_id] = [int(round(value)) for value in row]
            else:
                class_vectors[class_id] = [
                    1 if value >= continuous_threshold else 0 for value in row
                ]
    return class_vectors


def _read_numeric_matrix(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append([float(value) for value in line.split()])
    if not rows:
        raise ValueError(f"Empty AwA2 predicate matrix: {path}")
    return rows


def _image_paths_for_class(awa2_root: Path, class_name: str) -> list[Path]:
    image_root = _awa2_image_root(awa2_root)
    image_dir = image_root / class_name
    if not image_dir.exists():
        image_dir = image_root / class_name.replace("+", "_")
    if not image_dir.exists():
        return []
    return [
        path.resolve()
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _read_id_mapping(path: Path) -> dict[int, str]:
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item_id, value = line.split(maxsplit=1)
        mapping[int(item_id)] = value
    return mapping


def _read_awa2_classes(awa2_root: Path) -> dict[int, str]:
    classes_path = awa2_root / "classes.txt"
    if classes_path.exists():
        return _read_id_mapping(classes_path)
    image_root = _awa2_image_root(awa2_root)
    class_names = sorted(
        path.name
        for path in image_root.iterdir()
        if path.is_dir()
    )
    if not class_names:
        raise FileNotFoundError(
            f"AwA2 root is missing classes.txt and has no class directories under {image_root}"
        )
    return {
        class_id: class_name
        for class_id, class_name in enumerate(class_names, start=1)
    }


def _read_awa2_predicates(awa2_root: Path) -> dict[int, str]:
    predicates_path = awa2_root / "predicates.txt"
    if predicates_path.exists():
        return _read_id_mapping(predicates_path)
    matrix_path = _awa2_matrix_path(awa2_root, preferred="binary")
    rows = _read_numeric_matrix(matrix_path)
    if not rows or not rows[0]:
        raise FileNotFoundError(
            f"AwA2 root is missing predicates.txt and cannot infer predicates from {matrix_path}"
        )
    return {
        predicate_id: f"predicate_{predicate_id}"
        for predicate_id in range(1, len(rows[0]) + 1)
    }


def _sanitize_concept_name(name: str) -> str:
    return (
        name.replace("::", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("+", "_")
    )


def _validate_awa2_root(awa2_root: Path) -> None:
    required = []
    missing = [str(path) for path in required if not path.exists()]
    try:
        _awa2_image_root(awa2_root)
    except FileNotFoundError as exc:
        missing.append(str(exc))
    if not (awa2_root / "predicate-matrix-binary.txt").exists() and not (
        awa2_root / "predicate-matrix-continuous.txt"
    ).exists():
        missing.append(str(awa2_root / "predicate-matrix-binary.txt"))
    if missing:
        raise FileNotFoundError(f"AwA2 root is missing required files: {missing}")


def _awa2_image_root(awa2_root: Path) -> Path:
    direct = awa2_root / "JPEGImages"
    if direct.exists():
        return direct
    candidates = sorted(
        path
        for path in awa2_root.glob("**/JPEGImages")
        if path.is_dir()
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"{awa2_root / 'JPEGImages'}")


def _awa2_matrix_path(awa2_root: Path, *, preferred: str) -> Path:
    binary_path = awa2_root / "predicate-matrix-binary.txt"
    continuous_path = awa2_root / "predicate-matrix-continuous.txt"
    if preferred == "binary":
        if binary_path.exists():
            return binary_path
        if continuous_path.exists():
            return continuous_path
    elif preferred == "continuous":
        if continuous_path.exists():
            return continuous_path
        if binary_path.exists():
            return binary_path
    else:
        raise ValueError(f"Unsupported AwA2 matrix preference: {preferred!r}")
    raise FileNotFoundError(
        f"AwA2 root is missing predicate matrix files: {binary_path} or {continuous_path}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
