"""CUB-200-2011 manifest builder for CBM swap experiments."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KOH_CLASS_ATTR_DATA_10_ZERO_BASED = [
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99,
    101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151,
    152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193,
    194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235,
    236, 238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274,
    277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311,
]
KOH_CLASS_ATTR_DATA_10_ATTRIBUTE_IDS = [
    attribute_idx + 1 for attribute_idx in KOH_CLASS_ATTR_DATA_10_ZERO_BASED
]


@dataclass(frozen=True)
class CubBuildResult:
    output_dir: str
    cub_root: str
    num_attributes: int
    num_classes: int
    class_ids: list[int]
    train_rows: int
    val_rows: int
    test_rows: int


def build_cub_manifest(
    *,
    cub_root: Path,
    output_dir: Path,
    num_attributes: int = 112,
    class_ids: list[int] | None = None,
    attribute_policy: str = "balanced",
    val_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, Any]:
    """Build train/val/test metadata manifests from a raw CUB_200_2011 folder."""

    _validate_cub_root(cub_root)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    images = _read_id_mapping(cub_root / "images.txt")
    class_labels = _read_int_mapping(cub_root / "image_class_labels.txt")
    train_flags = _read_int_mapping(cub_root / "train_test_split.txt")
    classes = _read_id_mapping(cub_root / "classes.txt")
    attributes = _read_id_mapping(_cub_attributes_path(cub_root))
    all_attr_labels, all_attr_certainties = _read_attribute_annotations(
        cub_root / "attributes" / "image_attribute_labels.txt"
    )
    attribute_ids = sorted(attributes)

    selected_class_ids = class_ids or sorted(set(class_labels.values()))
    selected_class_ids = sorted(selected_class_ids)
    class_to_label = {
        class_id: compact_label
        for compact_label, class_id in enumerate(selected_class_ids)
    }
    selected_image_ids = [
        image_id
        for image_id, class_id in class_labels.items()
        if class_id in class_to_label
    ]
    if not selected_image_ids:
        raise ValueError("Selected CUB class subset contains no images")

    train_image_ids = [
        image_id
        for image_id in selected_image_ids
        if train_flags[image_id] == 1
    ]
    test_image_ids = [
        image_id
        for image_id in selected_image_ids
        if train_flags[image_id] == 0
    ]
    class_level_attr_labels = _majority_vote_class_attributes(
        all_attr_labels=all_attr_labels,
        all_attr_certainties=all_attr_certainties,
        image_ids=train_image_ids,
        class_labels=class_labels,
        class_ids=selected_class_ids,
        attribute_ids=attribute_ids,
    )
    if attribute_policy == "balanced":
        selected_attribute_ids = _select_balanced_attributes(
            all_attr_labels=all_attr_labels,
            image_ids=train_image_ids,
            num_attributes=num_attributes,
            attribute_ids=attribute_ids,
        )
        row_attr_labels = all_attr_labels
        concept_encoding = "binary_image_attributes"
        attribute_policy_details = {
            "selection": "train_image_prevalence_closest_to_0.5",
            "labels": "image_level_binary_annotations",
        }
    elif attribute_policy == "koh112":
        selected_attribute_ids = _select_koh112_attributes(attribute_ids)
        row_attr_labels = {
            image_id: class_level_attr_labels[class_labels[image_id]]
            for image_id in selected_image_ids
        }
        concept_encoding = "binary_class_attr_data_10_koh112"
        attribute_policy_details = {
            "selection": "official_class_attr_data_10_attribute_indices",
            "labels": "class_level_majority_over_train_images",
            "min_class_count": 10,
            "tie_break": "present",
            "ignore_certainty_id": 1,
            "source": "Koh et al. ConceptBottleneck CUB/generate_new_data.py",
        }
    else:
        raise ValueError(
            f"Unsupported CUB attribute_policy {attribute_policy!r}; "
            "expected 'balanced' or 'koh112'"
        )
    concept_names = [
        _sanitize_concept_name(attributes[attribute_id])
        for attribute_id in selected_attribute_ids
    ]

    rng = random.Random(seed)
    train_ids_by_class: dict[int, list[int]] = defaultdict(list)
    for image_id in train_image_ids:
        train_ids_by_class[class_labels[image_id]].append(image_id)

    final_train_ids = []
    val_ids = []
    for ids in train_ids_by_class.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_fraction))
        val_ids.extend(ids[:n_val])
        final_train_ids.extend(ids[n_val:])

    output_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "dataset": "CUB_200_2011",
        "cub_root": str(cub_root),
        "concept_names": concept_names,
        "concept_attribute_ids": selected_attribute_ids,
        "concept_encoding": concept_encoding,
        "attribute_policy": attribute_policy,
        "attribute_policy_details": attribute_policy_details,
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
                "description": "CUB species classification over the selected class subset.",
            }
        },
    }

    for split_name, image_ids in {
        "train": sorted(final_train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_image_ids),
    }.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        _write_json(split_dir / "schema.json", schema)
        rows = [
            _row_for_image(
                cub_root=cub_root,
                image_id=image_id,
                images=images,
                class_labels=class_labels,
                class_to_label=class_to_label,
                attr_labels=row_attr_labels,
                selected_attribute_ids=selected_attribute_ids,
                concept_names=concept_names,
            )
            for image_id in image_ids
        ]
        _write_jsonl(split_dir / "metadata.jsonl", rows)

    result = CubBuildResult(
        output_dir=str(output_dir),
        cub_root=str(cub_root),
        num_attributes=len(selected_attribute_ids),
        num_classes=len(selected_class_ids),
        class_ids=selected_class_ids,
        train_rows=len(final_train_ids),
        val_rows=len(val_ids),
        test_rows=len(test_image_ids),
    )
    _write_json(output_dir / "manifest_summary.json", result.__dict__)
    return result.__dict__


def _row_for_image(
    *,
    cub_root: Path,
    image_id: int,
    images: dict[int, str],
    class_labels: dict[int, int],
    class_to_label: dict[int, int],
    attr_labels: dict[int, dict[int, int]],
    selected_attribute_ids: list[int],
    concept_names: list[str],
) -> dict[str, Any]:
    class_id = class_labels[image_id]
    return {
        "image_id": image_id,
        "image_path": str(cub_root / "images" / images[image_id]),
        "class_id": class_id,
        "class_name": images[image_id].split("/", 1)[0],
        "concept_names": concept_names,
        "concept_vector": [
            int(attr_labels[image_id].get(attribute_id, 0))
            for attribute_id in selected_attribute_ids
        ],
        "task_labels": {
            "species": class_to_label[class_id],
        },
    }


def _select_balanced_attributes(
    *,
    all_attr_labels: dict[int, dict[int, int]],
    image_ids: list[int],
    num_attributes: int,
    attribute_ids: list[int],
) -> list[int]:
    if num_attributes <= 0:
        raise ValueError("num_attributes must be positive")
    prevalence = []
    for attribute_id in attribute_ids:
        values = [
            all_attr_labels[image_id].get(attribute_id, 0)
            for image_id in image_ids
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        prevalence.append((abs(mean - 0.5), attribute_id))
    prevalence.sort()
    return sorted(attribute_id for _, attribute_id in prevalence[:num_attributes])


def _majority_vote_class_attributes(
    *,
    all_attr_labels: dict[int, dict[int, int]],
    all_attr_certainties: dict[int, dict[int, int]],
    image_ids: list[int],
    class_labels: dict[int, int],
    class_ids: list[int],
    attribute_ids: list[int],
) -> dict[int, dict[int, int]]:
    image_ids_by_class: dict[int, list[int]] = defaultdict(list)
    for image_id in image_ids:
        image_ids_by_class[class_labels[image_id]].append(image_id)

    class_level: dict[int, dict[int, int]] = {}
    for class_id in class_ids:
        class_image_ids = image_ids_by_class[class_id]
        if not class_image_ids:
            raise ValueError(f"Class {class_id} has no training images for majority vote")
        class_level[class_id] = {}
        for attribute_id in attribute_ids:
            counts = [0, 0]
            for image_id in class_image_ids:
                label = all_attr_labels[image_id].get(attribute_id, 0)
                certainty = all_attr_certainties[image_id].get(attribute_id, 0)
                if label == 0 and certainty == 1:
                    continue
                counts[label] += 1
            class_level[class_id][attribute_id] = 1 if counts[1] >= counts[0] else 0
    return class_level


def _select_koh112_attributes(attribute_ids: list[int]) -> list[int]:
    available = set(attribute_ids)
    return [
        attribute_id
        for attribute_id in KOH_CLASS_ATTR_DATA_10_ATTRIBUTE_IDS
        if attribute_id in available
    ]


def _read_attribute_annotations(
    path: Path,
) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, int]]]:
    labels: dict[int, dict[int, int]] = defaultdict(dict)
    certainties: dict[int, dict[int, int]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        image_id_text, attr_id_text, label_text, certainty_text, *_ = line.split()
        image_id = int(image_id_text)
        attribute_id = int(attr_id_text)
        labels[image_id][attribute_id] = int(label_text)
        certainties[image_id][attribute_id] = int(certainty_text)
    return dict(labels), dict(certainties)


def _read_id_mapping(path: Path) -> dict[int, str]:
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item_id, value = line.split(maxsplit=1)
        mapping[int(item_id)] = value
    return mapping


def _read_int_mapping(path: Path) -> dict[int, int]:
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item_id, value = line.split(maxsplit=1)
        mapping[int(item_id)] = int(value)
    return mapping


def _sanitize_concept_name(name: str) -> str:
    return (
        name.replace("::", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _validate_cub_root(cub_root: Path) -> None:
    required = [
        cub_root / "images.txt",
        cub_root / "image_class_labels.txt",
        cub_root / "train_test_split.txt",
        cub_root / "classes.txt",
        cub_root / "attributes" / "image_attribute_labels.txt",
        cub_root / "images",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if not (cub_root / "attributes" / "attributes.txt").exists() and not (
        cub_root / "attributes.txt"
    ).exists():
        missing.append(str(cub_root / "attributes" / "attributes.txt"))
    if missing:
        raise FileNotFoundError(f"CUB root is missing required files: {missing}")


def _cub_attributes_path(cub_root: Path) -> Path:
    nested = cub_root / "attributes" / "attributes.txt"
    if nested.exists():
        return nested
    root_level = cub_root / "attributes.txt"
    if root_level.exists():
        return root_level
    raise FileNotFoundError(
        f"CUB root is missing required file: {nested} or {root_level}"
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
