"""Deterministic synthetic colored-shape data for CBM sanity checks."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from symbol_sanity.schemas import DEFAULT_CONCEPT_SCHEMA, TASK_SCHEMAS


PALETTE = {
    "red": (220, 56, 45),
    "blue": (45, 99, 220),
    "green": (54, 160, 90),
}

BACKGROUND = (246, 244, 235)
STROKE = (30, 30, 30)


@dataclass(frozen=True)
class SyntheticExample:
    index: int
    seed: int
    attributes: dict[str, str]
    concept_vector: list[int]
    task_labels: dict[str, int]

    def to_metadata(self, image_path: str) -> dict[str, object]:
        row = asdict(self)
        row["image_path"] = image_path
        row["concept_names"] = list(DEFAULT_CONCEPT_SCHEMA.names)
        return row


def make_example(index: int, seed: int, tasks: Iterable[str] | None = None) -> SyntheticExample:
    """Create one deterministic symbolic example."""

    rng = random.Random(f"{seed}:{index}")
    schema = DEFAULT_CONCEPT_SCHEMA
    attributes = {
        group: rng.choice(values)
        for group, values in schema.groups.items()
    }
    concept_vector = schema.encode(attributes)
    requested_tasks = list(tasks) if tasks is not None else list(TASK_SCHEMAS)
    task_labels = {
        task_name: TASK_SCHEMAS[task_name].label(attributes)
        for task_name in requested_tasks
    }
    return SyntheticExample(
        index=index,
        seed=seed,
        attributes=attributes,
        concept_vector=concept_vector,
        task_labels=task_labels,
    )


def render_example(example: SyntheticExample, image_size: int = 64) -> Image.Image:
    """Render an example as a simple RGB image."""

    image = Image.new("RGB", (image_size, image_size), BACKGROUND)
    draw = ImageDraw.Draw(image)

    margin = image_size // 8
    center_x = image_size // 3 if example.attributes["position"] == "left" else 2 * image_size // 3
    center_y = image_size // 2
    radius = image_size // 7 if example.attributes["size"] == "small" else image_size // 5
    fill = PALETTE[example.attributes["color"]]
    shape = example.attributes["shape"]

    if shape == "circle":
        box = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ]
        draw.ellipse(box, fill=fill, outline=STROKE, width=2)
    elif shape == "square":
        box = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ]
        draw.rectangle(box, fill=fill, outline=STROKE, width=2)
    elif shape == "triangle":
        points = [
            (center_x, center_y - radius - margin // 2),
            (center_x - radius, center_y + radius),
            (center_x + radius, center_y + radius),
        ]
        draw.polygon(points, fill=fill, outline=STROKE)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return image


def export_dataset(
    output_dir: Path,
    num_examples: int,
    seed: int,
    image_size: int = 64,
    tasks: Iterable[str] | None = None,
) -> None:
    """Export images plus metadata JSONL for downstream CBM training."""

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    requested_tasks = list(tasks) if tasks is not None else list(TASK_SCHEMAS)

    schema_path = output_dir / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "concept_names": list(DEFAULT_CONCEPT_SCHEMA.names),
                "concept_groups": DEFAULT_CONCEPT_SCHEMA.groups,
                "tasks": {
                    name: {
                        "num_classes": TASK_SCHEMAS[name].num_classes,
                        "description": TASK_SCHEMAS[name].description,
                    }
                    for name in requested_tasks
                },
                "concept_encoding": "binary_one_hot_groups",
                "detector_head_interface": "concept_vector_binary",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for index in range(num_examples):
            example = make_example(index=index, seed=seed, tasks=requested_tasks)
            image_name = f"{index:06d}.png"
            image_path = image_dir / image_name
            render_example(example, image_size=image_size).save(image_path)
            row = example.to_metadata(image_path=f"images/{image_name}")
            metadata_file.write(json.dumps(row, sort_keys=True) + "\n")

