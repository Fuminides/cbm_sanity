"""Concept and task schemas for the first symbol-sanity experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ConceptSchema:
    """Fixed concept order and encoding metadata.

    The order here is part of the detector/head interface. Changing it invalidates
    existing checkpoints unless a migration records the old and new order.
    """

    names: tuple[str, ...]
    groups: dict[str, tuple[str, ...]]

    def encode(self, attributes: dict[str, str]) -> list[int]:
        active = {
            f"shape_{attributes['shape']}",
            f"color_{attributes['color']}",
            f"size_{attributes['size']}",
            f"position_{attributes['position']}",
        }
        return [1 if name in active else 0 for name in self.names]

    def validate_vector(self, vector: list[int]) -> None:
        if len(vector) != len(self.names):
            raise ValueError(
                f"Expected {len(self.names)} concepts, got {len(vector)}"
            )
        for value in vector:
            if value not in (0, 1):
                raise ValueError(f"Concept vectors must be binary, got {value!r}")


DEFAULT_CONCEPT_SCHEMA = ConceptSchema(
    names=(
        "shape_circle",
        "shape_square",
        "shape_triangle",
        "color_red",
        "color_blue",
        "color_green",
        "size_small",
        "size_large",
        "position_left",
        "position_right",
    ),
    groups={
        "shape": ("circle", "square", "triangle"),
        "color": ("red", "blue", "green"),
        "size": ("small", "large"),
        "position": ("left", "right"),
    },
)


TaskFunction = Callable[[dict[str, str]], int]


@dataclass(frozen=True)
class TaskSchema:
    name: str
    num_classes: int
    description: str
    label_fn: TaskFunction

    def label(self, attributes: dict[str, str]) -> int:
        label = self.label_fn(attributes)
        if not 0 <= label < self.num_classes:
            raise ValueError(
                f"Task {self.name!r} produced label {label}, "
                f"expected [0, {self.num_classes})"
            )
        return label


def _task_shape_color(attributes: dict[str, str]) -> int:
    shape = attributes["shape"]
    color = attributes["color"]
    if shape == "circle" and color == "red":
        return 0
    if shape == "circle" and color in {"blue", "green"}:
        return 1
    if shape == "square" and color == "red":
        return 2
    if shape == "square" and color in {"blue", "green"}:
        return 3
    return 4


def _task_shape_position(attributes: dict[str, str]) -> int:
    shape = attributes["shape"]
    position = attributes["position"]
    if shape == "circle" and position == "left":
        return 0
    if shape == "circle" and position == "right":
        return 1
    if shape == "square" and position == "left":
        return 2
    if shape == "square" and position == "right":
        return 3
    return 4


def _task_color_size(attributes: dict[str, str]) -> int:
    color = attributes["color"]
    size = attributes["size"]
    if color == "red" and size == "small":
        return 0
    if color == "red" and size == "large":
        return 1
    if color == "blue" and size == "small":
        return 2
    if color == "blue" and size == "large":
        return 3
    return 4


def _task_mixed(attributes: dict[str, str]) -> int:
    shape = attributes["shape"]
    color = attributes["color"]
    size = attributes["size"]
    position = attributes["position"]
    if shape == "triangle" and color == "green":
        return 0
    if size == "large" and position == "left":
        return 1
    if shape == "square" and size == "small":
        return 2
    if color == "blue" and position == "right":
        return 3
    return 4


TASK_SCHEMAS: dict[str, TaskSchema] = {
    "shape_color": TaskSchema(
        name="shape_color",
        num_classes=5,
        description="Five classes based on shape and color combinations.",
        label_fn=_task_shape_color,
    ),
    "shape_position": TaskSchema(
        name="shape_position",
        num_classes=5,
        description="Five classes based on shape and left/right position.",
        label_fn=_task_shape_position,
    ),
    "color_size": TaskSchema(
        name="color_size",
        num_classes=5,
        description="Five classes based on color and size combinations.",
        label_fn=_task_color_size,
    ),
    "mixed": TaskSchema(
        name="mixed",
        num_classes=5,
        description="Five classes based on overlapping shape, color, size, and position rules.",
        label_fn=_task_mixed,
    ),
}

