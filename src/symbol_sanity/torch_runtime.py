"""Shared torch-backed runtime helpers for CBM experiments.

These helpers are used by both ``neural_synthetic`` and ``statistical_report``.
They live in their own module so report code does not have to reach into the
private API of the experiment runner. Importing this module is safe without
torch; the torch dependency is only required when the helpers are called.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from symbol_sanity.official_cbm import OfficialCBMSpec, build_official_cbm


def torch_module() -> Any:
    """Import torch lazily with a helpful error when it is missing."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Torch is required for neural CBM experiments. Install the "
            "dependencies from requirements.txt."
        ) from exc
    return torch


def make_loader(
    *,
    dataset_dir: Path,
    rows: list[dict[str, Any]],
    batch_size: int,
    shuffle: bool,
    seed: int,
    image_size: int,
    include_images: bool,
    task_name: str | None,
) -> Any:
    torch = torch_module()
    dataset = _SyntheticTorchDataset(
        dataset_dir=dataset_dir,
        rows=rows,
        image_size=image_size,
        include_images=include_images,
        task_name=task_name,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_workers = loader_num_workers() if include_images else 0
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def loader_num_workers() -> int:
    """Image-loader worker count, set via ``SYMBOL_SANITY_NUM_WORKERS``.

    Defaults to 0 (in-process loading) so tests and small CPU runs do not pay
    worker startup costs. Cluster guidance caps image loading at 4 workers, so
    higher environment overrides are clamped.
    """

    return min(4, max(0, int(os.environ.get("SYMBOL_SANITY_NUM_WORKERS", "0"))))


class _SyntheticTorchDataset:
    def __init__(
        self,
        *,
        dataset_dir: Path,
        rows: list[dict[str, Any]],
        image_size: int,
        include_images: bool,
        task_name: str | None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.rows = rows
        self.image_size = image_size
        self.include_images = include_images
        self.task_name = task_name

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = torch_module()
        row = self.rows[index]
        item = {
            "concept": torch.tensor(row["concept_vector"], dtype=torch.float32),
        }
        if self.task_name is not None:
            item["label"] = torch.tensor(
                int(row["task_labels"][self.task_name]),
                dtype=torch.long,
            )
        if self.include_images:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = self.dataset_dir / image_path
            image = Image.open(image_path).convert("RGB")
            item["image"] = image_to_tensor(image, self.image_size)
        return item


def image_to_tensor(image: Image.Image, image_size: int) -> Any:
    torch = torch_module()
    import numpy as np

    image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    # mean=0.5, std=2.0 matches the upstream Koh et al. ConceptBottleneck
    # CUB/dataset.py transform (transforms.Normalize([0.5]*3, [2]*3)); it is
    # intentionally not ImageNet standardization. Keep it for backend fidelity.
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def build_from_checkpoint(checkpoint: dict[str, Any]) -> Any:
    return build_official_cbm(OfficialCBMSpec(**checkpoint["model_spec"]))


def load_torch_checkpoint(path: Path, device: str) -> dict[str, Any]:
    torch = torch_module()
    # weights_only=True uses torch's restricted unpickler, which is sufficient
    # for our checkpoints (plain dicts/lists/scalars plus a tensor state_dict)
    # and avoids executing arbitrary pickles from checkpoint files.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} is not a dict")
    return checkpoint
