"""Ensemble-based uncertainty analysis for CBM symbol firing.

Implements the symbol-level and label-level decomposition of predictive
uncertainty into aleatoric and epistemic components over an ensemble of
independently trained concept detectors that share one frozen head, plus the
freeze-to-consensus attribution that links symbol disagreement to label
disagreement.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from symbol_sanity.io import load_dataset, require_concept_names, write_json
from symbol_sanity.logging_utils import log as _log
from symbol_sanity.torch_runtime import (
    build_from_checkpoint as _build_from_checkpoint,
    load_torch_checkpoint as _load_torch_checkpoint,
    make_loader as _make_loader,
    torch_module as _torch,
)

_EPS = 1e-7


def collect_detector_probabilities(
    *,
    dataset_dir: Path,
    detector_path: Path,
    batch_size: int,
    device: str,
) -> Any:
    """Run one detector over a dataset and return sigmoid probabilities `[N, K]`."""

    torch = _torch()
    schema, rows = load_dataset(dataset_dir)
    checkpoint = _load_torch_checkpoint(detector_path, device)
    require_concept_names(
        list(checkpoint["concept_names"]),
        list(schema["concept_names"]),
    )
    detector = _build_from_checkpoint(checkpoint).to(device)
    detector.load_state_dict(checkpoint["state_dict"])
    detector.eval()

    loader = _make_loader(
        dataset_dir=dataset_dir,
        rows=rows,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        image_size=int(checkpoint["image_size"]),
        include_images=True,
        task_name=None,
    )
    chunks = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = torch.cat(detector(images), dim=1)
            chunks.append(torch.sigmoid(logits).cpu())
    return torch.cat(chunks, dim=0)


def collect_ensemble_probabilities(
    *,
    dataset_dir: Path,
    detector_paths: list[Path],
    batch_size: int,
    device: str,
) -> Any:
    """Stack member probabilities into an `[M, N, K]` tensor."""

    torch = _torch()
    members = []
    for member_index, detector_path in enumerate(detector_paths):
        _log(
            f"ensemble member {member_index + 1}/{len(detector_paths)} "
            f"detector={detector_path}"
        )
        members.append(
            collect_detector_probabilities(
                dataset_dir=dataset_dir,
                detector_path=detector_path,
                batch_size=batch_size,
                device=device,
            )
        )
    return torch.stack(members, dim=0)


def binary_entropy(probabilities: Any) -> Any:
    torch = _torch()
    p = probabilities.clamp(_EPS, 1.0 - _EPS)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def categorical_entropy(probabilities: Any) -> Any:
    torch = _torch()
    p = probabilities.clamp_min(_EPS)
    return -(p * torch.log(p)).sum(dim=-1)


def symbol_uncertainty_decomposition(member_probs: Any) -> dict[str, Any]:
    """Decompose per-symbol uncertainty for member probabilities `[M, N, K]`."""

    mean_probs = member_probs.mean(dim=0)
    total = binary_entropy(mean_probs)
    aleatoric = binary_entropy(member_probs).mean(dim=0)
    return {
        "mean_probs": mean_probs,
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": total - aleatoric,
    }


def label_uncertainty_decomposition(label_probs: Any) -> dict[str, Any]:
    """Decompose label uncertainty for member distributions `[M, N, C]`."""

    mean_probs = label_probs.mean(dim=0)
    total = categorical_entropy(mean_probs)
    aleatoric = categorical_entropy(label_probs).mean(dim=0)
    return {
        "mean_probs": mean_probs,
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": total - aleatoric,
    }


def member_label_distributions(
    *,
    head: Any,
    member_probs: Any,
    batch_size: int,
) -> Any:
    """Push every member's concept probabilities through one frozen head."""

    torch = _torch()
    member_count = member_probs.shape[0]
    distributions = []
    with torch.no_grad():
        for member_index in range(member_count):
            chunks = []
            concepts = member_probs[member_index]
            for start in range(0, concepts.shape[0], batch_size):
                logits = head(concepts[start : start + batch_size])
                chunks.append(torch.softmax(logits, dim=1))
            distributions.append(torch.cat(chunks, dim=0))
    return torch.stack(distributions, dim=0)


def compute_epistemic_table(
    *,
    dataset_dir: Path,
    detector_paths: list[Path],
    batch_size: int,
    device: str,
) -> Any:
    """Per-example, per-concept epistemic uncertainty `[N, K]` for training use."""

    member_probs = collect_ensemble_probabilities(
        dataset_dir=dataset_dir,
        detector_paths=detector_paths,
        batch_size=batch_size,
        device=device,
    )
    return symbol_uncertainty_decomposition(member_probs)["epistemic"]


def evaluate_ensemble_uncertainty(
    *,
    dataset_dir: Path,
    detector_paths: list[Path],
    head_path: Path,
    batch_size: int,
    device: str,
    output_dir: Path,
    num_worked_examples: int = 5,
    member_probs: Any = None,
) -> dict[str, Any]:
    """Full uncertainty report for an ensemble of detectors and one frozen head.

    ``member_probs`` may carry precomputed `[M, N, K]` detector probabilities
    (in ``detector_paths`` order) to avoid re-running the detector passes.
    """

    torch = _torch()
    if len(detector_paths) < 2:
        raise ValueError("Ensemble uncertainty requires at least two detectors")
    schema, rows = load_dataset(dataset_dir)
    concept_names = list(schema["concept_names"])

    head_checkpoint = _load_torch_checkpoint(head_path, device)
    require_concept_names(list(head_checkpoint["concept_names"]), concept_names)
    task_name = str(head_checkpoint["task_name"])
    if task_name not in schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in dataset schema")
    head = _build_from_checkpoint(head_checkpoint)
    head.load_state_dict(head_checkpoint["state_dict"])
    head.eval()

    _log(
        f"start uncertainty report dataset={dataset_dir} members={len(detector_paths)} "
        f"head={head_path} task={task_name}"
    )
    if member_probs is None:
        member_probs = collect_ensemble_probabilities(
            dataset_dir=dataset_dir,
            detector_paths=detector_paths,
            batch_size=batch_size,
            device=device,
        )
    elif tuple(member_probs.shape[:2]) != (len(detector_paths), len(rows)):
        raise ValueError(
            "member_probs shape does not match detector_paths and dataset size"
        )

    symbol = symbol_uncertainty_decomposition(member_probs)
    label_probs = member_label_distributions(
        head=head,
        member_probs=member_probs,
        batch_size=batch_size,
    )
    label = label_uncertainty_decomposition(label_probs)

    labels_true = torch.tensor(
        [int(row["task_labels"][task_name]) for row in rows],
        dtype=torch.long,
    )
    mean_label_probs = label["mean_probs"]
    labels_pred = mean_label_probs.argmax(dim=1)
    correct = (labels_pred == labels_true).float()
    cross_entropy = -torch.log(
        mean_label_probs.gather(1, labels_true.view(-1, 1)).clamp_min(_EPS)
    ).squeeze(1)

    _log("compute freeze-to-consensus attribution per concept")
    delta = _freeze_attribution(
        head=head,
        member_probs=member_probs,
        symbol_mean_probs=symbol["mean_probs"],
        label_epistemic=label["epistemic"],
        batch_size=batch_size,
    )

    symbol_epistemic = symbol["epistemic"]
    symbol_epistemic_mean = symbol_epistemic.mean(dim=1)
    symbol_epistemic_max = symbol_epistemic.max(dim=1).values

    output_dir.mkdir(parents=True, exist_ok=True)
    concept_csv = output_dir / "per_concept_uncertainty.csv"
    _write_per_concept_csv(
        concept_csv,
        concept_names=concept_names,
        rows=rows,
        symbol=symbol,
        delta=delta,
        label_epistemic=label["epistemic"],
    )
    example_csv = output_dir / "per_example_uncertainty.csv"
    _write_per_example_csv(
        example_csv,
        rows=rows,
        concept_names=concept_names,
        labels_true=labels_true,
        labels_pred=labels_pred,
        correct=correct,
        cross_entropy=cross_entropy,
        label=label,
        symbol_epistemic=symbol_epistemic,
        delta=delta,
    )

    correlations = {
        "symbol_epistemic_mean_vs_label_epistemic": _correlation_pair(
            symbol_epistemic_mean.tolist(), label["epistemic"].tolist()
        ),
        "symbol_epistemic_mean_vs_cross_entropy": _correlation_pair(
            symbol_epistemic_mean.tolist(), cross_entropy.tolist()
        ),
        "symbol_epistemic_mean_vs_correct": _correlation_pair(
            symbol_epistemic_mean.tolist(), correct.tolist()
        ),
        "label_epistemic_vs_cross_entropy": _correlation_pair(
            label["epistemic"].tolist(), cross_entropy.tolist()
        ),
        "label_epistemic_vs_correct": _correlation_pair(
            label["epistemic"].tolist(), correct.tolist()
        ),
    }

    worked_examples = _select_worked_examples(
        rows=rows,
        concept_names=concept_names,
        member_probs=member_probs,
        symbol_epistemic=symbol_epistemic,
        delta=delta,
        labels_true=labels_true,
        labels_pred=labels_pred,
        label_epistemic=label["epistemic"],
        cross_entropy=cross_entropy,
        num_examples=num_worked_examples,
    )

    report = {
        "dataset_dir": str(dataset_dir),
        "detector_paths": [str(path) for path in detector_paths],
        "head_path": str(head_path),
        "task_name": task_name,
        "ensemble_size": len(detector_paths),
        "num_examples": len(rows),
        "num_concepts": len(concept_names),
        "ensemble_accuracy": float(correct.mean()),
        "mean_cross_entropy": float(cross_entropy.mean()),
        "mean_symbol_total": float(symbol["total"].mean()),
        "mean_symbol_aleatoric": float(symbol["aleatoric"].mean()),
        "mean_symbol_epistemic": float(symbol_epistemic.mean()),
        "mean_label_total": float(label["total"].mean()),
        "mean_label_aleatoric": float(label["aleatoric"].mean()),
        "mean_label_epistemic": float(label["epistemic"].mean()),
        "correlations": correlations,
        "worked_examples": worked_examples,
        "per_concept_uncertainty_csv": str(concept_csv),
        "per_example_uncertainty_csv": str(example_csv),
    }
    write_json(output_dir / "uncertainty_report.json", report)
    _log(f"wrote uncertainty report path={output_dir / 'uncertainty_report.json'}")
    return report


def _freeze_attribution(
    *,
    head: Any,
    member_probs: Any,
    symbol_mean_probs: Any,
    label_epistemic: Any,
    batch_size: int,
) -> Any:
    """Delta `[N, K]`: label epistemic drop when concept k is frozen to consensus."""

    torch = _torch()
    num_concepts = member_probs.shape[2]
    deltas = []
    for concept_index in range(num_concepts):
        frozen = member_probs.clone()
        frozen[:, :, concept_index] = symbol_mean_probs[:, concept_index].unsqueeze(0)
        frozen_label_probs = member_label_distributions(
            head=head,
            member_probs=frozen,
            batch_size=batch_size,
        )
        frozen_epistemic = label_uncertainty_decomposition(frozen_label_probs)[
            "epistemic"
        ]
        deltas.append(label_epistemic - frozen_epistemic)
    return torch.stack(deltas, dim=1)


def _select_worked_examples(
    *,
    rows: list[dict[str, Any]],
    concept_names: list[str],
    member_probs: Any,
    symbol_epistemic: Any,
    delta: Any,
    labels_true: Any,
    labels_pred: Any,
    label_epistemic: Any,
    cross_entropy: Any,
    num_examples: int,
) -> list[dict[str, Any]]:
    """Pick examples where one uncertain symbol carries weight in the decision."""

    score_per_concept = symbol_epistemic * delta.abs()
    score, top_concept = score_per_concept.max(dim=1)
    order = score.argsort(descending=True)[:num_examples].tolist()
    examples = []
    for example_index in order:
        concept_index = int(top_concept[example_index])
        examples.append(
            {
                "example_index": example_index,
                "image_path": rows[example_index].get("image_path"),
                "label_true": int(labels_true[example_index]),
                "label_pred": int(labels_pred[example_index]),
                "cross_entropy": float(cross_entropy[example_index]),
                "label_epistemic": float(label_epistemic[example_index]),
                "concept_index": concept_index,
                "concept_name": concept_names[concept_index],
                "concept_epistemic": float(
                    symbol_epistemic[example_index, concept_index]
                ),
                "concept_delta": float(delta[example_index, concept_index]),
                "member_probabilities": [
                    float(value)
                    for value in member_probs[:, example_index, concept_index].tolist()
                ],
            }
        )
    return examples


def _write_per_concept_csv(
    path: Path,
    *,
    concept_names: list[str],
    rows: list[dict[str, Any]],
    symbol: dict[str, Any],
    delta: Any,
    label_epistemic: Any,
) -> None:
    label_epistemic_values = label_epistemic.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "concept_index",
                "concept_name",
                "prevalence",
                "mean_total",
                "mean_aleatoric",
                "mean_epistemic",
                "mean_delta",
                "mean_abs_delta",
                "pearson_epistemic_vs_label_epistemic",
            ]
        )
        for concept_index, concept_name in enumerate(concept_names):
            prevalence = sum(
                int(row["concept_vector"][concept_index]) for row in rows
            ) / len(rows)
            epistemic_values = symbol["epistemic"][:, concept_index].tolist()
            writer.writerow(
                [
                    concept_index,
                    concept_name,
                    prevalence,
                    float(symbol["total"][:, concept_index].mean()),
                    float(symbol["aleatoric"][:, concept_index].mean()),
                    float(symbol["epistemic"][:, concept_index].mean()),
                    float(delta[:, concept_index].mean()),
                    float(delta[:, concept_index].abs().mean()),
                    _pearson(epistemic_values, label_epistemic_values),
                ]
            )


def _write_per_example_csv(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    concept_names: list[str],
    labels_true: Any,
    labels_pred: Any,
    correct: Any,
    cross_entropy: Any,
    label: dict[str, Any],
    symbol_epistemic: Any,
    delta: Any,
) -> None:
    top_epistemic_concept = symbol_epistemic.argmax(dim=1)
    top_delta_concept = delta.abs().argmax(dim=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "example_index",
                "image_path",
                "label_true",
                "label_pred",
                "correct",
                "cross_entropy",
                "label_total",
                "label_aleatoric",
                "label_epistemic",
                "symbol_epistemic_mean",
                "symbol_epistemic_max",
                "top_epistemic_concept",
                "top_delta_concept",
            ]
        )
        for example_index, row in enumerate(rows):
            writer.writerow(
                [
                    example_index,
                    row.get("image_path"),
                    int(labels_true[example_index]),
                    int(labels_pred[example_index]),
                    int(correct[example_index]),
                    float(cross_entropy[example_index]),
                    float(label["total"][example_index]),
                    float(label["aleatoric"][example_index]),
                    float(label["epistemic"][example_index]),
                    float(symbol_epistemic[example_index].mean()),
                    float(symbol_epistemic[example_index].max()),
                    concept_names[int(top_epistemic_concept[example_index])],
                    concept_names[int(top_delta_concept[example_index])],
                ]
            )


def _correlation_pair(xs: list[float], ys: list[float]) -> dict[str, float | None]:
    return {
        "pearson": _pearson(xs, ys),
        "spearman": _pearson(_ranks(xs), _ranks(ys)),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + end + 1) / 2
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks
