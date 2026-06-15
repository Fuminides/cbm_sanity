"""Statistical reports for CBM detector/head compatibility experiments."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from symbol_sanity.io import load_dataset, read_json, require_concept_names, write_json
from symbol_sanity.logging_utils import log as _log
from symbol_sanity.metrics import accuracy, macro_f1
from symbol_sanity.torch_runtime import (
    build_from_checkpoint as _build_from_checkpoint,
    load_torch_checkpoint as _load_torch_checkpoint,
    make_loader as _make_loader,
    torch_module as _torch,
)


@dataclass(frozen=True)
class ConceptReliabilityRow:
    concept_index: int
    concept_name: str
    prevalence: float
    accuracy_mean: float
    accuracy_std: float
    balanced_accuracy_mean: float | None
    balanced_accuracy_std: float | None
    f1_mean: float
    f1_std: float
    auroc_mean: float | None
    auroc_std: float | None
    auprc_mean: float | None
    auprc_std: float | None
    permutation_p_value_min: float
    detector_count: int


def generate_statistical_report(
    *,
    summary_path: Path,
    output_dir: Path,
    num_permutations: int,
    seed: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Create concept reliability and head null reports from an experiment summary."""

    _log(
        f"start statistical report summary={summary_path} output_dir={output_dir} "
        f"permutations={num_permutations} device={device}"
    )
    summary = read_json(summary_path)
    for required_key in ("eval_dir", "head_train", "detector_evaluations"):
        if required_key not in summary:
            raise ValueError(
                f"Summary {summary_path} is missing {required_key!r}. "
                "Statistical reports require a manifest-experiment summary "
                "(run-official-manifest-experiment), which records the "
                "evaluation split, the trained head, and detector evaluations."
            )
    eval_dir = Path(summary["eval_dir"])
    head_path = Path(summary["head_train"]["checkpoint_path"])
    detector_paths = {
        str(detector_seed): Path(evaluation["detector_path"])
        for detector_seed, evaluation in summary["detector_evaluations"].items()
    }
    schema, rows = load_dataset(eval_dir)
    concept_names = list(schema["concept_names"])

    output_dir.mkdir(parents=True, exist_ok=True)
    detector_reports = {}
    per_detector_concept_metrics: dict[str, list[dict[str, Any]]] = {}
    for detector_seed, detector_path in detector_paths.items():
        _log(f"evaluate concept reliability detector_seed={detector_seed}")
        detector_report = evaluate_detector_concepts(
            dataset_dir=eval_dir,
            detector_path=detector_path,
            batch_size=batch_size,
            device=device,
            num_permutations=num_permutations,
            seed=seed + int(detector_seed),
        )
        detector_reports[detector_seed] = {
            key: value
            for key, value in detector_report.items()
            if key != "concept_metrics"
        }
        per_detector_concept_metrics[detector_seed] = detector_report[
            "concept_metrics"
        ]
        _log(
            f"detector_seed={detector_seed} "
            f"mean_balanced_accuracy={detector_report['mean_balanced_accuracy']} "
            f"mean_accuracy={detector_report['mean_accuracy']:.4f}"
        )

    _log("aggregate per-concept reliability across detectors")
    concept_rows = aggregate_concept_metrics(
        concept_names=concept_names,
        per_detector_concept_metrics=per_detector_concept_metrics,
    )
    concept_csv_path = output_dir / "concept_reliability.csv"
    write_concept_reliability_csv(concept_csv_path, concept_rows)
    _log(f"wrote concept reliability csv path={concept_csv_path}")

    _log("evaluate head compatibility null distributions on oracle concepts")
    head_report = evaluate_head_compatibility_nulls(
        dataset_dir=eval_dir,
        head_path=head_path,
        batch_size=batch_size,
        device=device,
        num_permutations=num_permutations,
        seed=seed,
    )
    head_null_csv_path = output_dir / "head_compatibility_nulls.csv"
    write_head_null_csv(head_null_csv_path, head_report["null_rows"])
    _log(f"wrote head null csv path={head_null_csv_path}")

    primary_detector_seed = min(detector_paths, key=int)
    _log(
        "evaluate head compatibility null distributions on predicted concepts "
        f"detector_seed={primary_detector_seed}"
    )
    predicted_head_report = evaluate_head_compatibility_nulls(
        dataset_dir=eval_dir,
        head_path=head_path,
        batch_size=batch_size,
        device=device,
        num_permutations=num_permutations,
        seed=seed,
        detector_path=detector_paths[primary_detector_seed],
    )
    predicted_null_csv_path = output_dir / "head_compatibility_nulls_predicted.csv"
    write_head_null_csv(predicted_null_csv_path, predicted_head_report["null_rows"])
    _log(f"wrote predicted head null csv path={predicted_null_csv_path}")

    report = {
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "eval_dir": str(eval_dir),
        "num_examples": len(rows),
        "num_concepts": len(concept_names),
        "num_permutations": num_permutations,
        "seed": seed,
        "detector_reports": detector_reports,
        "concept_reliability_csv": str(concept_csv_path),
        "head_compatibility_nulls_csv": str(head_null_csv_path),
        "head_compatibility_nulls_predicted_csv": str(predicted_null_csv_path),
        "head_compatibility": {
            key: value
            for key, value in head_report.items()
            if key != "null_rows"
        },
        "head_compatibility_predicted": {
            key: value
            for key, value in predicted_head_report.items()
            if key != "null_rows"
        },
        "concept_reliability_summary": summarize_concept_rows(concept_rows),
    }
    write_json(output_dir / "statistical_report.json", report)
    _log(f"wrote statistical report path={output_dir / 'statistical_report.json'}")
    return report


def evaluate_detector_concepts(
    *,
    dataset_dir: Path,
    detector_path: Path,
    batch_size: int,
    device: str,
    num_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate per-concept reliability for one detector checkpoint."""

    torch = _torch()
    schema, rows = load_dataset(dataset_dir)
    detector_checkpoint = _load_torch_checkpoint(detector_path, device)
    concept_names = list(schema["concept_names"])
    require_concept_names(list(detector_checkpoint["concept_names"]), concept_names)

    detector = _build_from_checkpoint(detector_checkpoint).to(device)
    detector.load_state_dict(detector_checkpoint["state_dict"])
    detector.eval()
    loader = _make_loader(
        dataset_dir=dataset_dir,
        rows=rows,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        image_size=int(detector_checkpoint["image_size"]),
        include_images=True,
        task_name=None,
    )

    probabilities: list[list[float]] = []
    predictions: list[list[int]] = []
    targets: list[list[int]] = []
    _log(
        f"detector concept pass examples={len(rows)} concepts={len(concept_names)} "
        f"detector={detector_path}"
    )
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            concepts = batch["concept"].to(device)
            logits = torch.cat(detector(images), dim=1)
            probs = torch.sigmoid(logits)
            probabilities.extend(probs.cpu().tolist())
            predictions.extend((probs.cpu() >= 0.5).int().tolist())
            targets.extend(concepts.cpu().int().tolist())

    concept_metrics = []
    for concept_index, concept_name in enumerate(concept_names):
        y_true = [row[concept_index] for row in targets]
        y_pred = [row[concept_index] for row in predictions]
        y_score = [row[concept_index] for row in probabilities]
        metric = binary_metric_row(
            concept_index=concept_index,
            concept_name=concept_name,
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
        )
        metric["permutation_p_value"] = (
            1.0
            if metric["balanced_accuracy"] is None
            else permutation_p_value_for_balanced_accuracy(
                y_true=y_true,
                y_pred=y_pred,
                observed_balanced_accuracy=float(metric["balanced_accuracy"]),
                num_permutations=num_permutations,
                seed=seed + concept_index,
            )
        )
        concept_metrics.append(metric)

    return {
        "dataset_dir": str(dataset_dir),
        "detector_path": str(detector_path),
        "num_examples": len(rows),
        "num_concepts": len(concept_names),
        "mean_accuracy": mean(float(row["accuracy"]) for row in concept_metrics),
        "mean_balanced_accuracy": optional_mean(
            row["balanced_accuracy"] for row in concept_metrics
        ),
        "mean_f1": mean(float(row["f1"]) for row in concept_metrics),
        "concept_metrics": concept_metrics,
    }


def evaluate_head_compatibility_nulls(
    *,
    dataset_dir: Path,
    head_path: Path,
    batch_size: int,
    device: str,
    num_permutations: int,
    seed: int,
    detector_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a `C -> Y` head against concept-interface null distributions.

    With ``detector_path`` the nulls are computed on the detector's predicted
    sigmoid concept probabilities (the distribution jointly trained heads
    actually see) instead of the oracle binary concept vectors.
    """

    torch = _torch()
    schema, rows = load_dataset(dataset_dir)
    head_checkpoint = _load_torch_checkpoint(head_path, device)
    concept_names = list(schema["concept_names"])
    require_concept_names(list(head_checkpoint["concept_names"]), concept_names)
    task_name = str(head_checkpoint["task_name"])
    if task_name not in schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in dataset schema")

    head = _build_from_checkpoint(head_checkpoint).to(device)
    head.load_state_dict(head_checkpoint["state_dict"])
    head.eval()

    if detector_path is not None:
        from symbol_sanity.uncertainty import collect_detector_probabilities

        concepts = collect_detector_probabilities(
            dataset_dir=dataset_dir,
            detector_path=detector_path,
            batch_size=batch_size,
            device=device,
        ).to(device)
    else:
        concepts = torch.tensor(
            [row["concept_vector"] for row in rows],
            dtype=torch.float32,
            device=device,
        )
    labels = torch.tensor(
        [int(row["task_labels"][task_name]) for row in rows],
        dtype=torch.long,
        device=device,
    )
    observed = evaluate_head_on_concepts(
        head=head,
        concepts=concepts,
        labels=labels,
        batch_size=batch_size,
        num_classes=int(head_checkpoint["num_classes"]),
    )

    rng = random.Random(seed)
    null_rows = []
    log_every = max(1, num_permutations // 10) if num_permutations else 1
    for permutation_index in range(num_permutations):
        if permutation_index == 0 or (permutation_index + 1) % log_every == 0:
            _log(
                f"head null permutation {permutation_index + 1}/{num_permutations}"
            )
        concept_permutation = list(range(len(concept_names)))
        rng.shuffle(concept_permutation)
        permuted = concepts[:, concept_permutation]
        null_rows.append(
            {
                "null_type": "concept_dimension_permutation",
                "permutation_index": permutation_index,
                **evaluate_head_on_concepts(
                    head=head,
                    concepts=permuted,
                    labels=labels,
                    batch_size=batch_size,
                    num_classes=int(head_checkpoint["num_classes"]),
                ),
            }
        )

        shuffled = concepts.clone()
        row_indices = list(range(concepts.shape[0]))
        for concept_index in range(concepts.shape[1]):
            rng.shuffle(row_indices)
            index_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
            shuffled[:, concept_index] = concepts[index_tensor, concept_index]
        null_rows.append(
            {
                "null_type": "within_concept_sample_shuffle",
                "permutation_index": permutation_index,
                **evaluate_head_on_concepts(
                    head=head,
                    concepts=shuffled,
                    labels=labels,
                    batch_size=batch_size,
                    num_classes=int(head_checkpoint["num_classes"]),
                ),
            }
        )

    dimension_null = [
        row["accuracy"]
        for row in null_rows
        if row["null_type"] == "concept_dimension_permutation"
    ]
    shuffle_null = [
        row["accuracy"]
        for row in null_rows
        if row["null_type"] == "within_concept_sample_shuffle"
    ]
    return {
        "dataset_dir": str(dataset_dir),
        "head_path": str(head_path),
        "task_name": task_name,
        "concept_source": (
            "oracle" if detector_path is None else "predicted"
        ),
        "detector_path": None if detector_path is None else str(detector_path),
        "num_examples": len(rows),
        "observed_accuracy": observed["accuracy"],
        "observed_macro_f1": observed["macro_f1"],
        "dimension_permutation_accuracy_mean": mean(dimension_null),
        "sample_shuffle_accuracy_mean": mean(shuffle_null),
        "dimension_permutation_p_value": empirical_right_tail_p_value(
            observed["accuracy"],
            dimension_null,
        ),
        "sample_shuffle_p_value": empirical_right_tail_p_value(
            observed["accuracy"],
            shuffle_null,
        ),
        "null_rows": null_rows,
    }


def evaluate_head_on_concepts(
    *,
    head: Any,
    concepts: Any,
    labels: Any,
    batch_size: int,
    num_classes: int,
) -> dict[str, float]:
    torch = _torch()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for start in range(0, concepts.shape[0], batch_size):
            end = start + batch_size
            logits = head(concepts[start:end])
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(int(value) for value in labels[start:end].cpu().tolist())
            y_pred.extend(int(value) for value in predictions.cpu().tolist())
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
    }


def binary_metric_row(
    *,
    concept_index: int,
    concept_name: str,
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float],
) -> dict[str, Any]:
    positives = sum(y_true)
    prevalence = positives / len(y_true)
    return {
        "concept_index": concept_index,
        "concept_name": concept_name,
        "prevalence": prevalence,
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "f1": binary_f1(y_true, y_pred),
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
    }


def aggregate_concept_metrics(
    *,
    concept_names: list[str],
    per_detector_concept_metrics: dict[str, list[dict[str, Any]]],
) -> list[ConceptReliabilityRow]:
    rows = []
    for concept_index, concept_name in enumerate(concept_names):
        metrics = [
            detector_metrics[concept_index]
            for detector_metrics in per_detector_concept_metrics.values()
        ]
        rows.append(
            ConceptReliabilityRow(
                concept_index=concept_index,
                concept_name=concept_name,
                prevalence=mean(float(metric["prevalence"]) for metric in metrics),
                accuracy_mean=mean(float(metric["accuracy"]) for metric in metrics),
                accuracy_std=std(float(metric["accuracy"]) for metric in metrics),
                balanced_accuracy_mean=optional_mean(
                    metric["balanced_accuracy"] for metric in metrics
                ),
                balanced_accuracy_std=optional_std(
                    metric["balanced_accuracy"] for metric in metrics
                ),
                f1_mean=mean(float(metric["f1"]) for metric in metrics),
                f1_std=std(float(metric["f1"]) for metric in metrics),
                auroc_mean=optional_mean(metric["auroc"] for metric in metrics),
                auroc_std=optional_std(metric["auroc"] for metric in metrics),
                auprc_mean=optional_mean(metric["auprc"] for metric in metrics),
                auprc_std=optional_std(metric["auprc"] for metric in metrics),
                permutation_p_value_min=min(
                    float(metric["permutation_p_value"]) for metric in metrics
                ),
                detector_count=len(metrics),
            )
        )
    return rows


def summarize_concept_rows(rows: list[ConceptReliabilityRow]) -> dict[str, Any]:
    balanced = [
        row.balanced_accuracy_mean
        for row in rows
        if row.balanced_accuracy_mean is not None
    ]
    unreliable = [
        row
        for row in rows
        if row.balanced_accuracy_mean is None
        or row.balanced_accuracy_mean < 0.6
        or row.permutation_p_value_min > 0.05
    ]
    sortable_rows = [
        row
        for row in rows
        if row.balanced_accuracy_mean is not None
    ]
    return {
        "mean_balanced_accuracy": mean(balanced),
        "std_balanced_accuracy": std(balanced),
        "num_undefined_balanced_accuracy": len(rows) - len(sortable_rows),
        "num_unreliable_balanced_acc_lt_0_6_or_p_gt_0_05": len(unreliable),
        "lowest_balanced_accuracy": [
            {
                "concept_index": row.concept_index,
                "concept_name": row.concept_name,
                "balanced_accuracy_mean": row.balanced_accuracy_mean,
                "prevalence": row.prevalence,
                "permutation_p_value_min": row.permutation_p_value_min,
            }
            for row in sorted(
                sortable_rows,
                key=lambda item: item.balanced_accuracy_mean
                if item.balanced_accuracy_mean is not None
                else -1.0,
            )[:10]
        ],
    }


def write_concept_reliability_csv(
    path: Path,
    rows: list[ConceptReliabilityRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ConceptReliabilityRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_head_null_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["null_type", "permutation_index", "accuracy", "macro_f1"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float | None:
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    if tp + fn == 0 or tn + fp == 0:
        return None
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr + tnr) / 2


def binary_f1(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2 * tp) / denominator


def auroc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def auprc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = sum(y_true)
    if positives == 0:
        return None
    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    previous_recall = 0.0
    area = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += precision * (recall - previous_recall)
        previous_recall = recall
    return area


def permutation_p_value_for_balanced_accuracy(
    *,
    y_true: list[int],
    y_pred: list[int],
    observed_balanced_accuracy: float,
    num_permutations: int,
    seed: int,
) -> float:
    if num_permutations <= 0:
        return 1.0
    rng = random.Random(seed)
    null_equal_or_better = 0
    shuffled = list(y_true)
    for _ in range(num_permutations):
        rng.shuffle(shuffled)
        null_score = balanced_accuracy(shuffled, y_pred)
        null_equal_or_better += int(
            null_score is not None and null_score >= observed_balanced_accuracy
        )
    return (1 + null_equal_or_better) / (1 + num_permutations)


def empirical_right_tail_p_value(observed: float, null_values: list[float]) -> float:
    if not null_values:
        return 1.0
    return (1 + sum(value >= observed for value in null_values)) / (
        1 + len(null_values)
    )


def mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: Any) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return (sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def optional_mean(values: Any) -> float | None:
    filtered = [value for value in values if value is not None]
    return None if not filtered else mean(filtered)


def optional_std(values: Any) -> float | None:
    filtered = [value for value in values if value is not None]
    return None if not filtered else std(filtered)
