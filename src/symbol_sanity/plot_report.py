"""Paper-oriented plots for CBM symbol-swap experiment reports."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from symbol_sanity.io import read_json, write_json
from symbol_sanity.logging_utils import log as _log


def generate_plot_report(
    *,
    summary_path: Path,
    statistical_report_dir: Path,
    output_dir: Path,
    formats: list[str] | None = None,
) -> dict[str, Any]:
    """Generate paper plots from an official run summary and statistical report."""

    formats = formats or ["png", "pdf"]
    _log(
        f"start plot report summary={summary_path} "
        f"statistical_report_dir={statistical_report_dir} output_dir={output_dir}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()

    summary = read_json(summary_path)
    report = read_json(statistical_report_dir / "statistical_report.json")
    concept_rows = _read_csv(statistical_report_dir / "concept_reliability.csv")
    null_rows = _read_csv(statistical_report_dir / "head_compatibility_nulls.csv")

    outputs: dict[str, list[str] | str] = {}
    _log("plot swap accuracy heatmap")
    outputs["swap_accuracy_heatmap"] = _save_figure(
        _plot_swap_heatmap(plt, summary, value_key="swapped_accuracy"),
        output_dir / "swap_accuracy_heatmap",
        formats,
    )
    _log("plot swap drop heatmap")
    outputs["swap_drop_heatmap"] = _save_figure(
        _plot_swap_heatmap(plt, summary, value_key="swap_drop"),
        output_dir / "swap_drop_heatmap",
        formats,
    )
    if summary.get("head_detector_rows"):
        _log("plot head x detector accuracy heatmap")
        outputs["head_detector_accuracy_heatmap"] = _save_figure(
            _plot_head_detector_heatmap(plt, summary),
            output_dir / "head_detector_accuracy_heatmap",
            formats,
        )
    _log("plot detector metrics")
    outputs["detector_metrics_bar"] = _save_figure(
        _plot_detector_metrics(plt, summary),
        output_dir / "detector_metrics_bar",
        formats,
    )
    _log("plot concept balanced accuracy histogram")
    outputs["concept_balanced_accuracy_hist"] = _save_figure(
        _plot_concept_metric_histogram(
            plt,
            concept_rows,
            metric="balanced_accuracy_mean",
            title="Concept Identifiability",
            xlabel="Mean balanced accuracy across detectors",
        ),
        output_dir / "concept_balanced_accuracy_hist",
        formats,
    )
    _log("plot concept prevalence scatter")
    outputs["concept_prevalence_vs_balanced_accuracy"] = _save_figure(
        _plot_concept_prevalence_scatter(plt, concept_rows),
        output_dir / "concept_prevalence_vs_balanced_accuracy",
        formats,
    )
    _log("plot head compatibility nulls")
    outputs["head_compatibility_nulls"] = _save_figure(
        _plot_head_compatibility_nulls(
            plt,
            null_rows,
            observed_accuracy=float(
                report["head_compatibility"]["observed_accuracy"]
            ),
        ),
        output_dir / "head_compatibility_nulls",
        formats,
    )
    top_bottom_path = output_dir / "concept_reliability_top_bottom.csv"
    _write_top_bottom_concepts(top_bottom_path, concept_rows)
    _log(f"wrote top/bottom concept table path={top_bottom_path}")
    outputs["concept_reliability_top_bottom"] = str(top_bottom_path)
    model_table_path = output_dir / "model_results_table.csv"
    _write_model_results_table(model_table_path, summary)
    _log(f"wrote model results table path={model_table_path}")
    outputs["model_results_table"] = str(model_table_path)
    if summary.get("head_detector_rows"):
        matrix_table_path = output_dir / "head_detector_matrix.csv"
        _write_head_detector_matrix_table(matrix_table_path, summary)
        _log(f"wrote head-detector matrix table path={matrix_table_path}")
        outputs["head_detector_matrix_table"] = str(matrix_table_path)

    result = {
        "summary_path": str(summary_path),
        "statistical_report_dir": str(statistical_report_dir),
        "output_dir": str(output_dir),
        "formats": formats,
        "outputs": outputs,
    }
    write_json(output_dir / "plot_report.json", result)
    _log(f"wrote plot report path={output_dir / 'plot_report.json'}")
    return result


def _plot_swap_heatmap(plt: Any, summary: dict[str, Any], *, value_key: str) -> Any:
    seeds = sorted(
        {
            int(row["original_seed"])
            for row in summary["swap_rows"]
        }
        | {
            int(row["swap_seed"])
            for row in summary["swap_rows"]
        }
    )
    index = {seed: idx for idx, seed in enumerate(seeds)}
    matrix = [[0.0 for _ in seeds] for _ in seeds]
    for row in summary["swap_rows"]:
        matrix[index[int(row["original_seed"])]][index[int(row["swap_seed"])]] = float(
            row[value_key]
        )

    fig, ax = plt.subplots(figsize=(max(4.8, len(seeds) * 0.8), 4.2))
    cmap = "viridis" if value_key == "swapped_accuracy" else "coolwarm"
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(seeds)), labels=[str(seed) for seed in seeds])
    ax.set_yticks(range(len(seeds)), labels=[str(seed) for seed in seeds])
    ax.set_xlabel("Swapped detector seed")
    ax.set_ylabel("Head/original detector seed")
    title = "Detector Swap Accuracy" if value_key == "swapped_accuracy" else "Detector Swap Drop"
    ax.set_title(title)
    for row_idx, values in enumerate(matrix):
        for col_idx, value in enumerate(values):
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.45 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _plot_detector_metrics(plt: Any, summary: dict[str, Any]) -> Any:
    seeds = sorted(summary["detector_evaluations"], key=lambda value: int(value))
    metrics = [
        ("accuracy", "Task accuracy"),
        ("macro_f1", "Macro F1"),
        ("concept_agreement_with_oracle", "Concept agreement"),
    ]
    width = 0.24
    x_positions = list(range(len(seeds)))

    fig, ax = plt.subplots(figsize=(max(5.5, len(seeds) * 0.8), 4.2))
    for metric_idx, (metric_key, label) in enumerate(metrics):
        offset = (metric_idx - 1) * width
        values = [
            float(summary["detector_evaluations"][seed][metric_key])
            for seed in seeds
        ]
        ax.bar(
            [x_value + offset for x_value in x_positions],
            values,
            width=width,
            label=label,
        )
    ax.set_xticks(x_positions, labels=seeds)
    ax.set_xlabel("Detector seed")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Detector and Bottleneck Metrics")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def _plot_head_detector_heatmap(plt: Any, summary: dict[str, Any]) -> Any:
    rows = summary["head_detector_rows"]
    head_seeds = sorted({int(row["head_seed"]) for row in rows})
    detector_seeds = sorted({int(row["detector_seed"]) for row in rows})
    head_index = {seed: idx for idx, seed in enumerate(head_seeds)}
    detector_index = {seed: idx for idx, seed in enumerate(detector_seeds)}
    matrix = [[0.0 for _ in detector_seeds] for _ in head_seeds]
    for row in rows:
        matrix[head_index[int(row["head_seed"])]][
            detector_index[int(row["detector_seed"])]
        ] = float(row["accuracy"])

    fig, ax = plt.subplots(
        figsize=(max(5.0, len(detector_seeds) * 0.8), max(4.2, len(head_seeds) * 0.6))
    )
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(
        range(len(detector_seeds)),
        labels=[str(seed) for seed in detector_seeds],
    )
    ax.set_yticks(range(len(head_seeds)), labels=[str(seed) for seed in head_seeds])
    ax.set_xlabel("Detector seed")
    ax.set_ylabel("Head seed")
    ax.set_title("Classification Head x Detector Accuracy")
    for row_idx, values in enumerate(matrix):
        for col_idx, value in enumerate(values):
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.45 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _plot_concept_metric_histogram(
    plt: Any,
    rows: list[dict[str, str]],
    *,
    metric: str,
    title: str,
    xlabel: str,
) -> Any:
    values = [_optional_float(row[metric]) for row in rows]
    defined = [value for value in values if value is not None]
    undefined_count = len(values) - len(defined)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    if defined:
        ax.hist(defined, bins=20, color="#3A6EA5", edgecolor="white", alpha=0.9)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Concept count")
    ax.set_title(title)
    ax.text(
        0.02,
        0.96,
        f"Undefined: {undefined_count}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_concept_prevalence_scatter(plt: Any, rows: list[dict[str, str]]) -> Any:
    points = [
        (
            float(row["prevalence"]),
            _optional_float(row["balanced_accuracy_mean"]),
            float(row["permutation_p_value_min"]),
        )
        for row in rows
    ]
    defined = [point for point in points if point[1] is not None]

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    if defined:
        scatter = ax.scatter(
            [point[0] for point in defined],
            [float(point[1]) for point in defined],
            c=[point[2] for point in defined],
            cmap="viridis_r",
            s=28,
            alpha=0.85,
            edgecolors="none",
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Permutation p-value")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Concept prevalence")
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title("Concept Prevalence vs Identifiability")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    return fig


def _plot_head_compatibility_nulls(
    plt: Any,
    rows: list[dict[str, str]],
    *,
    observed_accuracy: float,
) -> Any:
    by_type: dict[str, list[float]] = {}
    for row in rows:
        by_type.setdefault(row["null_type"], []).append(float(row["accuracy"]))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = {
        "concept_dimension_permutation": "#E07A5F",
        "within_concept_sample_shuffle": "#3D405B",
    }
    labels = {
        "concept_dimension_permutation": "Dimension permutation",
        "within_concept_sample_shuffle": "Sample shuffle",
    }
    for null_type, values in sorted(by_type.items()):
        ax.hist(
            values,
            bins=20,
            alpha=0.65,
            label=labels.get(null_type, null_type),
            color=colors.get(null_type),
            edgecolor="white",
        )
    ax.axvline(
        observed_accuracy,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Observed: {observed_accuracy:.2f}",
    )
    ax.set_xlabel("Head accuracy")
    ax.set_ylabel("Null count")
    ax.set_title("Head Compatibility Nulls")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _write_top_bottom_concepts(path: Path, rows: list[dict[str, str]], top_k: int = 15) -> None:
    defined = [
        row
        for row in rows
        if _optional_float(row["balanced_accuracy_mean"]) is not None
    ]
    lowest = sorted(
        defined,
        key=lambda row: float(row["balanced_accuracy_mean"]),
    )[:top_k]
    highest = sorted(
        defined,
        key=lambda row: float(row["balanced_accuracy_mean"]),
        reverse=True,
    )[:top_k]
    fields = [
        "rank_type",
        "concept_index",
        "concept_name",
        "prevalence",
        "balanced_accuracy_mean",
        "accuracy_mean",
        "f1_mean",
        "auroc_mean",
        "auprc_mean",
        "permutation_p_value_min",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank_type, selected in [("lowest", lowest), ("highest", highest)]:
            for row in selected:
                writer.writerow(
                    {
                        field: rank_type if field == "rank_type" else row.get(field, "")
                        for field in fields
                    }
                )


def _write_model_results_table(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("model_rows")
    if not rows:
        head_train = summary.get("head_train", {})
        oracle = summary.get("oracle_head_evaluation", {})
        rows = [
            {
                "model_seed": seed,
                "head_path": head_train.get("checkpoint_path", ""),
                "detector_path": evaluation.get("detector_path", ""),
                "accuracy": evaluation["accuracy"],
                "macro_f1": evaluation["macro_f1"],
                "concept_agreement_with_oracle": evaluation[
                    "concept_agreement_with_oracle"
                ],
                "head_oracle_accuracy": oracle.get("accuracy", ""),
                "head_oracle_macro_f1": oracle.get("macro_f1", ""),
            }
            for seed, evaluation in sorted(
                summary["detector_evaluations"].items(),
                key=lambda item: int(item[0]),
            )
        ]
    fields = [
        "model_seed",
        "accuracy",
        "macro_f1",
        "concept_agreement_with_oracle",
        "head_oracle_accuracy",
        "head_oracle_macro_f1",
        "head_path",
        "detector_path",
    ]
    _write_dict_rows(path, rows, fields)


def _write_head_detector_matrix_table(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "head_seed",
        "detector_seed",
        "accuracy",
        "macro_f1",
        "concept_agreement_with_oracle",
        "is_matched_seed_pair",
        "head_path",
        "detector_path",
    ]
    _write_dict_rows(path, summary["head_detector_rows"], fields)


def _write_dict_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _save_figure(fig: Any, stem: Path, formats: list[str]) -> list[str]:
    paths = []
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        _log(f"wrote figure path={path}")
        paths.append(str(path))
    fig.clear()
    return paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _pyplot() -> Any:
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path("/tmp/symbol_sanity_matplotlib")
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )
    return plt
