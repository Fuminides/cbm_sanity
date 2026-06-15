"""Command line entry points for symbol-sanity utilities."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from symbol_sanity.awa2 import build_awa2_manifest
from symbol_sanity.cub import build_cub_manifest
from symbol_sanity.detectors import create_noisy_detector, create_oracle_detector
from symbol_sanity.evaluation import evaluate_detector_head, evaluate_swap
from symbol_sanity.heads import train_lookup_head
from symbol_sanity.io import load_dataset
from symbol_sanity.multihead import (
    run_reliability_comparison_experiment,
    train_multihead_cbm,
)
from symbol_sanity.neural_synthetic import (
    evaluate_official_oracle_head,
    evaluate_official_synthetic,
    evaluate_official_synthetic_swap,
    run_official_manifest_experiment,
    run_official_synthetic_experiment,
    run_shared_extractor_multihead_manifest_experiment,
    train_official_synthetic_detector,
    train_official_synthetic_head,
)
from symbol_sanity.official_cbm import backend_status
from symbol_sanity.plot_report import generate_plot_report
from symbol_sanity.schemas import TASK_SCHEMAS
from symbol_sanity.statistical_report import generate_statistical_report
from symbol_sanity.synthetic import export_dataset
from symbol_sanity.uncertainty import evaluate_ensemble_uncertainty


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="symbol-sanity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-synthetic",
        help="Generate deterministic colored-shape images and metadata.",
    )
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--num-examples", type=int, default=1000)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--image-size", type=int, default=64)
    generate.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_SCHEMAS),
        choices=sorted(TASK_SCHEMAS),
    )

    oracle_detector = subparsers.add_parser(
        "create-oracle-detector",
        help="Create a detector checkpoint that returns metadata concept vectors.",
    )
    oracle_detector.add_argument("--dataset-dir", required=True, type=Path)
    oracle_detector.add_argument("--output-path", required=True, type=Path)
    oracle_detector.add_argument("--name", default="oracle_detector")

    noisy_detector = subparsers.add_parser(
        "create-noisy-detector",
        help="Create a detector checkpoint that flips oracle concepts with fixed probability.",
    )
    noisy_detector.add_argument("--dataset-dir", required=True, type=Path)
    noisy_detector.add_argument("--output-path", required=True, type=Path)
    noisy_detector.add_argument("--name", default="noisy_detector")
    noisy_detector.add_argument("--flip-probability", required=True, type=float)
    noisy_detector.add_argument("--seed", required=True, type=int)

    train_head = subparsers.add_parser(
        "train-lookup-head",
        help="Train a majority-lookup concept head from metadata concept vectors.",
    )
    train_head.add_argument("--dataset-dir", required=True, type=Path)
    train_head.add_argument("--task", required=True, choices=sorted(TASK_SCHEMAS))
    train_head.add_argument("--output-path", required=True, type=Path)
    train_head.add_argument("--name", default="lookup_head")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a detector checkpoint feeding a frozen head checkpoint.",
    )
    evaluate.add_argument("--dataset-dir", required=True, type=Path)
    evaluate.add_argument("--detector", required=True, type=Path)
    evaluate.add_argument("--head", required=True, type=Path)
    evaluate.add_argument("--output-path", type=Path)

    swap = subparsers.add_parser(
        "evaluate-swap",
        help="Evaluate detector-head swap drop for a frozen head.",
    )
    swap.add_argument("--dataset-dir", required=True, type=Path)
    swap.add_argument("--original-detector", required=True, type=Path)
    swap.add_argument("--swap-detector", required=True, type=Path)
    swap.add_argument("--head", required=True, type=Path)
    swap.add_argument("--output-path", type=Path)

    subparsers.add_parser(
        "official-cbm-status",
        help="Check whether the torchvision-backed CBM backend is importable.",
    )

    train_official_detector = subparsers.add_parser(
        "train-official-synthetic-detector",
        help="Train an official Inception X->C detector on synthetic images.",
    )
    train_official_detector.add_argument("--dataset-dir", required=True, type=Path)
    train_official_detector.add_argument("--output-path", required=True, type=Path)
    train_official_detector.add_argument("--epochs", type=int, default=1)
    train_official_detector.add_argument("--batch-size", type=int, default=8)
    train_official_detector.add_argument("--lr", type=float, default=1e-3)
    train_official_detector.add_argument("--seed", type=int, default=0)
    train_official_detector.add_argument("--device", default="cpu")
    train_official_detector.add_argument("--image-size", type=int, default=299)
    train_official_detector.add_argument("--pretrained", action="store_true")
    train_official_detector.add_argument("--freeze", action="store_true")

    train_official_head = subparsers.add_parser(
        "train-official-synthetic-head",
        help="Train an official C->Y MLP head on synthetic oracle concepts.",
    )
    train_official_head.add_argument("--dataset-dir", required=True, type=Path)
    train_official_head.add_argument("--task", required=True)
    train_official_head.add_argument("--output-path", required=True, type=Path)
    train_official_head.add_argument("--epochs", type=int, default=50)
    train_official_head.add_argument("--batch-size", type=int, default=32)
    train_official_head.add_argument("--lr", type=float, default=1e-2)
    train_official_head.add_argument("--seed", type=int, default=0)
    train_official_head.add_argument("--device", default="cpu")
    train_official_head.add_argument("--expand-dim", type=int, default=0)

    evaluate_official = subparsers.add_parser(
        "evaluate-official-synthetic",
        help="Evaluate an official synthetic detector feeding a frozen official head.",
    )
    evaluate_official.add_argument("--dataset-dir", required=True, type=Path)
    evaluate_official.add_argument("--detector", required=True, type=Path)
    evaluate_official.add_argument("--head", required=True, type=Path)
    evaluate_official.add_argument("--batch-size", type=int, default=16)
    evaluate_official.add_argument("--device", default="cpu")
    evaluate_official.add_argument("--output-path", type=Path)

    evaluate_official_swap = subparsers.add_parser(
        "evaluate-official-synthetic-swap",
        help="Evaluate a neural official detector swap against a frozen official head.",
    )
    evaluate_official_swap.add_argument("--dataset-dir", required=True, type=Path)
    evaluate_official_swap.add_argument("--original-detector", required=True, type=Path)
    evaluate_official_swap.add_argument("--swap-detector", required=True, type=Path)
    evaluate_official_swap.add_argument("--head", required=True, type=Path)
    evaluate_official_swap.add_argument("--batch-size", type=int, default=16)
    evaluate_official_swap.add_argument("--device", default="cpu")
    evaluate_official_swap.add_argument("--output-path", type=Path)

    evaluate_oracle_head = subparsers.add_parser(
        "evaluate-official-oracle-head",
        help="Evaluate a frozen official C->Y head on oracle manifest concepts.",
    )
    evaluate_oracle_head.add_argument("--dataset-dir", required=True, type=Path)
    evaluate_oracle_head.add_argument("--head", required=True, type=Path)
    evaluate_oracle_head.add_argument("--batch-size", type=int, default=32)
    evaluate_oracle_head.add_argument("--device", default="cpu")
    evaluate_oracle_head.add_argument("--output-path", type=Path)

    run_official = subparsers.add_parser(
        "run-official-synthetic-experiment",
        help="Run data generation, official detector/head training, and swap evaluation.",
    )
    run_official.add_argument("--output-dir", required=True, type=Path)
    run_official.add_argument("--num-examples", type=int, default=64)
    run_official.add_argument("--data-seed", type=int, default=0)
    run_official.add_argument("--task", required=True)
    run_official.add_argument("--detector-seeds", nargs="+", type=int, default=[0, 1])
    run_official.add_argument("--detector-epochs", type=int, default=1)
    run_official.add_argument("--head-epochs", type=int, default=50)
    run_official.add_argument("--batch-size", type=int, default=8)
    run_official.add_argument("--detector-lr", type=float, default=1e-3)
    run_official.add_argument("--head-lr", type=float, default=1e-2)
    run_official.add_argument("--device", default="cpu")
    run_official.add_argument("--detector-image-size", type=int, default=299)

    run_manifest = subparsers.add_parser(
        "run-official-manifest-experiment",
        help="Run official detector/head/swap evaluation on existing manifests.",
    )
    run_manifest.add_argument("--train-dir", required=True, type=Path)
    run_manifest.add_argument("--eval-dir", required=True, type=Path)
    run_manifest.add_argument("--output-dir", required=True, type=Path)
    run_manifest.add_argument("--task", required=True)
    run_manifest.add_argument("--detector-seeds", nargs="+", type=int, default=[0, 1])
    run_manifest.add_argument("--detector-epochs", type=int, default=1)
    run_manifest.add_argument("--head-epochs", type=int, default=50)
    run_manifest.add_argument("--batch-size", type=int, default=8)
    run_manifest.add_argument("--detector-lr", type=float, default=1e-3)
    run_manifest.add_argument("--head-lr", type=float, default=1e-2)
    run_manifest.add_argument("--device", default="cpu")
    run_manifest.add_argument("--detector-image-size", type=int, default=299)
    run_manifest.add_argument("--pretrained", action="store_true")
    run_manifest.add_argument("--freeze", action="store_true")

    run_shared = subparsers.add_parser(
        "run-shared-extractor-multihead-manifest-experiment",
        help="Train one shared official X->C extractor jointly with multiple C->Y heads.",
    )
    run_shared.add_argument("--train-dir", required=True, type=Path)
    run_shared.add_argument("--eval-dir", required=True, type=Path)
    run_shared.add_argument("--output-dir", required=True, type=Path)
    run_shared.add_argument("--task", required=True)
    run_shared.add_argument("--head-seeds", nargs="+", type=int, default=[0, 1])
    run_shared.add_argument("--epochs", type=int, default=25)
    run_shared.add_argument("--batch-size", type=int, default=8)
    run_shared.add_argument("--lr", type=float, default=1e-4)
    run_shared.add_argument("--concept-loss-weight", type=float, default=1.0)
    run_shared.add_argument("--task-loss-weight", type=float, default=1.0)
    run_shared.add_argument("--device", default="cpu")
    run_shared.add_argument("--detector-image-size", type=int, default=299)
    run_shared.add_argument("--pretrained", action="store_true")
    run_shared.add_argument("--freeze", action="store_true")
    run_shared.add_argument("--seed", type=int, default=0)

    cub_manifest = subparsers.add_parser(
        "build-cub-manifest",
        help="Convert a raw CUB_200_2011 folder into train/val/test metadata manifests.",
    )
    cub_manifest.add_argument("--cub-root", required=True, type=Path)
    cub_manifest.add_argument("--output-dir", required=True, type=Path)
    cub_manifest.add_argument(
        "--attribute-policy",
        default="balanced",
        choices=["balanced", "koh112"],
    )
    cub_manifest.add_argument("--num-attributes", type=int, default=112)
    cub_manifest.add_argument("--class-ids", nargs="+", type=int)
    cub_manifest.add_argument("--num-classes", type=int)
    cub_manifest.add_argument("--class-start", type=int, default=1)
    cub_manifest.add_argument("--val-fraction", type=float, default=0.15)
    cub_manifest.add_argument("--seed", type=int, default=0)

    awa2_manifest = subparsers.add_parser(
        "build-awa2-manifest",
        help="Convert an AwA2 folder into train/val/test metadata manifests.",
    )
    awa2_manifest.add_argument("--awa2-root", required=True, type=Path)
    awa2_manifest.add_argument("--output-dir", required=True, type=Path)
    awa2_manifest.add_argument("--class-ids", nargs="+", type=int)
    awa2_manifest.add_argument("--num-classes", type=int)
    awa2_manifest.add_argument("--class-start", type=int, default=1)
    awa2_manifest.add_argument("--val-fraction", type=float, default=0.15)
    awa2_manifest.add_argument("--test-fraction", type=float, default=0.2)
    awa2_manifest.add_argument("--seed", type=int, default=0)
    awa2_manifest.add_argument(
        "--attribute-kind",
        default="binary",
        choices=["binary", "continuous-threshold"],
    )
    awa2_manifest.add_argument("--continuous-threshold", type=float, default=50.0)

    train_multihead = subparsers.add_parser(
        "train-multihead-cbm",
        help="Jointly train a shared X->C detector with R independent C->Y heads.",
    )
    train_multihead.add_argument("--train-dir", required=True, type=Path)
    train_multihead.add_argument("--task", required=True)
    train_multihead.add_argument("--output-dir", required=True, type=Path)
    train_multihead.add_argument("--num-heads", type=int, default=5)
    train_multihead.add_argument("--lambda-concept", type=float, default=1.0)
    train_multihead.add_argument("--epochs", type=int, default=1)
    train_multihead.add_argument("--batch-size", type=int, default=8)
    train_multihead.add_argument("--detector-lr", type=float, default=1e-4)
    train_multihead.add_argument("--head-lr", type=float, default=1e-3)
    train_multihead.add_argument("--seed", type=int, default=0)
    train_multihead.add_argument("--device", default="cpu")
    train_multihead.add_argument("--image-size", type=int, default=299)
    train_multihead.add_argument("--pretrained", action="store_true")
    train_multihead.add_argument("--freeze", action="store_true")
    train_multihead.add_argument("--eta", type=float, default=0.0)
    train_multihead.add_argument("--beta", type=float, default=0.0)
    train_multihead.add_argument("--concept-dropout", type=float, default=0.0)
    train_multihead.add_argument(
        "--reliability-detector",
        type=Path,
        help=(
            "Detector checkpoint used to compute per-concept balanced-accuracy "
            "reliability scores."
        ),
    )
    train_multihead.add_argument(
        "--reliability-dataset-dir",
        type=Path,
        help="Validation manifest used for reliability scores (never the test set).",
    )
    train_multihead.add_argument("--init-detector", type=Path)
    train_multihead.add_argument("--init-heads", nargs="+", type=Path)

    evaluate_uncertainty = subparsers.add_parser(
        "evaluate-uncertainty",
        help="Ensemble aleatoric/epistemic uncertainty report for symbol firing.",
    )
    evaluate_uncertainty.add_argument("--dataset-dir", required=True, type=Path)
    evaluate_uncertainty.add_argument(
        "--detectors", required=True, nargs="+", type=Path
    )
    evaluate_uncertainty.add_argument("--head", required=True, type=Path)
    evaluate_uncertainty.add_argument("--output-dir", required=True, type=Path)
    evaluate_uncertainty.add_argument("--batch-size", type=int, default=16)
    evaluate_uncertainty.add_argument("--device", default="cpu")
    evaluate_uncertainty.add_argument("--num-worked-examples", type=int, default=5)

    run_reliability = subparsers.add_parser(
        "run-reliability-experiment",
        help="Compare joint, multi-head, and reliability-aware CBM training arms.",
    )
    run_reliability.add_argument("--train-dir", required=True, type=Path)
    run_reliability.add_argument("--eval-dir", required=True, type=Path)
    run_reliability.add_argument("--val-dir", type=Path)
    run_reliability.add_argument("--output-dir", required=True, type=Path)
    run_reliability.add_argument("--task", required=True)
    run_reliability.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    run_reliability.add_argument("--num-heads", type=int, default=5)
    run_reliability.add_argument("--lambda-concept", type=float, default=1.0)
    run_reliability.add_argument("--eta", type=float, default=0.1)
    run_reliability.add_argument("--beta", type=float, default=1.0)
    run_reliability.add_argument("--concept-dropout", type=float, default=0.0)
    run_reliability.add_argument("--epochs", type=int, default=1)
    run_reliability.add_argument("--reliability-epochs", type=int, default=1)
    run_reliability.add_argument("--batch-size", type=int, default=8)
    run_reliability.add_argument("--detector-lr", type=float, default=1e-4)
    run_reliability.add_argument("--head-lr", type=float, default=1e-3)
    run_reliability.add_argument("--device", default="cpu")
    run_reliability.add_argument("--detector-image-size", type=int, default=299)
    run_reliability.add_argument("--pretrained", action="store_true")
    run_reliability.add_argument("--freeze", action="store_true")
    run_reliability.add_argument(
        "--arms",
        nargs="+",
        default=["joint", "multihead", "reliability"],
        choices=["joint", "multihead", "reliability"],
    )

    statistical_report = subparsers.add_parser(
        "statistical-report",
        help="Compute concept reliability and head-compatibility null reports.",
    )
    statistical_report.add_argument("--summary-path", required=True, type=Path)
    statistical_report.add_argument("--output-dir", required=True, type=Path)
    statistical_report.add_argument("--num-permutations", type=int, default=100)
    statistical_report.add_argument("--seed", type=int, default=0)
    statistical_report.add_argument("--batch-size", type=int, default=16)
    statistical_report.add_argument("--device", default="cpu")

    plot_report = subparsers.add_parser(
        "plot-report",
        help="Generate paper plots from a run summary and statistical report.",
    )
    plot_report.add_argument("--summary-path", required=True, type=Path)
    plot_report.add_argument("--statistical-report-dir", required=True, type=Path)
    plot_report.add_argument("--output-dir", required=True, type=Path)
    plot_report.add_argument("--formats", nargs="+", default=["png", "pdf"])

    environment = subparsers.add_parser(
        "experiment-environment",
        help="Write environment metadata for reproducible experiment runs.",
    )
    environment.add_argument("--output-path", required=True, type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-synthetic":
        export_dataset(
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            seed=args.seed,
            image_size=args.image_size,
            tasks=args.tasks,
        )
        return 0

    if args.command == "create-oracle-detector":
        schema, _ = load_dataset(args.dataset_dir)
        create_oracle_detector(
            args.output_path,
            name=args.name,
            concept_names=list(schema["concept_names"]),
        )
        return 0

    if args.command == "create-noisy-detector":
        schema, _ = load_dataset(args.dataset_dir)
        create_noisy_detector(
            args.output_path,
            name=args.name,
            concept_names=list(schema["concept_names"]),
            flip_probability=args.flip_probability,
            seed=args.seed,
        )
        return 0

    if args.command == "train-lookup-head":
        schema, rows = load_dataset(args.dataset_dir)
        if args.task not in schema["tasks"]:
            parser.error(f"Task {args.task!r} is not present in dataset schema")
        train_lookup_head(
            args.output_path,
            name=args.name,
            task_name=args.task,
            concept_names=list(schema["concept_names"]),
            num_classes=int(schema["tasks"][args.task]["num_classes"]),
            rows=rows,
        )
        return 0

    if args.command == "evaluate":
        result = evaluate_detector_head(
            dataset_dir=args.dataset_dir,
            detector_path=args.detector,
            head_path=args.head,
            output_path=args.output_path,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "evaluate-swap":
        result = evaluate_swap(
            dataset_dir=args.dataset_dir,
            original_detector_path=args.original_detector,
            swap_detector_path=args.swap_detector,
            head_path=args.head,
            output_path=args.output_path,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "official-cbm-status":
        print(json.dumps(backend_status(), sort_keys=True))
        return 0

    if args.command == "train-official-synthetic-detector":
        result = train_official_synthetic_detector(
            dataset_dir=args.dataset_dir,
            output_path=args.output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            image_size=args.image_size,
            pretrained=args.pretrained,
            freeze=args.freeze,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "train-official-synthetic-head":
        result = train_official_synthetic_head(
            dataset_dir=args.dataset_dir,
            task_name=args.task,
            output_path=args.output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            expand_dim=args.expand_dim,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "evaluate-official-synthetic":
        result = evaluate_official_synthetic(
            dataset_dir=args.dataset_dir,
            detector_path=args.detector,
            head_path=args.head,
            batch_size=args.batch_size,
            device=args.device,
            output_path=args.output_path,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "evaluate-official-synthetic-swap":
        result = evaluate_official_synthetic_swap(
            dataset_dir=args.dataset_dir,
            original_detector_path=args.original_detector,
            swap_detector_path=args.swap_detector,
            head_path=args.head,
            batch_size=args.batch_size,
            device=args.device,
            output_path=args.output_path,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "evaluate-official-oracle-head":
        result = evaluate_official_oracle_head(
            dataset_dir=args.dataset_dir,
            head_path=args.head,
            batch_size=args.batch_size,
            device=args.device,
            output_path=args.output_path,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "run-official-synthetic-experiment":
        result = run_official_synthetic_experiment(
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            data_seed=args.data_seed,
            task_name=args.task,
            detector_seeds=args.detector_seeds,
            detector_epochs=args.detector_epochs,
            head_epochs=args.head_epochs,
            batch_size=args.batch_size,
            detector_lr=args.detector_lr,
            head_lr=args.head_lr,
            device=args.device,
            detector_image_size=args.detector_image_size,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "run-official-manifest-experiment":
        result = run_official_manifest_experiment(
            train_dir=args.train_dir,
            eval_dir=args.eval_dir,
            output_dir=args.output_dir,
            task_name=args.task,
            detector_seeds=args.detector_seeds,
            detector_epochs=args.detector_epochs,
            head_epochs=args.head_epochs,
            batch_size=args.batch_size,
            detector_lr=args.detector_lr,
            head_lr=args.head_lr,
            device=args.device,
            detector_image_size=args.detector_image_size,
            pretrained=args.pretrained,
            freeze=args.freeze,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "run-shared-extractor-multihead-manifest-experiment":
        result = run_shared_extractor_multihead_manifest_experiment(
            train_dir=args.train_dir,
            eval_dir=args.eval_dir,
            output_dir=args.output_dir,
            task_name=args.task,
            head_seeds=args.head_seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            concept_loss_weight=args.concept_loss_weight,
            task_loss_weight=args.task_loss_weight,
            device=args.device,
            detector_image_size=args.detector_image_size,
            pretrained=args.pretrained,
            freeze=args.freeze,
            seed=args.seed,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "build-cub-manifest":
        class_ids = args.class_ids
        if class_ids is None and args.num_classes is not None:
            class_ids = list(
                range(args.class_start, args.class_start + args.num_classes)
            )
        result = build_cub_manifest(
            cub_root=args.cub_root,
            output_dir=args.output_dir,
            num_attributes=args.num_attributes,
            class_ids=class_ids,
            attribute_policy=args.attribute_policy,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "build-awa2-manifest":
        result = build_awa2_manifest(
            awa2_root=args.awa2_root,
            output_dir=args.output_dir,
            class_ids=args.class_ids,
            num_classes=args.num_classes,
            class_start=args.class_start,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            attribute_kind=args.attribute_kind,
            continuous_threshold=args.continuous_threshold,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "train-multihead-cbm":
        reliability_scores = None
        if args.eta > 0:
            if args.reliability_detector is None:
                parser.error("--eta > 0 requires --reliability-detector")
            from symbol_sanity.multihead import compute_concept_reliability_scores

            reliability_scores = compute_concept_reliability_scores(
                dataset_dir=args.reliability_dataset_dir or args.train_dir,
                detector_path=args.reliability_detector,
                batch_size=args.batch_size,
                device=args.device,
            )
        result = train_multihead_cbm(
            train_dir=args.train_dir,
            task_name=args.task,
            output_dir=args.output_dir,
            num_heads=args.num_heads,
            lambda_concept=args.lambda_concept,
            epochs=args.epochs,
            batch_size=args.batch_size,
            detector_lr=args.detector_lr,
            head_lr=args.head_lr,
            seed=args.seed,
            device=args.device,
            image_size=args.image_size,
            pretrained=args.pretrained,
            freeze=args.freeze,
            eta=args.eta,
            beta=args.beta,
            concept_dropout=args.concept_dropout,
            reliability_scores=reliability_scores,
            init_detector_path=args.init_detector,
            init_head_paths=args.init_heads,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "evaluate-uncertainty":
        result = evaluate_ensemble_uncertainty(
            dataset_dir=args.dataset_dir,
            detector_paths=args.detectors,
            head_path=args.head,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            num_worked_examples=args.num_worked_examples,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "run-reliability-experiment":
        result = run_reliability_comparison_experiment(
            train_dir=args.train_dir,
            eval_dir=args.eval_dir,
            output_dir=args.output_dir,
            task_name=args.task,
            seeds=args.seeds,
            num_heads=args.num_heads,
            lambda_concept=args.lambda_concept,
            eta=args.eta,
            beta=args.beta,
            concept_dropout=args.concept_dropout,
            epochs=args.epochs,
            reliability_epochs=args.reliability_epochs,
            batch_size=args.batch_size,
            detector_lr=args.detector_lr,
            head_lr=args.head_lr,
            device=args.device,
            detector_image_size=args.detector_image_size,
            pretrained=args.pretrained,
            freeze=args.freeze,
            arms=args.arms,
            val_dir=args.val_dir,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "statistical-report":
        result = generate_statistical_report(
            summary_path=args.summary_path,
            output_dir=args.output_dir,
            num_permutations=args.num_permutations,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "plot-report":
        result = generate_plot_report(
            summary_path=args.summary_path,
            statistical_report_dir=args.statistical_report_dir,
            output_dir=args.output_dir,
            formats=args.formats,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "experiment-environment":
        result = experiment_environment()
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def experiment_environment() -> dict[str, object]:
    metadata: dict[str, object] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "official_cbm_backend": backend_status(),
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git_output(["git", "status", "--short"]),
    }
    try:
        import torch

        metadata["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ModuleNotFoundError:
        metadata["torch"] = None
    return metadata


def _git_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
