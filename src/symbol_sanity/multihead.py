"""Multi-head and reliability-aware CBM training.

Implements the paper's mitigation strategy: a shared concept detector trained
jointly against an ensemble of independently initialized classification heads
(`J_multi`), optionally extended with a reliability-aware penalty that
discourages heads from relying on globally or instance-wise unreliable symbols
(`J_rel`).
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from symbol_sanity.io import load_dataset, read_json, require_concept_names, write_json
from symbol_sanity.logging_utils import log as _log
from symbol_sanity.metrics import accuracy, concept_agreement, macro_f1
from symbol_sanity.neural_synthetic import (
    _set_seed,
    evaluate_official_oracle_head,
)
from symbol_sanity.official_cbm import OfficialCBMSpec, build_official_cbm
from symbol_sanity.torch_runtime import (
    _SyntheticTorchDataset,
    build_from_checkpoint as _build_from_checkpoint,
    loader_num_workers,
    load_torch_checkpoint as _load_torch_checkpoint,
    torch_module as _torch,
)
from symbol_sanity.statistical_report import balanced_accuracy
from symbol_sanity.uncertainty import (
    collect_detector_probabilities,
    compute_epistemic_table,
    evaluate_ensemble_uncertainty,
)


def train_multihead_cbm(
    *,
    train_dir: Path,
    task_name: str,
    output_dir: Path,
    num_heads: int,
    lambda_concept: float,
    epochs: int,
    batch_size: int,
    detector_lr: float,
    head_lr: float,
    seed: int,
    device: str,
    image_size: int,
    pretrained: bool = False,
    freeze: bool = False,
    expand_dim: int = 0,
    eta: float = 0.0,
    beta: float = 0.0,
    concept_dropout: float = 0.0,
    reliability_scores: list[float] | None = None,
    epistemic_table: Any = None,
    init_detector_path: Path | None = None,
    init_head_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Jointly train a shared `X -> C` detector with `R` `C -> Y` heads.

    With ``num_heads=1`` and ``eta=0`` this is the standard joint CBM objective.
    With ``eta > 0`` the reliability-aware sensitivity penalty is added, which
    requires ``reliability_scores`` (per-concept reliability in `[0, 1]`) and,
    when ``beta > 0``, a per-example ``epistemic_table`` of shape `[N, K]`.
    ``concept_dropout`` applies an independent dropout mask on the concept
    probabilities per head and batch; linear heads trained on identical inputs
    converge to the same convex solution, so dropout is what keeps the head
    ensemble diverse throughout training.
    """

    torch = _torch()
    if num_heads < 1:
        raise ValueError("num_heads must be >= 1")
    if not 0.0 <= concept_dropout < 1.0:
        raise ValueError("concept_dropout must be in [0, 1)")
    schema, rows = load_dataset(train_dir)
    concept_names = list(schema["concept_names"])
    if task_name not in schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in dataset schema")
    num_classes = int(schema["tasks"][task_name]["num_classes"])

    if eta > 0.0 and reliability_scores is None:
        raise ValueError("eta > 0 requires reliability_scores")
    if reliability_scores is not None and len(reliability_scores) != len(concept_names):
        raise ValueError(
            "reliability_scores length must match the number of concepts"
        )
    if epistemic_table is not None:
        epistemic_table = torch.as_tensor(epistemic_table, dtype=torch.float32)
        if tuple(epistemic_table.shape) != (len(rows), len(concept_names)):
            raise ValueError(
                "epistemic_table must have shape [num_examples, num_concepts]"
            )

    _set_seed(seed)
    detector_spec = OfficialCBMSpec(
        mode="Concept_XtoC",
        n_attributes=len(concept_names),
        num_classes=num_classes,
        pretrained=pretrained,
        freeze=freeze,
        use_aux=False,
    )
    detector = build_official_cbm(detector_spec).to(device)
    if init_detector_path is not None:
        checkpoint = _load_torch_checkpoint(init_detector_path, device)
        require_concept_names(list(checkpoint["concept_names"]), concept_names)
        detector.load_state_dict(checkpoint["state_dict"])

    head_spec = OfficialCBMSpec(
        mode="Independent_CtoY",
        n_attributes=len(concept_names),
        num_classes=num_classes,
        expand_dim=expand_dim,
    )
    heads = []
    for head_index in range(num_heads):
        _set_seed(seed * 1000 + head_index + 1)
        head = build_official_cbm(head_spec).to(device)
        if init_head_paths is not None:
            checkpoint = _load_torch_checkpoint(init_head_paths[head_index], device)
            require_concept_names(list(checkpoint["concept_names"]), concept_names)
            head.load_state_dict(checkpoint["state_dict"])
        heads.append(head)
    _set_seed(seed)

    loader = _make_indexed_loader(
        dataset_dir=train_dir,
        rows=rows,
        batch_size=batch_size,
        seed=seed,
        image_size=image_size,
        task_name=task_name,
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    parameter
                    for parameter in detector.parameters()
                    if parameter.requires_grad
                ],
                "lr": detector_lr,
            },
            {
                "params": [
                    parameter for head in heads for parameter in head.parameters()
                ],
                "lr": head_lr,
            },
        ]
    )
    bce = torch.nn.BCEWithLogitsLoss()

    reliability = None
    if reliability_scores is not None:
        reliability = torch.tensor(
            reliability_scores, dtype=torch.float32, device=device
        ).clamp(0.0, 1.0)
    if epistemic_table is not None:
        epistemic_table = epistemic_table.to(device)

    detector.train()
    for head in heads:
        head.train()

    _log(
        "train multi-head CBM "
        f"dataset={train_dir} task={task_name} examples={len(rows)} "
        f"concepts={len(concept_names)} classes={num_classes} heads={num_heads} "
        f"lambda={lambda_concept} eta={eta} beta={beta} "
        f"concept_dropout={concept_dropout} epochs={epochs} "
        f"batch_size={batch_size} seed={seed} device={device}"
    )
    final_task_loss = 0.0
    final_concept_loss = 0.0
    final_penalty = 0.0
    for epoch in range(epochs):
        epoch_start = time.time()
        batch_count = 0
        for batch in loader:
            batch_count += 1
            images = batch["image"].to(device)
            concepts = batch["concept"].to(device)
            labels = batch["label"].to(device)
            indices = batch["index"].to(device)

            concept_logits = torch.cat(detector(images), dim=1)
            concept_probs = torch.sigmoid(concept_logits)
            concept_loss = bce(concept_logits, concepts)

            per_sample_losses = []
            for head in heads:
                head_input = concept_probs
                if concept_dropout > 0.0:
                    head_input = torch.nn.functional.dropout(
                        concept_probs, p=concept_dropout, training=True
                    )
                label_logits = head(head_input)
                per_sample_losses.append(
                    torch.nn.functional.cross_entropy(
                        label_logits, labels, reduction="none"
                    )
                )
            task_loss = torch.stack(
                [losses.mean() for losses in per_sample_losses]
            ).mean()
            loss = task_loss + lambda_concept * concept_loss

            penalty = None
            if eta > 0.0:
                omega = (1.0 - reliability).unsqueeze(0).expand(
                    concept_probs.shape[0], -1
                )
                if beta > 0.0 and epistemic_table is not None:
                    omega = omega + beta * epistemic_table[indices]
                sensitivity = torch.zeros(
                    concept_probs.shape[0], device=concept_probs.device
                )
                for losses in per_sample_losses:
                    grads = torch.autograd.grad(
                        losses.sum(),
                        concept_probs,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    sensitivity = sensitivity + (omega * grads.abs()).sum(dim=1)
                penalty = (sensitivity / num_heads).mean()
                loss = loss + eta * penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_task_loss = float(task_loss.detach().cpu().item())
            final_concept_loss = float(concept_loss.detach().cpu().item())
            final_penalty = (
                0.0 if penalty is None else float(penalty.detach().cpu().item())
            )
        _log(
            "multi-head epoch "
            f"{epoch + 1}/{epochs} seed={seed} batches={batch_count} "
            f"task_loss={final_task_loss:.6f} concept_loss={final_concept_loss:.6f} "
            f"penalty={final_penalty:.6f} elapsed_s={time.time() - epoch_start:.1f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    detector_path = output_dir / "detector.pt"
    training_metadata = {
        "num_heads": num_heads,
        "lambda_concept": lambda_concept,
        "eta": eta,
        "beta": beta,
        "concept_dropout": concept_dropout,
        "epochs": epochs,
        "seed": seed,
        "task_name": task_name,
        "reliability_scores": reliability_scores,
        "init_detector_path": (
            None if init_detector_path is None else str(init_detector_path)
        ),
    }
    torch.save(
        {
            "component": "multihead_detector",
            "interface": "concept_logits",
            "concept_output": "sigmoid_probabilities_for_head",
            "concept_names": concept_names,
            "image_size": image_size,
            "model_spec": asdict(detector_spec),
            "training": training_metadata,
            "state_dict": detector.state_dict(),
        },
        detector_path,
    )
    head_paths = []
    for head_index, head in enumerate(heads):
        head_path = output_dir / f"head_{head_index}.pt"
        head_paths.append(head_path)
        torch.save(
            {
                "component": "multihead_head",
                "interface": "concept_probabilities_to_label_logits",
                "task_name": task_name,
                "num_classes": num_classes,
                "concept_names": concept_names,
                "head_index": head_index,
                "model_spec": asdict(head_spec),
                "training": training_metadata,
                "state_dict": head.state_dict(),
            },
            head_path,
        )

    result = {
        "output_dir": str(output_dir),
        "detector_path": str(detector_path),
        "head_paths": [str(path) for path in head_paths],
        "train_dir": str(train_dir),
        "task_name": task_name,
        "num_examples": len(rows),
        "num_heads": num_heads,
        "lambda_concept": lambda_concept,
        "eta": eta,
        "beta": beta,
        "concept_dropout": concept_dropout,
        "epochs": epochs,
        "seed": seed,
        "final_task_loss": final_task_loss,
        "final_concept_loss": final_concept_loss,
        "final_penalty": final_penalty,
    }
    write_json(output_dir / "train_summary.json", result)
    return result


def compute_concept_reliability_scores(
    *,
    dataset_dir: Path,
    detector_path: Path,
    batch_size: int,
    device: str,
) -> list[float]:
    """Per-concept balanced detection accuracy `r_k` for one detector checkpoint.

    Balanced accuracy is robust to the strong prevalence imbalance of real
    attribute sets (e.g. CUB), unlike F1. Concepts that are constant in the
    validation split have undefined balanced accuracy and fall back to the
    chance level 0.5.
    """

    schema, rows = load_dataset(dataset_dir)
    probabilities = collect_detector_probabilities(
        dataset_dir=dataset_dir,
        detector_path=detector_path,
        batch_size=batch_size,
        device=device,
    )
    predictions = (probabilities >= 0.5).int().tolist()
    scores = []
    for concept_index in range(len(schema["concept_names"])):
        y_true = [int(row["concept_vector"][concept_index]) for row in rows]
        y_pred = [row[concept_index] for row in predictions]
        score = balanced_accuracy(y_true, y_pred)
        scores.append(0.5 if score is None else score)
    return scores


def run_reliability_comparison_experiment(
    *,
    train_dir: Path,
    eval_dir: Path,
    output_dir: Path,
    task_name: str,
    seeds: list[int],
    num_heads: int,
    lambda_concept: float,
    eta: float,
    beta: float,
    epochs: int,
    reliability_epochs: int,
    batch_size: int,
    detector_lr: float,
    head_lr: float,
    device: str,
    detector_image_size: int,
    pretrained: bool = False,
    freeze: bool = False,
    concept_dropout: float = 0.0,
    arms: list[str] | None = None,
    val_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and compare joint, multi-head, and reliability-aware CBM arms.

    ``concept_dropout`` is applied to the multihead and reliability arms only;
    the joint arm stays the standard single-head formulation.

    Seed runs whose checkpoints already exist under ``output_dir`` are reused
    instead of retrained, so arms can run as separate cluster jobs sharing one
    output directory: the reliability arm picks up the multihead phase-1
    checkpoints written by an earlier job. Delete a ``seed_*`` directory to
    force retraining after a configuration change.

    Each arm trains one CBM per seed, then reuses the standard evaluation
    battery: matched/swapped head-detector matrix on ``eval_dir``, oracle head
    baselines, and the ensemble uncertainty report. Each arm writes a
    ``summary.json`` compatible with the ``statistical-report`` subcommand.
    Reliability scores are computed on ``val_dir`` (defaults to ``train_dir``;
    never use the test split here).
    """

    requested_arms = list(arms) if arms else ["joint", "multihead", "reliability"]
    known_arms = {"joint", "multihead", "reliability"}
    unknown = sorted(set(requested_arms) - known_arms)
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}")
    validation_dir = val_dir if val_dir is not None else train_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    arm_results: dict[str, Any] = {}

    # The reliability arm warm-starts from the multihead phase-1 checkpoints,
    # so those are trained (or reused) even when only "reliability" is requested.
    multihead_runs: dict[int, dict[str, Any]] = {}
    if "multihead" in requested_arms or "reliability" in requested_arms:
        for seed in seeds:
            seed_dir = output_dir / "arms" / "multihead" / f"seed_{seed}"
            multihead_runs[seed] = _train_or_load(
                seed_dir,
                lambda seed=seed, seed_dir=seed_dir: train_multihead_cbm(
                    train_dir=train_dir,
                    task_name=task_name,
                    output_dir=seed_dir,
                    num_heads=num_heads,
                    lambda_concept=lambda_concept,
                    epochs=epochs,
                    batch_size=batch_size,
                    detector_lr=detector_lr,
                    head_lr=head_lr,
                    seed=seed,
                    device=device,
                    image_size=detector_image_size,
                    pretrained=pretrained,
                    freeze=freeze,
                    concept_dropout=concept_dropout,
                ),
            )

    for arm in ("joint", "multihead", "reliability"):
        if arm not in requested_arms:
            continue
        arm_dir = output_dir / "arms" / arm
        _log(f"=== arm {arm} ===")
        runs: dict[int, dict[str, Any]] = {}
        if arm == "joint":
            for seed in seeds:
                seed_dir = arm_dir / f"seed_{seed}"
                runs[seed] = _train_or_load(
                    seed_dir,
                    lambda seed=seed, seed_dir=seed_dir: train_multihead_cbm(
                        train_dir=train_dir,
                        task_name=task_name,
                        output_dir=seed_dir,
                        num_heads=1,
                        lambda_concept=lambda_concept,
                        epochs=epochs,
                        batch_size=batch_size,
                        detector_lr=detector_lr,
                        head_lr=head_lr,
                        seed=seed,
                        device=device,
                        image_size=detector_image_size,
                        pretrained=pretrained,
                        freeze=freeze,
                    ),
                )
        elif arm == "multihead":
            runs = multihead_runs
        else:
            epistemic_cache: dict[str, Any] = {}

            def _reliability_train(seed: int, seed_dir: Path) -> dict[str, Any]:
                phase1 = multihead_runs[seed]
                _log(
                    f"compute reliability scores seed={seed} val_dir={validation_dir}"
                )
                reliability_scores = compute_concept_reliability_scores(
                    dataset_dir=validation_dir,
                    detector_path=Path(phase1["detector_path"]),
                    batch_size=batch_size,
                    device=device,
                )
                epistemic_table = None
                if beta > 0.0 and len(seeds) >= 2:
                    if "table" not in epistemic_cache:
                        _log("compute train epistemic table from phase-1 detectors")
                        epistemic_cache["table"] = compute_epistemic_table(
                            dataset_dir=train_dir,
                            detector_paths=[
                                Path(multihead_runs[other]["detector_path"])
                                for other in seeds
                            ],
                            batch_size=batch_size,
                            device=device,
                        )
                    epistemic_table = epistemic_cache["table"]
                return train_multihead_cbm(
                    train_dir=train_dir,
                    task_name=task_name,
                    output_dir=seed_dir,
                    num_heads=num_heads,
                    lambda_concept=lambda_concept,
                    epochs=reliability_epochs,
                    batch_size=batch_size,
                    detector_lr=detector_lr,
                    head_lr=head_lr,
                    seed=seed,
                    device=device,
                    image_size=detector_image_size,
                    pretrained=pretrained,
                    freeze=freeze,
                    eta=eta,
                    beta=beta,
                    concept_dropout=concept_dropout,
                    reliability_scores=reliability_scores,
                    epistemic_table=epistemic_table,
                    init_detector_path=Path(phase1["detector_path"]),
                    init_head_paths=[Path(path) for path in phase1["head_paths"]],
                )

            for seed in seeds:
                seed_dir = arm_dir / f"seed_{seed}"
                runs[seed] = _train_or_load(
                    seed_dir,
                    lambda seed=seed, seed_dir=seed_dir: _reliability_train(
                        seed, seed_dir
                    ),
                )
        arm_results[arm] = _evaluate_arm(
            arm=arm,
            arm_dir=arm_dir,
            runs=runs,
            seeds=seeds,
            eval_dir=eval_dir,
            batch_size=batch_size,
            device=device,
        )

    # When arms run as separate jobs sharing the output directory, fold the
    # already-completed arms into the comparison so the last job leaves a
    # complete comparison.json.
    for arm in ("joint", "multihead", "reliability"):
        if arm in arm_results:
            continue
        summary_path = output_dir / "arms" / arm / "summary.json"
        if summary_path.exists():
            previous = read_json(summary_path)
            arm_results[arm] = {
                "summary_path": str(summary_path),
                "aggregate": previous.get("aggregate", {}),
            }

    comparison = {
        "output_dir": str(output_dir),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "val_dir": str(validation_dir),
        "task_name": task_name,
        "seeds": seeds,
        "num_heads": num_heads,
        "lambda_concept": lambda_concept,
        "eta": eta,
        "beta": beta,
        "concept_dropout": concept_dropout,
        "epochs": epochs,
        "reliability_epochs": reliability_epochs,
        "batch_size": batch_size,
        "detector_lr": detector_lr,
        "head_lr": head_lr,
        "detector_image_size": detector_image_size,
        "pretrained": pretrained,
        "freeze": freeze,
        "arms": {arm: result["aggregate"] for arm, result in arm_results.items()},
        "arm_summaries": {
            arm: result["summary_path"] for arm, result in arm_results.items()
        },
    }
    write_json(output_dir / "comparison.json", comparison)
    _log(f"wrote comparison path={output_dir / 'comparison.json'}")
    return comparison


def _train_or_load(seed_dir: Path, train_fn: Any) -> dict[str, Any]:
    """Reuse an existing seed run if its checkpoints are already on disk."""

    summary_path = seed_dir / "train_summary.json"
    if summary_path.exists() and (seed_dir / "detector.pt").exists():
        _log(f"reusing existing run at {seed_dir} (delete it to force retraining)")
        return read_json(summary_path)
    return train_fn()


def _evaluate_arm(
    *,
    arm: str,
    arm_dir: Path,
    runs: dict[int, dict[str, Any]],
    seeds: list[int],
    eval_dir: Path,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Swap matrix, oracle baselines, and uncertainty report for one arm.

    Detector probabilities over the evaluation set are computed once per
    detector and reused for every head pairing and the uncertainty report,
    so the arm needs only one detector pass per seed.
    """

    torch = _torch()
    detector_paths = {seed: Path(runs[seed]["detector_path"]) for seed in seeds}
    head_paths = {seed: Path(runs[seed]["head_paths"][0]) for seed in seeds}
    primary_seed = seeds[0]
    primary_head_path = head_paths[primary_seed]

    schema, rows = load_dataset(eval_dir)
    concept_names = list(schema["concept_names"])
    _log(f"arm={arm} cache detector probabilities ({len(seeds)} detector passes)")
    probs_by_seed = {
        seed: collect_detector_probabilities(
            dataset_dir=eval_dir,
            detector_path=detector_paths[seed],
            batch_size=batch_size,
            device=device,
        )
        for seed in seeds
    }
    oracle_concepts = [[int(value) for value in row["concept_vector"]] for row in rows]
    agreement_by_seed = {
        seed: concept_agreement(
            (probs_by_seed[seed] >= 0.5).int().tolist(),
            oracle_concepts,
        )
        for seed in seeds
    }

    heads = {}
    task_name = None
    num_classes = None
    for seed in seeds:
        checkpoint = _load_torch_checkpoint(head_paths[seed], device)
        require_concept_names(list(checkpoint["concept_names"]), concept_names)
        task_name = str(checkpoint["task_name"])
        num_classes = int(checkpoint["num_classes"])
        head = _build_from_checkpoint(checkpoint)
        head.load_state_dict(checkpoint["state_dict"])
        head.eval()
        heads[seed] = head
    labels = [int(row["task_labels"][task_name]) for row in rows]

    head_evaluations = {
        str(seed): evaluate_official_oracle_head(
            dataset_dir=eval_dir,
            head_path=head_paths[seed],
            batch_size=batch_size,
            device=device,
        )
        for seed in seeds
    }

    _log(f"arm={arm} evaluate full head x detector matrix from cached probabilities")
    head_detector_rows = []
    for head_seed in seeds:
        for detector_seed in seeds:
            row_accuracy, row_macro_f1 = _evaluate_head_on_probabilities(
                head=heads[head_seed],
                probabilities=probs_by_seed[detector_seed],
                labels=labels,
                num_classes=num_classes,
                batch_size=batch_size,
            )
            head_detector_rows.append(
                {
                    "head_seed": head_seed,
                    "detector_seed": detector_seed,
                    "head_path": str(head_paths[head_seed]),
                    "detector_path": str(detector_paths[detector_seed]),
                    "accuracy": row_accuracy,
                    "macro_f1": row_macro_f1,
                    "concept_agreement_with_oracle": agreement_by_seed[detector_seed],
                    "is_matched_seed_pair": head_seed == detector_seed,
                }
            )

    detector_evaluations = {
        str(row["detector_seed"]): {
            "dataset_dir": str(eval_dir),
            "detector_path": row["detector_path"],
            "head_path": str(primary_head_path),
            "task_name": task_name,
            "num_examples": len(rows),
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
            "concept_agreement_with_oracle": row["concept_agreement_with_oracle"],
        }
        for row in head_detector_rows
        if row["head_seed"] == primary_seed
    }

    swap_rows = []
    matched = {
        row["head_seed"]: row
        for row in head_detector_rows
        if row["is_matched_seed_pair"]
    }
    for row in head_detector_rows:
        original_accuracy = matched[row["head_seed"]]["accuracy"]
        swap_rows.append(
            {
                "head_seed": row["head_seed"],
                "detector_seed": row["detector_seed"],
                "original_accuracy": original_accuracy,
                "swapped_accuracy": row["accuracy"],
                "swap_drop": original_accuracy - row["accuracy"],
                "relative_retention": (
                    0.0
                    if original_accuracy == 0
                    else row["accuracy"] / original_accuracy
                ),
            }
        )

    uncertainty_report = None
    if len(seeds) >= 2:
        uncertainty_report = evaluate_ensemble_uncertainty(
            dataset_dir=eval_dir,
            detector_paths=[detector_paths[seed] for seed in seeds],
            head_path=primary_head_path,
            batch_size=batch_size,
            device=device,
            output_dir=arm_dir / "uncertainty",
            member_probs=torch.stack([probs_by_seed[seed] for seed in seeds]),
        )
    else:
        _log(f"arm={arm} skipping uncertainty report: needs >= 2 seeds")

    matched_accuracies = [row["accuracy"] for row in matched.values()]
    off_diagonal = [
        row for row in head_detector_rows if not row["is_matched_seed_pair"]
    ]
    aggregate = {
        "matched_accuracy_mean": _mean(matched_accuracies),
        "swapped_accuracy_mean": _mean([row["accuracy"] for row in off_diagonal]),
        "swap_drop_mean": _mean(
            [
                row["swap_drop"]
                for row in swap_rows
                if row["head_seed"] != row["detector_seed"]
            ]
        ),
        "relative_retention_mean": _mean(
            [
                row["relative_retention"]
                for row in swap_rows
                if row["head_seed"] != row["detector_seed"]
            ]
        ),
        "concept_agreement_mean": _mean(
            [row["concept_agreement_with_oracle"] for row in matched.values()]
        ),
        "oracle_head_accuracy_mean": _mean(
            [evaluation["accuracy"] for evaluation in head_evaluations.values()]
        ),
    }
    if uncertainty_report is not None:
        aggregate.update(
            {
                "ensemble_accuracy": uncertainty_report["ensemble_accuracy"],
                "mean_symbol_epistemic": uncertainty_report["mean_symbol_epistemic"],
                "mean_label_epistemic": uncertainty_report["mean_label_epistemic"],
                "uncertainty_correlations": uncertainty_report["correlations"],
            }
        )

    summary = {
        "arm": arm,
        "eval_dir": str(eval_dir),
        "task_name": detector_evaluations[str(primary_seed)]["task_name"],
        "primary_head_seed": primary_seed,
        "head_train": {"checkpoint_path": str(primary_head_path)},
        "train_runs": {str(seed): runs[seed] for seed in seeds},
        "detector_evaluations": detector_evaluations,
        "head_evaluations": head_evaluations,
        "head_detector_rows": head_detector_rows,
        "swap_rows": swap_rows,
        "aggregate": aggregate,
        "uncertainty_report": (
            None
            if uncertainty_report is None
            else str(arm_dir / "uncertainty" / "uncertainty_report.json")
        ),
    }
    summary_path = arm_dir / "summary.json"
    write_json(summary_path, summary)
    _log(f"arm={arm} wrote summary path={summary_path}")
    return {"summary_path": str(summary_path), "aggregate": aggregate}


def _evaluate_head_on_probabilities(
    *,
    head: Any,
    probabilities: Any,
    labels: list[int],
    num_classes: int,
    batch_size: int,
) -> tuple[float, float]:
    torch = _torch()
    y_pred: list[int] = []
    with torch.no_grad():
        for start in range(0, probabilities.shape[0], batch_size):
            logits = head(probabilities[start : start + batch_size])
            y_pred.extend(int(value) for value in logits.argmax(dim=1).tolist())
    return accuracy(labels, y_pred), macro_f1(labels, y_pred, num_classes)


class _IndexedDataset:
    """Wraps the manifest dataset so batches carry example indices."""

    def __init__(self, base: _SyntheticTorchDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _torch()
        item = self.base[index]
        item["index"] = torch.tensor(index, dtype=torch.long)
        return item


def _make_indexed_loader(
    *,
    dataset_dir: Path,
    rows: list[dict[str, Any]],
    batch_size: int,
    seed: int,
    image_size: int,
    task_name: str,
) -> Any:
    torch = _torch()
    dataset = _IndexedDataset(
        _SyntheticTorchDataset(
            dataset_dir=dataset_dir,
            rows=rows,
            image_size=image_size,
            include_images=True,
            task_name=task_name,
        )
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_workers = loader_num_workers()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)
