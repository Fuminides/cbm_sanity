"""Torch-backed synthetic CBM experiments using the official CUB architecture."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from symbol_sanity.io import load_dataset, require_concept_names, write_json
from symbol_sanity.logging_utils import log as _log
from symbol_sanity.metrics import accuracy, concept_agreement, macro_f1
from symbol_sanity.official_cbm import OfficialCBMSpec, build_official_cbm
from symbol_sanity.synthetic import export_dataset
from symbol_sanity.torch_runtime import (
    build_from_checkpoint as _build_from_checkpoint,
    load_torch_checkpoint as _load_torch_checkpoint,
    make_loader as _make_loader,
    torch_module as _torch,
)


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: str
    component: str
    dataset_dir: str
    num_examples: int
    epochs: int
    seed: int
    final_loss: float


def train_official_synthetic_detector(
    *,
    dataset_dir: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    image_size: int,
    pretrained: bool = False,
    freeze: bool = False,
) -> dict[str, Any]:
    """Train an official `X -> C` concept detector on synthetic metadata."""

    torch = _torch()
    schema, rows = load_dataset(dataset_dir)
    concept_names = list(schema["concept_names"])
    _set_seed(seed)

    model = build_official_cbm(
        OfficialCBMSpec(
            mode="Concept_XtoC",
            n_attributes=len(concept_names),
            num_classes=_max_num_classes(schema),
            pretrained=pretrained,
            freeze=freeze,
            use_aux=False,
        )
    ).to(device)
    model.train()

    loader = _make_loader(
        dataset_dir=dataset_dir,
        rows=rows,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        image_size=image_size,
        include_images=True,
        task_name=None,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    final_loss = 0.0

    _log(
        "train X->C detector "
        f"dataset={dataset_dir} examples={len(rows)} concepts={len(concept_names)} "
        f"epochs={epochs} batch_size={batch_size} seed={seed} device={device} "
        f"pretrained={pretrained} freeze={freeze}"
    )
    for epoch in range(epochs):
        epoch_start = time.time()
        batch_count = 0
        for batch in loader:
            batch_count += 1
            images = batch["image"].to(device)
            concepts = batch["concept"].to(device)
            outputs = model(images)
            logits = torch.cat(outputs, dim=1)
            loss = criterion(logits, concepts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
        _log(
            "detector epoch "
            f"{epoch + 1}/{epochs} seed={seed} batches={batch_count} "
            f"final_loss={final_loss:.6f} elapsed_s={time.time() - epoch_start:.1f}"
        )

    checkpoint = {
        "component": "official_synthetic_detector",
        "interface": "concept_logits",
        "concept_output": "sigmoid_probabilities_for_head",
        "concept_names": concept_names,
        "image_size": image_size,
        "model_spec": asdict(
            OfficialCBMSpec(
                mode="Concept_XtoC",
                n_attributes=len(concept_names),
                num_classes=_max_num_classes(schema),
                pretrained=pretrained,
                freeze=freeze,
                use_aux=False,
            )
        ),
        "state_dict": model.state_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    result = TrainResult(
        checkpoint_path=str(output_path),
        component="official_synthetic_detector",
        dataset_dir=str(dataset_dir),
        num_examples=len(rows),
        epochs=epochs,
        seed=seed,
        final_loss=final_loss,
    )
    return asdict(result)


def train_official_synthetic_head(
    *,
    dataset_dir: Path,
    task_name: str,
    output_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    expand_dim: int = 0,
) -> dict[str, Any]:
    """Train an official `C -> Y` MLP head from oracle concept vectors."""

    torch = _torch()
    schema, rows = load_dataset(dataset_dir)
    concept_names = list(schema["concept_names"])
    if task_name not in schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in dataset schema")
    num_classes = int(schema["tasks"][task_name]["num_classes"])
    _set_seed(seed)

    model = build_official_cbm(
        OfficialCBMSpec(
            mode="Independent_CtoY",
            n_attributes=len(concept_names),
            num_classes=num_classes,
            expand_dim=expand_dim,
        )
    ).to(device)
    model.train()

    loader = _make_loader(
        dataset_dir=dataset_dir,
        rows=rows,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        image_size=64,
        include_images=False,
        task_name=task_name,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    final_loss = 0.0

    _log(
        "train C->Y head "
        f"dataset={dataset_dir} task={task_name} examples={len(rows)} "
        f"concepts={len(concept_names)} classes={num_classes} epochs={epochs} "
        f"batch_size={batch_size} seed={seed} device={device}"
    )
    log_every = max(1, min(25, epochs // 10 if epochs >= 10 else 1))
    for epoch in range(epochs):
        epoch_start = time.time()
        batch_count = 0
        for batch in loader:
            batch_count += 1
            concepts = batch["concept"].to(device)
            labels = batch["label"].to(device)
            logits = model(concepts)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % log_every == 0:
            _log(
                "head epoch "
                f"{epoch + 1}/{epochs} batches={batch_count} "
                f"final_loss={final_loss:.6f} elapsed_s={time.time() - epoch_start:.1f}"
            )

    checkpoint = {
        "component": "official_synthetic_head",
        "interface": "concept_probabilities_to_label_logits",
        "task_name": task_name,
        "num_classes": num_classes,
        "concept_names": concept_names,
        "model_spec": asdict(
            OfficialCBMSpec(
                mode="Independent_CtoY",
                n_attributes=len(concept_names),
                num_classes=num_classes,
                expand_dim=expand_dim,
            )
        ),
        "state_dict": model.state_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    result = TrainResult(
        checkpoint_path=str(output_path),
        component="official_synthetic_head",
        dataset_dir=str(dataset_dir),
        num_examples=len(rows),
        epochs=epochs,
        seed=seed,
        final_loss=final_loss,
    )
    return asdict(result)


def run_shared_extractor_multihead_manifest_experiment(
    *,
    train_dir: Path,
    eval_dir: Path,
    output_dir: Path,
    task_name: str,
    head_seeds: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    concept_loss_weight: float,
    task_loss_weight: float,
    device: str,
    detector_image_size: int,
    pretrained: bool = False,
    freeze: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Train one shared `X -> C` extractor jointly with several `C -> Y` heads."""

    torch = _torch()
    _log(
        "start shared-extractor multi-head experiment "
        f"train_dir={train_dir} eval_dir={eval_dir} output_dir={output_dir} "
        f"task={task_name} head_seeds={head_seeds}"
    )
    if not head_seeds:
        raise ValueError("head_seeds must contain at least one seed")

    output_dir.mkdir(parents=True, exist_ok=True)
    detector_dir = output_dir / "detectors"
    head_dir = output_dir / "heads"
    detector_dir.mkdir(exist_ok=True)
    head_dir.mkdir(exist_ok=True)

    train_schema, train_rows = load_dataset(train_dir)
    eval_schema, eval_rows = load_dataset(eval_dir)
    concept_names = list(train_schema["concept_names"])
    require_concept_names(concept_names, list(eval_schema["concept_names"]))
    if task_name not in train_schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in train schema")
    if task_name not in eval_schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in eval schema")
    num_classes = int(train_schema["tasks"][task_name]["num_classes"])
    _set_seed(seed)

    detector_spec = OfficialCBMSpec(
        mode="Concept_XtoC",
        n_attributes=len(concept_names),
        num_classes=_max_num_classes(train_schema),
        pretrained=pretrained,
        freeze=freeze,
        use_aux=False,
    )
    head_spec = OfficialCBMSpec(
        mode="Independent_CtoY",
        n_attributes=len(concept_names),
        num_classes=num_classes,
    )
    detector = build_official_cbm(detector_spec).to(device)
    heads = {}
    for head_seed in head_seeds:
        _set_seed(head_seed)
        heads[head_seed] = build_official_cbm(head_spec).to(device)
    _set_seed(seed)

    detector.train()
    for head in heads.values():
        head.train()

    loader = _make_loader(
        dataset_dir=train_dir,
        rows=train_rows,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        image_size=detector_image_size,
        include_images=True,
        task_name=task_name,
    )
    parameters = [
        parameter for parameter in detector.parameters() if parameter.requires_grad
    ]
    for head in heads.values():
        parameters.extend(head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    concept_criterion = torch.nn.BCEWithLogitsLoss()
    task_criterion = torch.nn.CrossEntropyLoss()
    final_loss = 0.0
    final_concept_loss = 0.0
    final_task_loss = 0.0

    _log(
        "train shared X->C with multi heads "
        f"examples={len(train_rows)} concepts={len(concept_names)} "
        f"classes={num_classes} heads={len(heads)} epochs={epochs} "
        f"batch_size={batch_size} seed={seed} device={device} "
        f"concept_loss_weight={concept_loss_weight} task_loss_weight={task_loss_weight}"
    )
    log_every = max(1, min(25, epochs // 10 if epochs >= 10 else 1))
    for epoch in range(epochs):
        epoch_start = time.time()
        batch_count = 0
        for batch in loader:
            batch_count += 1
            images = batch["image"].to(device)
            concepts = batch["concept"].to(device)
            labels = batch["label"].to(device)

            concept_logits = torch.cat(detector(images), dim=1)
            concept_probs = torch.sigmoid(concept_logits)
            concept_loss = concept_criterion(concept_logits, concepts)
            task_losses = [
                task_criterion(head(concept_probs), labels)
                for head in heads.values()
            ]
            task_loss = torch.stack(task_losses).mean()
            loss = (
                concept_loss_weight * concept_loss
                + task_loss_weight * task_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = float(loss.detach().cpu().item())
            final_concept_loss = float(concept_loss.detach().cpu().item())
            final_task_loss = float(task_loss.detach().cpu().item())

        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % log_every == 0:
            _log(
                "shared multi-head epoch "
                f"{epoch + 1}/{epochs} batches={batch_count} "
                f"loss={final_loss:.6f} concept_loss={final_concept_loss:.6f} "
                f"task_loss={final_task_loss:.6f} elapsed_s={time.time() - epoch_start:.1f}"
            )

    detector_path = detector_dir / f"shared_detector_seed_{seed}.pt"
    detector_checkpoint = {
        "component": "official_shared_multihead_detector",
        "interface": "concept_logits",
        "concept_output": "sigmoid_probabilities_for_head",
        "concept_names": concept_names,
        "image_size": detector_image_size,
        "model_spec": asdict(detector_spec),
        "state_dict": detector.state_dict(),
    }
    torch.save(detector_checkpoint, detector_path)

    head_paths = {}
    head_train_results = []
    for head_seed, head in heads.items():
        head_path = head_dir / f"shared_head_seed_{head_seed}.pt"
        head_paths[head_seed] = head_path
        checkpoint = {
            "component": "official_shared_multihead_head",
            "interface": "concept_probabilities_to_label_logits",
            "task_name": task_name,
            "num_classes": num_classes,
            "concept_names": concept_names,
            "model_spec": asdict(head_spec),
            "state_dict": head.state_dict(),
        }
        torch.save(checkpoint, head_path)
        head_train_results.append(
            {
                "checkpoint_path": str(head_path),
                "component": "official_shared_multihead_head",
                "dataset_dir": str(train_dir),
                "num_examples": len(train_rows),
                "epochs": epochs,
                "seed": head_seed,
                "final_loss": final_loss,
            }
        )

    detector_train = {
        "checkpoint_path": str(detector_path),
        "component": "official_shared_multihead_detector",
        "dataset_dir": str(train_dir),
        "num_examples": len(train_rows),
        "epochs": epochs,
        "seed": seed,
        "final_loss": final_loss,
        "final_concept_loss": final_concept_loss,
        "final_task_loss": final_task_loss,
    }

    _log("phase evaluate shared detector with all jointly trained heads")
    head_detector_rows = []
    model_rows = []
    head_evaluations = {}
    per_head_detector_evaluations = {}
    for head_seed, head_path in head_paths.items():
        evaluation = evaluate_official_synthetic(
            dataset_dir=eval_dir,
            detector_path=detector_path,
            head_path=head_path,
            batch_size=batch_size,
            device=device,
        )
        oracle_evaluation = evaluate_official_oracle_head(
            dataset_dir=eval_dir,
            head_path=head_path,
            batch_size=batch_size,
            device=device,
        )
        head_evaluations[str(head_seed)] = oracle_evaluation
        per_head_detector_evaluations[str(head_seed)] = evaluation
        row = {
            "head_seed": head_seed,
            "detector_seed": seed,
            "head_path": str(head_path),
            "detector_path": str(detector_path),
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "concept_agreement_with_oracle": evaluation[
                "concept_agreement_with_oracle"
            ],
            "is_matched_seed_pair": head_seed == seed,
            "training_regime": "shared_extractor_multihead",
        }
        head_detector_rows.append(row)
        model_rows.append(
            {
                "model_seed": head_seed,
                "head_path": str(head_path),
                "detector_path": str(detector_path),
                "accuracy": evaluation["accuracy"],
                "macro_f1": evaluation["macro_f1"],
                "concept_agreement_with_oracle": evaluation[
                    "concept_agreement_with_oracle"
                ],
                "head_oracle_accuracy": oracle_evaluation["accuracy"],
                "head_oracle_macro_f1": oracle_evaluation["macro_f1"],
                "training_regime": "shared_extractor_multihead",
            }
        )

    primary_head_seed = head_seeds[0]
    primary_detector_evaluation = per_head_detector_evaluations[str(primary_head_seed)]
    detector_evaluations = {str(seed): primary_detector_evaluation}
    swap_rows = [
        {
            "original_seed": seed,
            "swap_seed": seed,
            "original_detector_path": str(detector_path),
            "swap_detector_path": str(detector_path),
            "original_accuracy": primary_detector_evaluation["accuracy"],
            "swapped_accuracy": primary_detector_evaluation["accuracy"],
            "swap_drop": 0.0,
            "relative_retention": 1.0
            if primary_detector_evaluation["accuracy"] != 0
            else 0.0,
            "original_concept_agreement_with_oracle": primary_detector_evaluation[
                "concept_agreement_with_oracle"
            ],
            "swap_concept_agreement_with_oracle": primary_detector_evaluation[
                "concept_agreement_with_oracle"
            ],
            "training_regime": "shared_extractor_multihead",
        }
    ]
    result = {
        "output_dir": str(output_dir),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "task_name": task_name,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "num_concepts": len(concept_names),
        "num_classes": num_classes,
        "training_regime": "shared_extractor_multihead",
        "detector_seeds": [seed],
        "head_seeds": head_seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "concept_loss_weight": concept_loss_weight,
        "task_loss_weight": task_loss_weight,
        "device": device,
        "detector_image_size": detector_image_size,
        "pretrained": pretrained,
        "freeze": freeze,
        "head_train": head_train_results[0],
        "head_train_all": head_train_results,
        "primary_head_seed": primary_head_seed,
        "oracle_head_evaluation": head_evaluations[str(primary_head_seed)],
        "head_evaluations": head_evaluations,
        "detector_train": [detector_train],
        "detector_evaluations": detector_evaluations,
        "per_head_detector_evaluations": per_head_detector_evaluations,
        "swap_rows": swap_rows,
        "model_rows": model_rows,
        "head_detector_rows": head_detector_rows,
    }
    write_json(output_dir / "summary.json", result)
    _log(f"wrote shared multi-head summary path={output_dir / 'summary.json'}")
    return result


def evaluate_official_synthetic(
    *,
    dataset_dir: Path,
    detector_path: Path,
    head_path: Path,
    batch_size: int,
    device: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate an official synthetic detector feeding a frozen official head."""

    torch = _torch()
    schema, rows = load_dataset(dataset_dir)
    _log(
        "evaluate detector->head "
        f"dataset={dataset_dir} examples={len(rows)} detector={detector_path} "
        f"head={head_path} batch_size={batch_size} device={device}"
    )
    detector_checkpoint = _load_torch_checkpoint(detector_path, device)
    head_checkpoint = _load_torch_checkpoint(head_path, device)
    concept_names = list(schema["concept_names"])
    require_concept_names(list(detector_checkpoint["concept_names"]), concept_names)
    require_concept_names(list(head_checkpoint["concept_names"]), concept_names)

    task_name = str(head_checkpoint["task_name"])
    if task_name not in schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in dataset schema")

    detector = _build_from_checkpoint(detector_checkpoint).to(device)
    head = _build_from_checkpoint(head_checkpoint).to(device)
    detector.load_state_dict(detector_checkpoint["state_dict"])
    head.load_state_dict(head_checkpoint["state_dict"])
    detector.eval()
    head.eval()

    loader = _make_loader(
        dataset_dir=dataset_dir,
        rows=rows,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        image_size=int(detector_checkpoint["image_size"]),
        include_images=True,
        task_name=task_name,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    predicted_concepts: list[list[int]] = []
    oracle_concepts: list[list[int]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            concept_true = batch["concept"].to(device)
            concept_logits = torch.cat(detector(images), dim=1)
            concept_probs = torch.sigmoid(concept_logits)
            label_logits = head(concept_probs)
            predictions = torch.argmax(label_logits, dim=1)

            y_true.extend(int(value) for value in labels.cpu().tolist())
            y_pred.extend(int(value) for value in predictions.cpu().tolist())
            predicted_concepts.extend(
                (concept_probs.cpu() >= 0.5).int().tolist()
            )
            oracle_concepts.extend(concept_true.cpu().int().tolist())

    result = {
        "dataset_dir": str(dataset_dir),
        "detector_path": str(detector_path),
        "head_path": str(head_path),
        "task_name": task_name,
        "num_examples": len(rows),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(
            y_true,
            y_pred,
            int(head_checkpoint["num_classes"]),
        ),
        "concept_agreement_with_oracle": concept_agreement(
            predicted_concepts,
            oracle_concepts,
        ),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def evaluate_official_synthetic_swap(
    *,
    dataset_dir: Path,
    original_detector_path: Path,
    swap_detector_path: Path,
    head_path: Path,
    batch_size: int,
    device: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    original = evaluate_official_synthetic(
        dataset_dir=dataset_dir,
        detector_path=original_detector_path,
        head_path=head_path,
        batch_size=batch_size,
        device=device,
    )
    swapped = evaluate_official_synthetic(
        dataset_dir=dataset_dir,
        detector_path=swap_detector_path,
        head_path=head_path,
        batch_size=batch_size,
        device=device,
    )
    result = {
        "dataset_dir": str(dataset_dir),
        "head_path": str(head_path),
        "task_name": original["task_name"],
        "original_detector_path": str(original_detector_path),
        "swap_detector_path": str(swap_detector_path),
        "num_examples": original["num_examples"],
        "original_accuracy": original["accuracy"],
        "swapped_accuracy": swapped["accuracy"],
        "swap_drop": original["accuracy"] - swapped["accuracy"],
        "relative_retention": (
            0.0
            if original["accuracy"] == 0
            else swapped["accuracy"] / original["accuracy"]
        ),
        "original_concept_agreement_with_oracle": original[
            "concept_agreement_with_oracle"
        ],
        "swap_concept_agreement_with_oracle": swapped[
            "concept_agreement_with_oracle"
        ],
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def evaluate_official_oracle_head(
    *,
    dataset_dir: Path,
    head_path: Path,
    batch_size: int,
    device: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate an official `C -> Y` head on oracle manifest concept vectors."""

    torch = _torch()
    schema, rows = load_dataset(dataset_dir)
    _log(
        "evaluate oracle C->Y "
        f"dataset={dataset_dir} examples={len(rows)} head={head_path} "
        f"batch_size={batch_size} device={device}"
    )
    head_checkpoint = _load_torch_checkpoint(head_path, device)
    concept_names = list(schema["concept_names"])
    require_concept_names(list(head_checkpoint["concept_names"]), concept_names)

    task_name = str(head_checkpoint["task_name"])
    if task_name not in schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in dataset schema")

    head = _build_from_checkpoint(head_checkpoint).to(device)
    head.load_state_dict(head_checkpoint["state_dict"])
    head.eval()

    loader = _make_loader(
        dataset_dir=dataset_dir,
        rows=rows,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        image_size=64,
        include_images=False,
        task_name=task_name,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            concepts = batch["concept"].to(device)
            labels = batch["label"].to(device)
            label_logits = head(concepts)
            predictions = torch.argmax(label_logits, dim=1)
            y_true.extend(int(value) for value in labels.cpu().tolist())
            y_pred.extend(int(value) for value in predictions.cpu().tolist())

    result = {
        "dataset_dir": str(dataset_dir),
        "head_path": str(head_path),
        "task_name": task_name,
        "num_examples": len(rows),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(
            y_true,
            y_pred,
            int(head_checkpoint["num_classes"]),
        ),
        "concept_source": "oracle_manifest",
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def run_official_synthetic_experiment(
    *,
    output_dir: Path,
    num_examples: int,
    data_seed: int,
    task_name: str,
    detector_seeds: list[int],
    detector_epochs: int,
    head_epochs: int,
    batch_size: int,
    detector_lr: float,
    head_lr: float,
    device: str,
    detector_image_size: int,
) -> dict[str, Any]:
    """Run a reproducible official-CBM synthetic detector-swap experiment."""

    _log(f"start synthetic experiment output_dir={output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "data"
    detector_dir = output_dir / "detectors"
    detector_dir.mkdir(exist_ok=True)
    head_path = output_dir / "head.pt"

    export_dataset(
        output_dir=dataset_dir,
        num_examples=num_examples,
        seed=data_seed,
        image_size=64,
    )
    _log(f"synthetic data generated dataset_dir={dataset_dir}")
    head_train = train_official_synthetic_head(
        dataset_dir=dataset_dir,
        task_name=task_name,
        output_path=head_path,
        epochs=head_epochs,
        batch_size=batch_size,
        lr=head_lr,
        seed=data_seed,
        device=device,
    )

    detector_train_results = []
    detector_paths: dict[int, Path] = {}
    for detector_seed in detector_seeds:
        _log(f"start detector training seed={detector_seed}")
        detector_path = detector_dir / f"detector_seed_{detector_seed}.pt"
        detector_paths[detector_seed] = detector_path
        detector_train_results.append(
            train_official_synthetic_detector(
                dataset_dir=dataset_dir,
                output_path=detector_path,
                epochs=detector_epochs,
                batch_size=batch_size,
                lr=detector_lr,
                seed=detector_seed,
                device=device,
                image_size=detector_image_size,
            )
        )

    detector_evaluations = {
        str(detector_seed): evaluate_official_synthetic(
            dataset_dir=dataset_dir,
            detector_path=detector_path,
            head_path=head_path,
            batch_size=batch_size,
            device=device,
        )
        for detector_seed, detector_path in detector_paths.items()
    }
    _log("detector evaluations complete")

    swap_rows = []
    for original_seed, original_path in detector_paths.items():
        original_accuracy = detector_evaluations[str(original_seed)]["accuracy"]
        for swap_seed, swap_path in detector_paths.items():
            swapped_accuracy = detector_evaluations[str(swap_seed)]["accuracy"]
            swap_rows.append(
                {
                    "original_seed": original_seed,
                    "swap_seed": swap_seed,
                    "original_detector_path": str(original_path),
                    "swap_detector_path": str(swap_path),
                    "original_accuracy": original_accuracy,
                    "swapped_accuracy": swapped_accuracy,
                    "swap_drop": original_accuracy - swapped_accuracy,
                    "relative_retention": (
                        0.0
                        if original_accuracy == 0
                        else swapped_accuracy / original_accuracy
                    ),
                    "original_concept_agreement_with_oracle": detector_evaluations[
                        str(original_seed)
                    ]["concept_agreement_with_oracle"],
                    "swap_concept_agreement_with_oracle": detector_evaluations[
                        str(swap_seed)
                    ]["concept_agreement_with_oracle"],
                }
            )

    result = {
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "eval_dir": str(dataset_dir),
        "task_name": task_name,
        "num_examples": num_examples,
        "data_seed": data_seed,
        "detector_seeds": detector_seeds,
        "detector_epochs": detector_epochs,
        "head_epochs": head_epochs,
        "batch_size": batch_size,
        "detector_lr": detector_lr,
        "head_lr": head_lr,
        "device": device,
        "detector_image_size": detector_image_size,
        "head_train": head_train,
        "detector_train": detector_train_results,
        "detector_evaluations": detector_evaluations,
        "swap_rows": swap_rows,
    }
    write_json(output_dir / "summary.json", result)
    _log(f"wrote synthetic summary path={output_dir / 'summary.json'}")
    return result


def run_official_manifest_experiment(
    *,
    train_dir: Path,
    eval_dir: Path,
    output_dir: Path,
    task_name: str,
    detector_seeds: list[int],
    detector_epochs: int,
    head_epochs: int,
    batch_size: int,
    detector_lr: float,
    head_lr: float,
    device: str,
    detector_image_size: int,
    pretrained: bool = False,
    freeze: bool = False,
) -> dict[str, Any]:
    """Run official-CBM training/evaluation over an existing manifest dataset."""

    _log(
        "start manifest experiment "
        f"train_dir={train_dir} eval_dir={eval_dir} output_dir={output_dir} "
        f"task={task_name} detector_seeds={detector_seeds}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_dir = output_dir / "detectors"
    head_dir = output_dir / "heads"
    detector_dir.mkdir(exist_ok=True)
    head_dir.mkdir(exist_ok=True)

    train_schema, train_rows = load_dataset(train_dir)
    eval_schema, eval_rows = load_dataset(eval_dir)
    require_concept_names(
        list(train_schema["concept_names"]),
        list(eval_schema["concept_names"]),
    )
    if task_name not in train_schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in train schema")
    if task_name not in eval_schema["tasks"]:
        raise ValueError(f"Task {task_name!r} is not present in eval schema")
    _log(
        f"loaded manifests train_examples={len(train_rows)} eval_examples={len(eval_rows)} "
        f"concepts={len(train_schema['concept_names'])} "
        f"classes={int(train_schema['tasks'][task_name]['num_classes'])}"
    )

    _log("phase train independent C->Y heads")
    head_train_results = []
    head_paths: dict[int, Path] = {}
    for head_seed in detector_seeds:
        _log(f"phase train C->Y head seed={head_seed}")
        head_path = head_dir / f"head_seed_{head_seed}.pt"
        head_paths[head_seed] = head_path
        head_train_results.append(
            train_official_synthetic_head(
                dataset_dir=train_dir,
                task_name=task_name,
                output_path=head_path,
                epochs=head_epochs,
                batch_size=batch_size,
                lr=head_lr,
                seed=head_seed,
                device=device,
            )
        )
    primary_head_seed = detector_seeds[0]
    primary_head_path = head_paths[primary_head_seed]

    detector_train_results = []
    detector_paths: dict[int, Path] = {}
    for detector_seed in detector_seeds:
        _log(f"phase train X->C detector seed={detector_seed}")
        detector_path = detector_dir / f"detector_seed_{detector_seed}.pt"
        detector_paths[detector_seed] = detector_path
        detector_train_results.append(
            train_official_synthetic_detector(
                dataset_dir=train_dir,
                output_path=detector_path,
                epochs=detector_epochs,
                batch_size=batch_size,
                lr=detector_lr,
                seed=detector_seed,
                device=device,
                image_size=detector_image_size,
                pretrained=pretrained,
                freeze=freeze,
            )
        )

    detector_evaluations = {}
    for detector_seed, detector_path in detector_paths.items():
        _log(
            f"phase evaluate detector seed={detector_seed} "
            f"with primary head seed={primary_head_seed}"
        )
        detector_evaluations[str(detector_seed)] = evaluate_official_synthetic(
            dataset_dir=eval_dir,
            detector_path=detector_path,
            head_path=primary_head_path,
            batch_size=batch_size,
            device=device,
        )
        _log(
            f"detector seed={detector_seed} "
            f"accuracy={detector_evaluations[str(detector_seed)]['accuracy']:.4f} "
            f"concept_agreement={detector_evaluations[str(detector_seed)]['concept_agreement_with_oracle']:.4f}"
        )
    _log("phase evaluate oracle C->Y baselines for all heads")
    head_evaluations = {}
    for head_seed, head_path in head_paths.items():
        head_evaluations[str(head_seed)] = evaluate_official_oracle_head(
            dataset_dir=eval_dir,
            head_path=head_path,
            batch_size=batch_size,
            device=device,
        )
    oracle_head_evaluation = head_evaluations[str(primary_head_seed)]
    _log(
        f"primary oracle C->Y head_seed={primary_head_seed} "
        f"accuracy={oracle_head_evaluation['accuracy']:.4f} "
        f"macro_f1={oracle_head_evaluation['macro_f1']:.4f}"
    )

    _log("phase build primary-head detector swap matrix")
    swap_rows = []
    for original_seed, original_path in detector_paths.items():
        original_accuracy = detector_evaluations[str(original_seed)]["accuracy"]
        for swap_seed, swap_path in detector_paths.items():
            swapped_accuracy = detector_evaluations[str(swap_seed)]["accuracy"]
            swap_rows.append(
                {
                    "original_seed": original_seed,
                    "swap_seed": swap_seed,
                    "original_detector_path": str(original_path),
                    "swap_detector_path": str(swap_path),
                    "original_accuracy": original_accuracy,
                    "swapped_accuracy": swapped_accuracy,
                    "swap_drop": original_accuracy - swapped_accuracy,
                    "relative_retention": (
                        0.0
                        if original_accuracy == 0
                        else swapped_accuracy / original_accuracy
                    ),
                    "original_concept_agreement_with_oracle": detector_evaluations[
                        str(original_seed)
                    ]["concept_agreement_with_oracle"],
                    "swap_concept_agreement_with_oracle": detector_evaluations[
                        str(swap_seed)
                    ]["concept_agreement_with_oracle"],
                }
            )

    _log("phase evaluate full head x detector matrix")
    head_detector_rows = []
    model_rows = []
    for head_seed, head_path in head_paths.items():
        for detector_seed, detector_path in detector_paths.items():
            evaluation = evaluate_official_synthetic(
                dataset_dir=eval_dir,
                detector_path=detector_path,
                head_path=head_path,
                batch_size=batch_size,
                device=device,
            )
            row = {
                "head_seed": head_seed,
                "detector_seed": detector_seed,
                "head_path": str(head_path),
                "detector_path": str(detector_path),
                "accuracy": evaluation["accuracy"],
                "macro_f1": evaluation["macro_f1"],
                "concept_agreement_with_oracle": evaluation[
                    "concept_agreement_with_oracle"
                ],
                "is_matched_seed_pair": head_seed == detector_seed,
            }
            head_detector_rows.append(row)
            if head_seed == detector_seed:
                model_rows.append(
                    {
                        "model_seed": head_seed,
                        "head_path": str(head_path),
                        "detector_path": str(detector_path),
                        "accuracy": evaluation["accuracy"],
                        "macro_f1": evaluation["macro_f1"],
                        "concept_agreement_with_oracle": evaluation[
                            "concept_agreement_with_oracle"
                        ],
                        "head_oracle_accuracy": head_evaluations[str(head_seed)][
                            "accuracy"
                        ],
                        "head_oracle_macro_f1": head_evaluations[str(head_seed)][
                            "macro_f1"
                        ],
                    }
                )

    result = {
        "output_dir": str(output_dir),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "task_name": task_name,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "num_concepts": len(train_schema["concept_names"]),
        "num_classes": int(train_schema["tasks"][task_name]["num_classes"]),
        "detector_seeds": detector_seeds,
        "detector_epochs": detector_epochs,
        "head_epochs": head_epochs,
        "batch_size": batch_size,
        "detector_lr": detector_lr,
        "head_lr": head_lr,
        "device": device,
        "detector_image_size": detector_image_size,
        "pretrained": pretrained,
        "freeze": freeze,
        "head_train": head_train_results[0],
        "head_train_all": head_train_results,
        "primary_head_seed": primary_head_seed,
        "oracle_head_evaluation": oracle_head_evaluation,
        "head_evaluations": head_evaluations,
        "detector_train": detector_train_results,
        "detector_evaluations": detector_evaluations,
        "swap_rows": swap_rows,
        "model_rows": model_rows,
        "head_detector_rows": head_detector_rows,
    }
    write_json(output_dir / "summary.json", result)
    _log(f"wrote manifest summary path={output_dir / 'summary.json'}")
    return result


def _max_num_classes(schema: dict[str, Any]) -> int:
    return max(int(task["num_classes"]) for task in schema["tasks"].values())


def _set_seed(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Pin cudnn for reproducible GPU detector training; without this,
        # convolution autotuning makes runs non-deterministic across launches.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
