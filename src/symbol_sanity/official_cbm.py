"""Torchvision-backed models matching the CBM experiment interfaces.

The original Koh et al. implementation copied and modified torchvision's
Inception v3 source. This module instead depends on torchvision directly and
adds only the concept-head and composition layers needed by this project.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Any


MIN_TORCH_VERSION = (2, 2)
MIN_TORCHVISION_VERSION = (0, 17)
MIN_NUMPY_VERSION = (1, 24)


class OfficialCBMUnavailableError(RuntimeError):
    """Raised when the optional neural-model dependencies are unavailable."""


@dataclass(frozen=True)
class OfficialCBMSpec:
    mode: str
    n_attributes: int
    num_classes: int
    n_class_attr: int = 2
    pretrained: bool = False
    freeze: bool = False
    use_aux: bool = False
    expand_dim: int = 0
    use_relu: bool = False
    use_sigmoid: bool = False


def backend_status() -> dict[str, Any]:
    """Return import status for the torchvision-backed CBM models."""

    try:
        versions = _check_modern_dependencies()
        _backend_types()
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "implementation": "torchvision",
        }

    return {
        "available": True,
        "reason": "ok",
        "implementation": "torchvision",
        "versions": versions,
    }


def build_official_cbm(spec: OfficialCBMSpec) -> Any:
    """Construct a CBM variant with the interfaces used by Koh et al.

    The Inception v3 backbone is supplied by torchvision rather than copied
    into this repository. The project-owned wrappers preserve the list-valued
    concept outputs expected by the training and evaluation code.
    """

    backend = _backend_types()

    if spec.mode in {"Independent_CtoY", "Sequential_CtoY"}:
        input_dim = spec.n_attributes * spec.n_class_attr if spec.n_class_attr == 3 else spec.n_attributes
        return backend.MLP(input_dim, spec.num_classes, spec.expand_dim)

    if spec.mode == "Concept_XtoC":
        return _build_inception(
            backend,
            spec,
            include_class_output=False,
        )

    if spec.mode == "Joint":
        detector = _build_inception(
            backend,
            spec,
            include_class_output=False,
        )
        input_dim = spec.n_attributes * spec.n_class_attr if spec.n_class_attr == 3 else spec.n_attributes
        head = backend.MLP(input_dim, spec.num_classes, spec.expand_dim)
        return backend.EndToEndModel(
            detector,
            head,
            use_relu=spec.use_relu,
            use_sigmoid=spec.use_sigmoid,
        )

    if spec.mode == "Standard":
        return _build_inception(
            backend,
            spec,
            include_class_output=True,
            n_attributes=0,
        )

    if spec.mode == "Multitask":
        return _build_inception(
            backend,
            spec,
            include_class_output=True,
        )

    raise ValueError(f"Unsupported official CBM mode: {spec.mode!r}")


def _build_inception(
    backend: SimpleNamespace,
    spec: OfficialCBMSpec,
    *,
    include_class_output: bool,
    n_attributes: int | None = None,
) -> Any:
    torchvision = importlib.import_module("torchvision")
    models = torchvision.models
    weights = models.Inception_V3_Weights.IMAGENET1K_V1 if spec.pretrained else None

    # torchvision requires auxiliary logits while loading pretrained weights.
    load_with_aux = spec.use_aux or spec.pretrained
    backbone = models.inception_v3(
        weights=weights,
        aux_logits=load_with_aux,
        transform_input=spec.pretrained,
        init_weights=not spec.pretrained,
    )
    if spec.pretrained and not spec.use_aux:
        backbone.aux_logits = False
        backbone.AuxLogits = None

    attribute_count = spec.n_attributes if n_attributes is None else n_attributes
    output_dims = []
    if include_class_output:
        output_dims.append(spec.num_classes)
    output_dims.extend([1] * attribute_count)

    backbone.fc = backend.OutputHeads(2048, output_dims, spec.expand_dim)
    if backbone.AuxLogits is not None:
        backbone.AuxLogits.fc = backend.OutputHeads(768, output_dims, spec.expand_dim)

    model = backend.InceptionOutputsAdapter(backbone)
    if spec.freeze:
        for name, parameter in model.named_parameters():
            if ".fc." not in name:
                parameter.requires_grad = False
    return model


@lru_cache(maxsize=1)
def _backend_types() -> SimpleNamespace:
    _check_modern_dependencies()
    torch = importlib.import_module("torch")
    nn = torch.nn

    class MLP(nn.Module):
        def __init__(self, input_dim: int, num_classes: int, expand_dim: int) -> None:
            super().__init__()
            if expand_dim > 0:
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, expand_dim),
                    nn.ReLU(),
                    nn.Linear(expand_dim, num_classes),
                )
            else:
                self.layers = nn.Linear(input_dim, num_classes)

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    class OutputHeads(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dims: list[int],
            expand_dim: int,
        ) -> None:
            super().__init__()
            self.heads = nn.ModuleList(
                MLP(input_dim, output_dim, expand_dim)
                for output_dim in output_dims
            )

        def forward(self, inputs: Any) -> list[Any]:
            return [head(inputs) for head in self.heads]

    class InceptionOutputsAdapter(nn.Module):
        def __init__(self, backbone: Any) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, inputs: Any) -> Any:
            outputs = self.backbone(inputs)
            if hasattr(outputs, "logits"):
                return outputs.logits, outputs.aux_logits
            return outputs

    class EndToEndModel(nn.Module):
        def __init__(
            self,
            first_model: Any,
            second_model: Any,
            *,
            use_relu: bool,
            use_sigmoid: bool,
        ) -> None:
            super().__init__()
            self.first_model = first_model
            self.sec_model = second_model
            self.use_relu = use_relu
            self.use_sigmoid = use_sigmoid

        def _forward_stage2(self, concept_outputs: list[Any]) -> list[Any]:
            if self.use_relu:
                head_inputs = [torch.relu(output) for output in concept_outputs]
            elif self.use_sigmoid:
                head_inputs = [torch.sigmoid(output) for output in concept_outputs]
            else:
                head_inputs = concept_outputs
            label_output = self.sec_model(torch.cat(head_inputs, dim=1))
            return [label_output, *concept_outputs]

        def forward(self, inputs: Any) -> Any:
            outputs = self.first_model(inputs)
            if isinstance(outputs, tuple):
                main_outputs, auxiliary_outputs = outputs
                return (
                    self._forward_stage2(main_outputs),
                    self._forward_stage2(auxiliary_outputs),
                )
            return self._forward_stage2(outputs)

    return SimpleNamespace(
        EndToEndModel=EndToEndModel,
        InceptionOutputsAdapter=InceptionOutputsAdapter,
        MLP=MLP,
        OutputHeads=OutputHeads,
    )


def _check_modern_dependencies() -> dict[str, str]:
    versions = {
        "torch": _import_version("torch"),
        "torchvision": _import_version("torchvision"),
        "numpy": _import_version("numpy"),
    }
    _require_min_version("torch", versions["torch"], MIN_TORCH_VERSION)
    _require_min_version("torchvision", versions["torchvision"], MIN_TORCHVISION_VERSION)
    _require_min_version("numpy", versions["numpy"], MIN_NUMPY_VERSION)
    return versions


def _import_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise OfficialCBMUnavailableError(
            "Neural CBM models require optional dependencies. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc
    return str(getattr(module, "__version__", "0"))


def _require_min_version(
    package_name: str,
    version: str,
    minimum: tuple[int, int],
) -> None:
    parsed = _parse_version(version)
    if parsed < minimum:
        minimum_text = ".".join(str(part) for part in minimum)
        raise OfficialCBMUnavailableError(
            f"{package_name}>={minimum_text} is required, found {version}. "
            "Old torch-era environments are intentionally unsupported."
        )


def _parse_version(version: str) -> tuple[int, int]:
    main = version.split("+", 1)[0]
    parts = main.split(".")
    parsed: list[int] = []
    for part in parts[:2]:
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        parsed.append(int(digits or 0))
    while len(parsed) < 2:
        parsed.append(0)
    return tuple(parsed)
