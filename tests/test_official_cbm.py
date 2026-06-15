from __future__ import annotations

import unittest

from symbol_sanity.official_cbm import (
    OfficialCBMSpec,
    OfficialCBMUnavailableError,
    _parse_version,
    _require_min_version,
    backend_status,
    build_official_cbm,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


class OfficialCBMAdapterTests(unittest.TestCase):
    def test_backend_status_is_structured(self) -> None:
        status = backend_status()
        self.assertIn("available", status)
        self.assertIn("reason", status)
        self.assertEqual(status["implementation"], "torchvision")

    def test_version_parser_handles_modern_torch_suffixes(self) -> None:
        self.assertEqual(_parse_version("2.2.1+cu121"), (2, 2))
        self.assertEqual(_parse_version("0.17.0"), (0, 17))
        self.assertEqual(_parse_version("1.24.4"), (1, 24))

    def test_old_torch_versions_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            OfficialCBMUnavailableError,
            "Old torch-era environments are intentionally unsupported",
        ):
            _require_min_version("torch", "1.13.1", (2, 2))

    def test_build_official_cbm_reports_missing_dependencies_cleanly(self) -> None:
        status = backend_status()
        if status["available"]:
            self.skipTest("Neural dependencies are installed in this environment")

        with self.assertRaises(RuntimeError):
            build_official_cbm(
                OfficialCBMSpec(
                    mode="Independent_CtoY",
                    n_attributes=10,
                    num_classes=5,
                )
            )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_modern_torch_can_build_model_modes(self) -> None:
        for mode in [
            "Independent_CtoY",
            "Sequential_CtoY",
            "Concept_XtoC",
            "Joint",
            "Standard",
            "Multitask",
        ]:
            with self.subTest(mode=mode):
                model = build_official_cbm(
                    OfficialCBMSpec(
                        mode=mode,
                        n_attributes=3,
                        num_classes=5,
                        pretrained=False,
                        use_aux=False,
                    )
                )
                self.assertIsNotNone(model)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_modern_torch_forward_shapes(self) -> None:
        head = build_official_cbm(
            OfficialCBMSpec(
                mode="Independent_CtoY",
                n_attributes=10,
                num_classes=5,
                expand_dim=16,
            )
        )
        self.assertEqual(tuple(head(torch.zeros(2, 10)).shape), (2, 5))

        concept_detector = build_official_cbm(
            OfficialCBMSpec(
                mode="Concept_XtoC",
                n_attributes=3,
                num_classes=5,
                pretrained=False,
                use_aux=False,
            )
        )
        concept_detector.eval()
        with torch.no_grad():
            concept_outputs = concept_detector(torch.zeros(1, 3, 299, 299))
        self.assertEqual([tuple(output.shape) for output in concept_outputs], [(1, 1)] * 3)

        joint = build_official_cbm(
            OfficialCBMSpec(
                mode="Joint",
                n_attributes=3,
                num_classes=5,
                pretrained=False,
                use_aux=False,
                use_sigmoid=True,
            )
        )
        joint.eval()
        with torch.no_grad():
            joint_outputs = joint(torch.zeros(1, 3, 299, 299))
        self.assertEqual(
            [tuple(output.shape) for output in joint_outputs],
            [(1, 5), (1, 1), (1, 1), (1, 1)],
        )


if __name__ == "__main__":
    unittest.main()
