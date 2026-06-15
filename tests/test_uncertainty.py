from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SYMBOL_SANITY_QUIET", "1")

from symbol_sanity.synthetic import export_dataset

try:
    import torch
except ModuleNotFoundError:
    torch = None


class UncertaintyDecompositionTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_symbol_decomposition_matches_mutual_information_identity(self) -> None:
        from symbol_sanity.uncertainty import symbol_uncertainty_decomposition

        member_probs = torch.tensor(
            [
                [[0.9, 0.5], [0.2, 0.5]],
                [[0.1, 0.5], [0.2, 0.5]],
            ]
        )
        decomposition = symbol_uncertainty_decomposition(member_probs)

        # Members disagree on concept 0 of example 0: epistemic must be positive.
        self.assertGreater(float(decomposition["epistemic"][0, 0]), 0.1)
        # Members agree everywhere else: epistemic must vanish.
        self.assertAlmostEqual(float(decomposition["epistemic"][0, 1]), 0.0, places=5)
        self.assertAlmostEqual(float(decomposition["epistemic"][1, 0]), 0.0, places=5)
        total = decomposition["aleatoric"] + decomposition["epistemic"]
        self.assertTrue(torch.allclose(total, decomposition["total"], atol=1e-6))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_label_decomposition_is_nonnegative(self) -> None:
        from symbol_sanity.uncertainty import label_uncertainty_decomposition

        label_probs = torch.tensor(
            [
                [[0.8, 0.1, 0.1], [0.4, 0.3, 0.3]],
                [[0.1, 0.8, 0.1], [0.4, 0.3, 0.3]],
            ]
        )
        decomposition = label_uncertainty_decomposition(label_probs)
        self.assertGreater(float(decomposition["epistemic"][0]), 0.1)
        self.assertAlmostEqual(float(decomposition["epistemic"][1]), 0.0, places=5)


class EnsembleUncertaintyReportTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_report_outputs_for_two_member_ensemble(self) -> None:
        from symbol_sanity.neural_synthetic import (
            train_official_synthetic_detector,
            train_official_synthetic_head,
        )
        from symbol_sanity.uncertainty import evaluate_ensemble_uncertainty

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=16, seed=9)

            head_path = root / "head.pt"
            train_official_synthetic_head(
                dataset_dir=dataset_dir,
                task_name="shape_color",
                output_path=head_path,
                epochs=2,
                batch_size=8,
                lr=0.01,
                seed=2,
                device="cpu",
            )

            detector_paths = []
            for seed in (0, 1):
                detector_path = root / f"detector_{seed}.pt"
                train_official_synthetic_detector(
                    dataset_dir=dataset_dir,
                    output_path=detector_path,
                    epochs=0,
                    batch_size=8,
                    lr=1e-4,
                    seed=seed,
                    device="cpu",
                    image_size=75,
                )
                detector_paths.append(detector_path)

            output_dir = root / "uncertainty"
            report = evaluate_ensemble_uncertainty(
                dataset_dir=dataset_dir,
                detector_paths=detector_paths,
                head_path=head_path,
                batch_size=8,
                device="cpu",
                output_dir=output_dir,
                num_worked_examples=3,
            )

            self.assertEqual(report["ensemble_size"], 2)
            self.assertEqual(report["num_examples"], 16)
            self.assertGreaterEqual(report["mean_symbol_epistemic"], -1e-4)
            self.assertGreaterEqual(report["mean_label_epistemic"], -1e-4)
            self.assertEqual(len(report["worked_examples"]), 3)
            self.assertIn("concept_name", report["worked_examples"][0])
            self.assertTrue((output_dir / "uncertainty_report.json").exists())
            self.assertTrue((output_dir / "per_concept_uncertainty.csv").exists())
            self.assertTrue((output_dir / "per_example_uncertainty.csv").exists())
            self.assertIn(
                "symbol_epistemic_mean_vs_label_epistemic", report["correlations"]
            )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_single_detector_is_rejected(self) -> None:
        from symbol_sanity.uncertainty import evaluate_ensemble_uncertainty

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=4, seed=1)
            with self.assertRaises(ValueError):
                evaluate_ensemble_uncertainty(
                    dataset_dir=dataset_dir,
                    detector_paths=[root / "only.pt"],
                    head_path=root / "head.pt",
                    batch_size=4,
                    device="cpu",
                    output_dir=root / "out",
                )


if __name__ == "__main__":
    unittest.main()
