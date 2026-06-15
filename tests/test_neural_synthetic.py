from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

os.environ.setdefault("SYMBOL_SANITY_QUIET", "1")

from symbol_sanity.neural_synthetic import (
    evaluate_official_synthetic,
    run_shared_extractor_multihead_manifest_experiment,
    train_official_synthetic_head,
)
from symbol_sanity.synthetic import export_dataset

try:
    import torch
except ModuleNotFoundError:
    torch = None


class NeuralSyntheticTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_official_head_trains_on_synthetic_oracle_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=100, seed=13)
            head_path = root / "head.pt"

            result = train_official_synthetic_head(
                dataset_dir=dataset_dir,
                task_name="shape_color",
                output_path=head_path,
                epochs=20,
                batch_size=16,
                lr=0.05,
                seed=3,
                device="cpu",
            )

            self.assertEqual(result["component"], "official_synthetic_head")
            self.assertTrue(head_path.exists())

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_official_evaluation_accepts_saved_torch_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=32, seed=14)
            head_path = root / "head.pt"
            detector_path = root / "detector.pt"

            train_official_synthetic_head(
                dataset_dir=dataset_dir,
                task_name="shape_color",
                output_path=head_path,
                epochs=2,
                batch_size=16,
                lr=0.01,
                seed=3,
                device="cpu",
            )

            # Keep this test light: use a freshly initialized detector checkpoint
            # with the same format expected by evaluation. Training the full
            # Inception detector is covered by CLI smoke runs.
            from symbol_sanity.neural_synthetic import train_official_synthetic_detector

            train_official_synthetic_detector(
                dataset_dir=dataset_dir,
                output_path=detector_path,
                epochs=0,
                batch_size=8,
                lr=0.001,
                seed=3,
                device="cpu",
                image_size=299,
            )

            result = evaluate_official_synthetic(
                dataset_dir=dataset_dir,
                detector_path=detector_path,
                head_path=head_path,
                batch_size=8,
                device="cpu",
            )

            self.assertEqual(result["num_examples"], 32)
            self.assertIn("accuracy", result)
            self.assertIn("concept_agreement_with_oracle", result)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_shared_extractor_multihead_summary_is_report_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=12, seed=21, image_size=75)
            output_dir = root / "shared"

            result = run_shared_extractor_multihead_manifest_experiment(
                train_dir=dataset_dir,
                eval_dir=dataset_dir,
                output_dir=output_dir,
                task_name="shape_color",
                head_seeds=[0, 1],
                epochs=0,
                batch_size=4,
                lr=0.001,
                concept_loss_weight=1.0,
                task_loss_weight=1.0,
                device="cpu",
                detector_image_size=75,
                pretrained=False,
                freeze=False,
                seed=0,
            )

            self.assertEqual(result["training_regime"], "shared_extractor_multihead")
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertEqual(len(result["head_detector_rows"]), 2)
            self.assertEqual(len(result["swap_rows"]), 1)
            self.assertEqual(sorted(result["detector_evaluations"]), ["0"])


if __name__ == "__main__":
    unittest.main()
