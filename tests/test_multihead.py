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


class MultiheadTrainingTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_indexed_loader_uses_configured_image_workers(self) -> None:
        from unittest.mock import patch

        from symbol_sanity.multihead import _make_indexed_loader
        from symbol_sanity.synthetic import export_dataset

        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            export_dataset(dataset_dir, num_examples=4, seed=2)

            with patch.dict(
                os.environ, {"SYMBOL_SANITY_NUM_WORKERS": "8"}, clear=False
            ), patch("torch.cuda.is_available", return_value=True):
                loader = _make_indexed_loader(
                    dataset_dir=dataset_dir,
                    rows=[
                        {
                            "image_path": f"images/example_{index:05d}.png",
                            "concept_vector": [0] * 10,
                            "task_labels": {"shape_color": 0},
                        }
                        for index in range(4)
                    ],
                    batch_size=2,
                    seed=0,
                    image_size=75,
                    task_name="shape_color",
                )

            self.assertEqual(loader.num_workers, 4)
            self.assertTrue(loader.persistent_workers)
            self.assertTrue(loader.pin_memory)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_joint_single_head_checkpoints_feed_existing_evaluation(self) -> None:
        from symbol_sanity.multihead import train_multihead_cbm
        from symbol_sanity.neural_synthetic import evaluate_official_synthetic

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=12, seed=7)

            result = train_multihead_cbm(
                train_dir=dataset_dir,
                task_name="shape_color",
                output_dir=root / "joint",
                num_heads=1,
                lambda_concept=1.0,
                epochs=1,
                batch_size=4,
                detector_lr=1e-4,
                head_lr=1e-3,
                seed=0,
                device="cpu",
                image_size=75,
            )

            self.assertTrue(Path(result["detector_path"]).exists())
            self.assertEqual(len(result["head_paths"]), 1)
            evaluation = evaluate_official_synthetic(
                dataset_dir=dataset_dir,
                detector_path=Path(result["detector_path"]),
                head_path=Path(result["head_paths"][0]),
                batch_size=4,
                device="cpu",
            )
            self.assertEqual(evaluation["num_examples"], 12)
            self.assertIn("accuracy", evaluation)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_reliability_penalty_trains_with_multiple_heads(self) -> None:
        from symbol_sanity.multihead import train_multihead_cbm

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=8, seed=11)

            epistemic_table = [[0.05] * 10 for _ in range(8)]
            result = train_multihead_cbm(
                train_dir=dataset_dir,
                task_name="shape_color",
                output_dir=root / "reliability",
                num_heads=2,
                lambda_concept=0.5,
                epochs=1,
                batch_size=4,
                detector_lr=1e-4,
                head_lr=1e-3,
                seed=1,
                device="cpu",
                image_size=75,
                eta=0.5,
                beta=1.0,
                concept_dropout=0.25,
                reliability_scores=[0.5] * 10,
                epistemic_table=epistemic_table,
            )

            self.assertEqual(len(result["head_paths"]), 2)
            self.assertGreaterEqual(result["final_penalty"], 0.0)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_eta_requires_reliability_scores(self) -> None:
        from symbol_sanity.multihead import train_multihead_cbm

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=4, seed=5)

            with self.assertRaises(ValueError):
                train_multihead_cbm(
                    train_dir=dataset_dir,
                    task_name="shape_color",
                    output_dir=root / "broken",
                    num_heads=1,
                    lambda_concept=1.0,
                    epochs=0,
                    batch_size=4,
                    detector_lr=1e-4,
                    head_lr=1e-3,
                    seed=0,
                    device="cpu",
                    image_size=75,
                    eta=0.1,
                )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_comparison_experiment_writes_arm_summaries(self) -> None:
        from symbol_sanity.io import read_json
        from symbol_sanity.multihead import run_reliability_comparison_experiment

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=12, seed=3)

            comparison = run_reliability_comparison_experiment(
                train_dir=dataset_dir,
                eval_dir=dataset_dir,
                output_dir=root / "experiment",
                task_name="shape_color",
                seeds=[0, 1],
                num_heads=2,
                lambda_concept=1.0,
                eta=0.0,
                beta=0.0,
                epochs=1,
                reliability_epochs=1,
                batch_size=4,
                detector_lr=1e-4,
                head_lr=1e-3,
                device="cpu",
                detector_image_size=75,
                arms=["joint"],
            )

            self.assertIn("joint", comparison["arms"])
            summary = read_json(Path(comparison["arm_summaries"]["joint"]))
            self.assertEqual(len(summary["head_detector_rows"]), 4)
            self.assertEqual(len(summary["swap_rows"]), 4)
            self.assertIn("checkpoint_path", summary["head_train"])
            self.assertIn("0", summary["detector_evaluations"])
            self.assertIsNotNone(summary["uncertainty_report"])
            self.assertTrue(Path(summary["uncertainty_report"]).exists())

            # A second invocation over the same output directory must reuse
            # the existing checkpoints instead of retraining, so arms can run
            # as separate cluster jobs sharing one run directory.
            detector_path = Path(
                summary["detector_evaluations"]["0"]["detector_path"]
            )
            trained_at = detector_path.stat().st_mtime
            rerun = run_reliability_comparison_experiment(
                train_dir=dataset_dir,
                eval_dir=dataset_dir,
                output_dir=root / "experiment",
                task_name="shape_color",
                seeds=[0, 1],
                num_heads=2,
                lambda_concept=1.0,
                eta=0.0,
                beta=0.0,
                epochs=1,
                reliability_epochs=1,
                batch_size=4,
                detector_lr=1e-4,
                head_lr=1e-3,
                device="cpu",
                detector_image_size=75,
                arms=["joint"],
            )
            self.assertIn("joint", rerun["arms"])
            self.assertEqual(detector_path.stat().st_mtime, trained_at)


if __name__ == "__main__":
    unittest.main()
