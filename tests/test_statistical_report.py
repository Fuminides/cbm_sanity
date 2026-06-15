from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SYMBOL_SANITY_QUIET", "1")

from symbol_sanity.statistical_report import (
    auprc,
    auroc,
    balanced_accuracy,
    binary_f1,
    empirical_right_tail_p_value,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


class StatisticalReportMetricTests(unittest.TestCase):
    def test_binary_metrics_handle_separable_scores(self) -> None:
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]

        self.assertEqual(balanced_accuracy(y_true, y_pred), 1.0)
        self.assertEqual(binary_f1(y_true, y_pred), 1.0)
        self.assertEqual(auroc(y_true, y_score), 1.0)
        self.assertEqual(auprc(y_true, y_score), 1.0)

    def test_empirical_right_tail_p_value_uses_plus_one_correction(self) -> None:
        self.assertEqual(
            empirical_right_tail_p_value(0.9, [0.1, 0.2, 0.3]),
            0.25,
        )

    def test_balanced_accuracy_is_undefined_for_single_class_targets(self) -> None:
        self.assertIsNone(balanced_accuracy([0, 0, 0], [0, 0, 0]))


class HeadCompatibilityNullTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_nulls_run_on_predicted_concepts_when_detector_given(self) -> None:
        from symbol_sanity.neural_synthetic import (
            train_official_synthetic_detector,
            train_official_synthetic_head,
        )
        from symbol_sanity.statistical_report import evaluate_head_compatibility_nulls
        from symbol_sanity.synthetic import export_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=12, seed=4)
            head_path = root / "head.pt"
            detector_path = root / "detector.pt"
            train_official_synthetic_head(
                dataset_dir=dataset_dir,
                task_name="shape_color",
                output_path=head_path,
                epochs=2,
                batch_size=8,
                lr=0.01,
                seed=0,
                device="cpu",
            )
            train_official_synthetic_detector(
                dataset_dir=dataset_dir,
                output_path=detector_path,
                epochs=0,
                batch_size=8,
                lr=1e-4,
                seed=0,
                device="cpu",
                image_size=75,
            )

            report = evaluate_head_compatibility_nulls(
                dataset_dir=dataset_dir,
                head_path=head_path,
                batch_size=8,
                device="cpu",
                num_permutations=2,
                seed=0,
                detector_path=detector_path,
            )

            self.assertEqual(report["concept_source"], "predicted")
            self.assertEqual(report["detector_path"], str(detector_path))
            self.assertIn("dimension_permutation_p_value", report)
            self.assertIn("sample_shuffle_p_value", report)
            self.assertEqual(len(report["null_rows"]), 4)


if __name__ == "__main__":
    unittest.main()
