from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

os.makedirs("/tmp/symbol_sanity_matplotlib_tests", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/symbol_sanity_matplotlib_tests")
os.environ.setdefault("SYMBOL_SANITY_QUIET", "1")

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    matplotlib = None

from symbol_sanity.plot_report import generate_plot_report


class PlotReportTests(unittest.TestCase):
    @unittest.skipIf(matplotlib is None, "matplotlib is not installed")
    def test_generate_plot_report_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "summary.json"
            stat_dir = root / "statistical_report"
            output_dir = root / "figures"
            stat_dir.mkdir()

            summary_path.write_text(
                json.dumps(
                    {
                        "detector_evaluations": {
                            "0": {
                                "accuracy": 0.8,
                                "macro_f1": 0.75,
                                "concept_agreement_with_oracle": 0.7,
                            },
                            "1": {
                                "accuracy": 0.6,
                                "macro_f1": 0.55,
                                "concept_agreement_with_oracle": 0.65,
                            },
                        },
                        "oracle_head_evaluation": {"accuracy": 0.95},
                        "swap_rows": [
                            {
                                "original_seed": 0,
                                "swap_seed": 0,
                                "swapped_accuracy": 0.8,
                                "swap_drop": 0.0,
                            },
                            {
                                "original_seed": 0,
                                "swap_seed": 1,
                                "swapped_accuracy": 0.6,
                                "swap_drop": 0.2,
                            },
                            {
                                "original_seed": 1,
                                "swap_seed": 0,
                                "swapped_accuracy": 0.8,
                                "swap_drop": -0.2,
                            },
                            {
                                "original_seed": 1,
                                "swap_seed": 1,
                                "swapped_accuracy": 0.6,
                                "swap_drop": 0.0,
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (stat_dir / "statistical_report.json").write_text(
                json.dumps({"head_compatibility": {"observed_accuracy": 0.95}})
                + "\n",
                encoding="utf-8",
            )
            with (stat_dir / "concept_reliability.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "concept_index",
                        "concept_name",
                        "prevalence",
                        "accuracy_mean",
                        "balanced_accuracy_mean",
                        "f1_mean",
                        "auroc_mean",
                        "auprc_mean",
                        "permutation_p_value_min",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "concept_index": 0,
                        "concept_name": "concept_a",
                        "prevalence": 0.25,
                        "accuracy_mean": 0.8,
                        "balanced_accuracy_mean": 0.7,
                        "f1_mean": 0.6,
                        "auroc_mean": 0.75,
                        "auprc_mean": 0.55,
                        "permutation_p_value_min": 0.02,
                    }
                )
                writer.writerow(
                    {
                        "concept_index": 1,
                        "concept_name": "concept_b",
                        "prevalence": 0.0,
                        "accuracy_mean": 1.0,
                        "balanced_accuracy_mean": "",
                        "f1_mean": 0.0,
                        "auroc_mean": "",
                        "auprc_mean": "",
                        "permutation_p_value_min": 1.0,
                    }
                )
            with (stat_dir / "head_compatibility_nulls.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["null_type", "permutation_index", "accuracy", "macro_f1"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "null_type": "concept_dimension_permutation",
                        "permutation_index": 0,
                        "accuracy": 0.3,
                        "macro_f1": 0.2,
                    }
                )
                writer.writerow(
                    {
                        "null_type": "within_concept_sample_shuffle",
                        "permutation_index": 0,
                        "accuracy": 0.25,
                        "macro_f1": 0.15,
                    }
                )

            result = generate_plot_report(
                summary_path=summary_path,
                statistical_report_dir=stat_dir,
                output_dir=output_dir,
                formats=["png"],
            )

            self.assertTrue((output_dir / "plot_report.json").exists())
            self.assertTrue((output_dir / "swap_accuracy_heatmap.png").exists())
            self.assertTrue((output_dir / "head_compatibility_nulls.png").exists())
            self.assertTrue((output_dir / "concept_reliability_top_bottom.csv").exists())
            self.assertIn("swap_drop_heatmap", result["outputs"])


if __name__ == "__main__":
    unittest.main()
