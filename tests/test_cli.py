from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from symbol_sanity.cli import main


class CliSmokeTests(unittest.TestCase):
    def test_official_cbm_status_prints_structured_json(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(["official-cbm-status"])
        self.assertEqual(exit_code, 0)
        status = json.loads(buffer.getvalue())
        self.assertIn("available", status)
        self.assertIsInstance(status["available"], bool)

    def test_generate_detector_head_evaluate_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_dir = tmp_path / "data"
            oracle_path = tmp_path / "oracle.json"
            head_path = tmp_path / "head.json"

            self.assertEqual(
                main(
                    [
                        "generate-synthetic",
                        "--output-dir",
                        str(dataset_dir),
                        "--num-examples",
                        "16",
                        "--seed",
                        "0",
                        "--image-size",
                        "32",
                    ]
                ),
                0,
            )
            self.assertTrue((dataset_dir / "schema.json").exists())
            self.assertTrue((dataset_dir / "metadata.jsonl").exists())
            self.assertEqual(len(list((dataset_dir / "images").glob("*.png"))), 16)

            self.assertEqual(
                main(
                    [
                        "create-oracle-detector",
                        "--dataset-dir",
                        str(dataset_dir),
                        "--output-path",
                        str(oracle_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "train-lookup-head",
                        "--dataset-dir",
                        str(dataset_dir),
                        "--task",
                        "shape_color",
                        "--output-path",
                        str(head_path),
                    ]
                ),
                0,
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "evaluate",
                        "--dataset-dir",
                        str(dataset_dir),
                        "--detector",
                        str(oracle_path),
                        "--head",
                        str(head_path),
                    ]
                )
            self.assertEqual(exit_code, 0)
            result = json.loads(buffer.getvalue())
            # An oracle detector feeding a lookup head fit on the same data is
            # perfect, which confirms the CLI wiring end to end.
            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(result["task_name"], "shape_color")

    def test_unknown_task_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "data"
            main(
                [
                    "generate-synthetic",
                    "--output-dir",
                    str(dataset_dir),
                    "--num-examples",
                    "4",
                    "--image-size",
                    "32",
                ]
            )
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(
                    [
                        "train-lookup-head",
                        "--dataset-dir",
                        str(dataset_dir),
                        "--task",
                        "not_a_real_task",
                        "--output-path",
                        str(Path(tmp) / "head.json"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
