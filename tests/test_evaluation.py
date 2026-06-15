from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from symbol_sanity.detectors import create_noisy_detector, create_oracle_detector
from symbol_sanity.evaluation import evaluate_detector_head, evaluate_swap
from symbol_sanity.heads import train_lookup_head
from symbol_sanity.io import read_json, write_json
from symbol_sanity.synthetic import export_dataset


class EvaluationTests(unittest.TestCase):
    def test_oracle_detector_and_lookup_head_are_perfect_on_synthetic_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=200, seed=1)

            schema = read_json(dataset_dir / "schema.json")
            detector_path = root / "oracle_detector.json"
            head_path = root / "shape_color_head.json"

            create_oracle_detector(
                detector_path,
                name="oracle",
                concept_names=list(schema["concept_names"]),
            )
            _, rows = self._load_dataset(dataset_dir)
            train_lookup_head(
                head_path,
                name="shape_color_lookup",
                task_name="shape_color",
                concept_names=list(schema["concept_names"]),
                num_classes=5,
                rows=rows,
            )

            result = evaluate_detector_head(dataset_dir, detector_path, head_path)

            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(result["macro_f1"], 1.0)
            self.assertEqual(result["concept_agreement_with_oracle"], 1.0)

    def test_noisy_detector_produces_swap_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=200, seed=2)

            schema = read_json(dataset_dir / "schema.json")
            _, rows = self._load_dataset(dataset_dir)
            oracle_path = root / "oracle.json"
            noisy_path = root / "noisy.json"
            head_path = root / "head.json"

            create_oracle_detector(
                oracle_path,
                name="oracle",
                concept_names=list(schema["concept_names"]),
            )
            create_noisy_detector(
                noisy_path,
                name="noisy",
                concept_names=list(schema["concept_names"]),
                flip_probability=0.5,
                seed=9,
            )
            train_lookup_head(
                head_path,
                name="shape_position_lookup",
                task_name="shape_position",
                concept_names=list(schema["concept_names"]),
                num_classes=5,
                rows=rows,
            )

            result = evaluate_swap(dataset_dir, oracle_path, noisy_path, head_path)

            self.assertEqual(result["original_accuracy"], 1.0)
            self.assertLess(result["swapped_accuracy"], 1.0)
            self.assertGreater(result["swap_drop"], 0.0)
            self.assertLess(result["swap_concept_agreement_with_oracle"], 1.0)

    def test_concept_order_mismatch_fails_hard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=20, seed=3)

            schema = read_json(dataset_dir / "schema.json")
            _, rows = self._load_dataset(dataset_dir)
            detector_path = root / "detector.json"
            head_path = root / "head.json"

            reversed_names = list(reversed(schema["concept_names"]))
            create_oracle_detector(
                detector_path,
                name="bad_detector",
                concept_names=reversed_names,
            )
            train_lookup_head(
                head_path,
                name="head",
                task_name="mixed",
                concept_names=list(schema["concept_names"]),
                num_classes=5,
                rows=rows,
            )

            with self.assertRaisesRegex(ValueError, "Concept schema mismatch"):
                evaluate_detector_head(dataset_dir, detector_path, head_path)

    def test_evaluation_can_write_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            export_dataset(dataset_dir, num_examples=50, seed=4)

            schema = read_json(dataset_dir / "schema.json")
            _, rows = self._load_dataset(dataset_dir)
            detector_path = root / "oracle.json"
            head_path = root / "head.json"
            output_path = root / "result.json"

            create_oracle_detector(
                detector_path,
                name="oracle",
                concept_names=list(schema["concept_names"]),
            )
            train_lookup_head(
                head_path,
                name="color_size_head",
                task_name="color_size",
                concept_names=list(schema["concept_names"]),
                num_classes=5,
                rows=rows,
            )

            evaluate_detector_head(dataset_dir, detector_path, head_path, output_path)
            result = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(result["task_name"], "color_size")

    def _load_dataset(self, dataset_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
        schema = json.loads((dataset_dir / "schema.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (dataset_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        return schema, rows


if __name__ == "__main__":
    unittest.main()

