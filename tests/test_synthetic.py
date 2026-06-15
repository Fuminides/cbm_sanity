from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from symbol_sanity.schemas import DEFAULT_CONCEPT_SCHEMA, TASK_SCHEMAS
from symbol_sanity.synthetic import export_dataset, make_example, render_example


class SyntheticDataTests(unittest.TestCase):
    def test_concept_order_is_explicit(self) -> None:
        self.assertEqual(
            DEFAULT_CONCEPT_SCHEMA.names,
            (
                "shape_circle",
                "shape_square",
                "shape_triangle",
                "color_red",
                "color_blue",
                "color_green",
                "size_small",
                "size_large",
                "position_left",
                "position_right",
            ),
        )

    def test_examples_are_deterministic(self) -> None:
        first = make_example(index=7, seed=3)
        second = make_example(index=7, seed=3)
        different = make_example(index=8, seed=3)

        self.assertEqual(first, second)
        self.assertNotEqual(first.attributes, different.attributes)

    def test_concept_vector_is_valid_binary_schema(self) -> None:
        example = make_example(index=0, seed=0)
        DEFAULT_CONCEPT_SCHEMA.validate_vector(example.concept_vector)
        self.assertEqual(sum(example.concept_vector[:3]), 1)
        self.assertEqual(sum(example.concept_vector[3:6]), 1)
        self.assertEqual(sum(example.concept_vector[6:8]), 1)
        self.assertEqual(sum(example.concept_vector[8:10]), 1)

    def test_task_labels_are_in_range(self) -> None:
        for index in range(100):
            example = make_example(index=index, seed=11)
            for task_name, label in example.task_labels.items():
                self.assertIn(task_name, TASK_SCHEMAS)
                self.assertGreaterEqual(label, 0)
                self.assertLess(label, TASK_SCHEMAS[task_name].num_classes)

    def test_render_returns_rgb_image(self) -> None:
        image = render_example(make_example(index=2, seed=5), image_size=32)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (32, 32))

    def test_export_dataset_writes_images_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            export_dataset(
                output_dir=output_dir,
                num_examples=3,
                seed=17,
                image_size=32,
                tasks=["shape_color", "mixed"],
            )

            schema = json.loads((output_dir / "schema.json").read_text())
            self.assertEqual(schema["concept_names"], list(DEFAULT_CONCEPT_SCHEMA.names))
            self.assertEqual(sorted(schema["tasks"]), ["mixed", "shape_color"])

            rows = [
                json.loads(line)
                for line in (output_dir / "metadata.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertTrue((output_dir / row["image_path"]).exists())
                self.assertEqual(row["concept_names"], list(DEFAULT_CONCEPT_SCHEMA.names))
                self.assertEqual(sorted(row["task_labels"]), ["mixed", "shape_color"])


if __name__ == "__main__":
    unittest.main()

