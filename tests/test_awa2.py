from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from symbol_sanity.awa2 import build_awa2_manifest


class Awa2ManifestTests(unittest.TestCase):
    def test_build_awa2_manifest_from_raw_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            awa2_root = root / "Animals_with_Attributes2"
            self._write_fake_awa2(awa2_root)
            output_dir = root / "awa2_manifest"

            result = build_awa2_manifest(
                awa2_root=awa2_root,
                output_dir=output_dir,
                class_ids=[1, 2],
                val_fraction=0.25,
                test_fraction=0.25,
                seed=0,
            )

            self.assertEqual(result["num_attributes"], 4)
            self.assertEqual(result["num_classes"], 2)
            self.assertEqual(result["train_rows"], 4)
            self.assertEqual(result["val_rows"], 2)
            self.assertEqual(result["test_rows"], 2)

            schema = json.loads((output_dir / "train" / "schema.json").read_text())
            self.assertEqual(schema["dataset"], "AwA2")
            self.assertEqual(schema["tasks"]["species"]["num_classes"], 2)
            self.assertEqual(schema["concept_names"], ["black", "white", "fast", "slow"])
            self.assertEqual(
                schema["concept_encoding"],
                "binary_class_level_attributes",
            )

            train_rows = [
                json.loads(line)
                for line in (output_dir / "train" / "metadata.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(train_rows)
            self.assertTrue(Path(train_rows[0]["image_path"]).is_absolute())
            self.assertEqual(sorted(train_rows[0]["task_labels"]), ["species"])

            vectors_by_class = {}
            for row in train_rows:
                vectors_by_class.setdefault(row["class_id"], set()).add(
                    tuple(row["concept_vector"])
                )
            for vectors in vectors_by_class.values():
                self.assertEqual(len(vectors), 1)

    def test_continuous_threshold_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            awa2_root = root / "Animals_with_Attributes2"
            self._write_fake_awa2(awa2_root)
            output_dir = root / "awa2_manifest"

            build_awa2_manifest(
                awa2_root=awa2_root,
                output_dir=output_dir,
                num_classes=1,
                val_fraction=0.0,
                test_fraction=0.25,
                seed=0,
                attribute_kind="continuous-threshold",
                continuous_threshold=50.0,
            )

            schema = json.loads((output_dir / "train" / "schema.json").read_text())
            self.assertEqual(
                schema["concept_encoding"],
                "continuous-threshold_class_level_attributes",
            )
            row = json.loads(
                (output_dir / "train" / "metadata.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(row["concept_vector"], [1, 0, 1, 0])

    def test_builder_accepts_missing_classes_txt_by_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            awa2_root = root / "Animals_with_Attributes2"
            self._write_fake_awa2(awa2_root)
            (awa2_root / "classes.txt").unlink()
            output_dir = root / "awa2_manifest"

            result = build_awa2_manifest(
                awa2_root=awa2_root,
                output_dir=output_dir,
                num_classes=2,
                val_fraction=0.0,
                test_fraction=0.25,
                seed=0,
            )

            self.assertEqual(result["class_ids"], [1, 2])
            schema = json.loads((output_dir / "train" / "schema.json").read_text())
            self.assertEqual(
                schema["classes"],
                {"1": "antelope", "2": "grizzly+bear"},
            )

    def test_builder_accepts_missing_predicates_txt_by_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            awa2_root = root / "Animals_with_Attributes2"
            self._write_fake_awa2(awa2_root)
            (awa2_root / "predicates.txt").unlink()
            output_dir = root / "awa2_manifest"

            result = build_awa2_manifest(
                awa2_root=awa2_root,
                output_dir=output_dir,
                num_classes=1,
                val_fraction=0.0,
                test_fraction=0.25,
                seed=0,
            )

            self.assertEqual(result["num_attributes"], 4)
            schema = json.loads((output_dir / "train" / "schema.json").read_text())
            self.assertEqual(
                schema["concept_names"],
                ["predicate_1", "predicate_2", "predicate_3", "predicate_4"],
            )

    def test_builder_accepts_missing_binary_matrix_by_thresholding_continuous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            awa2_root = root / "Animals_with_Attributes2"
            self._write_fake_awa2(awa2_root)
            (awa2_root / "predicate-matrix-binary.txt").unlink()
            output_dir = root / "awa2_manifest"

            build_awa2_manifest(
                awa2_root=awa2_root,
                output_dir=output_dir,
                num_classes=1,
                val_fraction=0.0,
                test_fraction=0.25,
                seed=0,
                attribute_kind="binary",
                continuous_threshold=50.0,
            )

            row = json.loads(
                (output_dir / "train" / "metadata.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(row["concept_vector"], [1, 0, 1, 0])

    def test_builder_accepts_missing_continuous_matrix_by_using_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            awa2_root = root / "Animals_with_Attributes2"
            self._write_fake_awa2(awa2_root)
            (awa2_root / "predicate-matrix-continuous.txt").unlink()
            output_dir = root / "awa2_manifest"

            build_awa2_manifest(
                awa2_root=awa2_root,
                output_dir=output_dir,
                num_classes=1,
                val_fraction=0.0,
                test_fraction=0.25,
                seed=0,
                attribute_kind="continuous-threshold",
                continuous_threshold=50.0,
            )

            row = json.loads(
                (output_dir / "train" / "metadata.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(row["concept_vector"], [1, 0, 1, 0])

    def test_builder_accepts_nested_jpegimages_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            awa2_root = root / "Animals_with_Attributes2"
            self._write_fake_awa2(awa2_root)
            nested_root = awa2_root / "Animals_with_Attributes2"
            (nested_root).mkdir()
            (awa2_root / "JPEGImages").rename(nested_root / "JPEGImages")
            output_dir = root / "awa2_manifest"

            result = build_awa2_manifest(
                awa2_root=awa2_root,
                output_dir=output_dir,
                num_classes=1,
                val_fraction=0.0,
                test_fraction=0.25,
                seed=0,
            )

            self.assertEqual(result["train_rows"], 3)
            row = json.loads(
                (output_dir / "train" / "metadata.jsonl").read_text().splitlines()[0]
            )
            self.assertIn("Animals_with_Attributes2/JPEGImages", row["image_path"])

    def _write_fake_awa2(self, awa2_root: Path) -> None:
        (awa2_root / "JPEGImages").mkdir(parents=True)
        classes = ["antelope", "grizzly+bear", "killer+whale"]
        for class_name in classes:
            image_dir = awa2_root / "JPEGImages" / class_name
            image_dir.mkdir(parents=True)
            for index in range(4):
                image_path = image_dir / f"{class_name}_{index}.jpg"
                Image.new("RGB", (8, 8), (index, index, index)).save(image_path)

        (awa2_root / "classes.txt").write_text(
            "\n".join(
                f"{class_id} {class_name}"
                for class_id, class_name in enumerate(classes, start=1)
            )
            + "\n",
            encoding="utf-8",
        )
        (awa2_root / "predicates.txt").write_text(
            "\n".join(
                f"{predicate_id} {name}"
                for predicate_id, name in enumerate(
                    ["black", "white", "fast", "slow"],
                    start=1,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        (awa2_root / "predicate-matrix-binary.txt").write_text(
            "1 0 1 0\n"
            "0 1 1 0\n"
            "1 1 0 1\n",
            encoding="utf-8",
        )
        (awa2_root / "predicate-matrix-continuous.txt").write_text(
            "90 10 70 20\n"
            "10 90 80 30\n"
            "60 60 20 90\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
