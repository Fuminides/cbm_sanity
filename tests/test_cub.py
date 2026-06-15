from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from symbol_sanity.cub import build_cub_manifest


class CubManifestTests(unittest.TestCase):
    def test_build_cub_manifest_from_raw_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cub_root = root / "CUB_200_2011"
            self._write_fake_cub(cub_root)
            output_dir = root / "cub_manifest"

            result = build_cub_manifest(
                cub_root=cub_root,
                output_dir=output_dir,
                num_attributes=3,
                class_ids=[1, 2],
                val_fraction=0.5,
                seed=0,
            )

            self.assertEqual(result["num_attributes"], 3)
            self.assertEqual(result["num_classes"], 2)
            self.assertEqual(result["test_rows"], 4)

            schema = json.loads((output_dir / "train" / "schema.json").read_text())
            self.assertEqual(schema["tasks"]["species"]["num_classes"], 2)
            self.assertEqual(len(schema["concept_names"]), 3)

            train_rows = [
                json.loads(line)
                for line in (output_dir / "train" / "metadata.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(train_rows)
            self.assertTrue(Path(train_rows[0]["image_path"]).is_absolute())
            self.assertEqual(sorted(train_rows[0]["task_labels"]), ["species"])

    def test_koh112_policy_uses_official_attribute_subset_and_class_level_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cub_root = root / "CUB_200_2011"
            self._write_fake_cub(cub_root, num_classes=12, images_per_class=4)
            output_dir = root / "cub_manifest"

            result = build_cub_manifest(
                cub_root=cub_root,
                output_dir=output_dir,
                attribute_policy="koh112",
                class_ids=list(range(1, 13)),
                val_fraction=0.0,
                seed=0,
            )

            self.assertEqual(result["num_attributes"], 112)
            schema = json.loads((output_dir / "train" / "schema.json").read_text())
            self.assertEqual(schema["attribute_policy"], "koh112")
            self.assertEqual(
                schema["concept_encoding"],
                "binary_class_attr_data_10_koh112",
            )
            self.assertEqual(len(schema["concept_attribute_ids"]), 112)
            self.assertEqual(schema["concept_attribute_ids"][0], 2)

            train_rows = [
                json.loads(line)
                for line in (output_dir / "train" / "metadata.jsonl")
                .read_text()
                .splitlines()
            ]
            rows_by_class = {}
            for row in train_rows:
                rows_by_class.setdefault(row["class_id"], set()).add(
                    tuple(row["concept_vector"])
                )
            self.assertTrue(rows_by_class)
            for vectors in rows_by_class.values():
                self.assertEqual(len(vectors), 1)

    def test_builder_accepts_root_level_attributes_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cub_root = root / "CUB_200_2011"
            self._write_fake_cub(cub_root)
            (cub_root / "attributes.txt").write_text(
                (cub_root / "attributes" / "attributes.txt").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            (cub_root / "attributes" / "attributes.txt").unlink()
            output_dir = root / "cub_manifest"

            result = build_cub_manifest(
                cub_root=cub_root,
                output_dir=output_dir,
                num_attributes=3,
                class_ids=[1, 2],
                val_fraction=0.0,
                seed=0,
            )

            self.assertEqual(result["num_attributes"], 3)
            schema = json.loads((output_dir / "train" / "schema.json").read_text())
            self.assertEqual(len(schema["concept_names"]), 3)

    def _write_fake_cub(
        self,
        cub_root: Path,
        num_classes: int = 2,
        images_per_class: int = 4,
    ) -> None:
        (cub_root / "attributes").mkdir(parents=True)
        for class_id in range(1, num_classes + 1):
            (cub_root / "images" / f"{class_id:03d}.Class_{class_id}").mkdir(
                parents=True
            )

        image_rows = []
        class_rows = []
        split_rows = []
        image_id = 1
        for class_id in range(1, num_classes + 1):
            class_dir = f"{class_id:03d}.Class_{class_id}"
            for offset in range(images_per_class):
                image_name = f"image_{image_id}.jpg"
                image_path = cub_root / "images" / class_dir / image_name
                Image.new("RGB", (8, 8), (image_id, image_id, image_id)).save(
                    image_path
                )
                image_rows.append(f"{image_id} {class_dir}/{image_name}")
                class_rows.append(f"{image_id} {class_id}")
                split_rows.append(f"{image_id} {1 if image_id % 2 == 0 else 0}")
                image_id += 1

        (cub_root / "images.txt").write_text("\n".join(image_rows) + "\n")
        (cub_root / "image_class_labels.txt").write_text("\n".join(class_rows) + "\n")
        (cub_root / "train_test_split.txt").write_text("\n".join(split_rows) + "\n")
        (cub_root / "classes.txt").write_text(
            "\n".join(
                f"{class_id} {class_id:03d}.Class_{class_id}"
                for class_id in range(1, num_classes + 1)
            )
            + "\n"
        )
        (cub_root / "attributes" / "attributes.txt").write_text(
            "\n".join(
                f"{attr_id} has_part::attribute_{attr_id}"
                for attr_id in range(1, 313)
            )
            + "\n"
        )
        attr_lines = []
        for image_id in range(1, image_id):
            class_id = 1 + ((image_id - 1) // images_per_class)
            for attr_id in range(1, 313):
                if attr_id % 5 == 0:
                    label = 1
                elif attr_id % 7 == 0:
                    label = 1 if class_id <= max(1, num_classes - 1) else 0
                elif attr_id % 11 == 0:
                    label = 1 if class_id == 1 else 0
                else:
                    label = 0
                attr_lines.append(f"{image_id} {attr_id} {label} 3 1.0")
        (cub_root / "attributes" / "image_attribute_labels.txt").write_text(
            "\n".join(attr_lines) + "\n"
        )


if __name__ == "__main__":
    unittest.main()
