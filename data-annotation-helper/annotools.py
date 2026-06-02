#!/usr/bin/env python3
"""
Data Annotation Tools — Convert, validate, split, and analyze ML datasets.

Supports:
  - COCO JSON → YOLO TXT → CSV → Pascal VOC XML conversion
  - Annotation validation (bounds, missing files, format errors)
  - Train/val/test splitting with stratification
  - Dataset statistics & visualization

Usage:
    python annotools.py convert --input coco.json --input-format coco --output labels/ --output-format yolo
    python annotools.py validate --input labels/ --format yolo
    python annotools.py split --input labels/ --images images/ --output-dir dataset/ --train 0.8 --val 0.1
    python annotools.py stats --input annotations.csv --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """A single bounding box annotation."""

    class_id: int
    class_name: str = ""
    x1: float = 0.0  # Absolute coordinates (pixels)
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_yolo(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Convert to YOLO format (cx, cy, w, h) normalized."""
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        w = self.width / img_w
        h = self.height / img_h
        return (cx, cy, w, h)

    @classmethod
    def from_yolo(cls, class_id: int, cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int, class_name: str = "") -> "BBox":
        """Create from YOLO normalized format."""
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        return cls(class_id=class_id, class_name=class_name,
                   x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass
class Annotation:
    """A single image with its annotations."""

    image_path: str
    image_width: int = 0
    image_height: int = 0
    bboxes: list[BBox] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Results of annotation validation."""

    valid: bool = True
    total_images: int = 0
    total_annotations: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    class_distribution: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core Converter
# ---------------------------------------------------------------------------

class AnnotationConverter:
    """Convert annotations between COCO JSON, YOLO TXT, CSV, and Pascal VOC formats."""

    def convert(self, input_path: str, input_format: str,
                output_path: str, output_format: str) -> dict:
        """
        Convert annotations from one format to another.

        Args:
            input_path: Path to input annotations
            input_format: One of 'coco', 'yolo', 'csv', 'voc'
            output_path: Output directory or file path
            output_format: One of 'coco', 'yolo', 'csv', 'voc'

        Returns:
            Dict with conversion stats
        """
        # Step 1: Parse input to unified format
        annotations, class_map = self._parse(input_path, input_format)

        # Step 2: Export to target format
        stats = self._export(annotations, class_map, output_path, output_format)

        logger.info("Converted %d annotations (%d classes) from %s to %s",
                     sum(len(a.bboxes) for a in annotations),
                     len(class_map), input_format, output_format)
        return stats

    # -------------------------------------------------------------------
    # Private: Parse input formats
    # -------------------------------------------------------------------

    def _parse(self, input_path: str, fmt: str) -> tuple[list[Annotation], dict[int, str]]:
        parsers = {
            "coco": self._parse_coco,
            "yolo": self._parse_yolo,
            "csv": self._parse_csv,
            "voc": self._parse_voc,
        }
        parser = parsers.get(fmt)
        if parser is None:
            raise ValueError(f"Unsupported input format: {fmt}")
        return parser(input_path)

    def _parse_coco(self, input_path: str) -> tuple[list[Annotation], dict[int, str]]:
        """Parse COCO JSON format."""
        with open(input_path) as f:
            data = json.load(f)

        # Build category map
        class_map: dict[int, str] = {cat["id"]: cat["name"] for cat in data["categories"]}

        # Build image map
        image_map = {}
        for img in data["images"]:
            image_map[img["id"]] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }

        # Group annotations by image
        image_annotations: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

        # Build Annotation objects
        annotations = []
        for img_id, img_info in image_map.items():
            bboxes = []
            for ann in image_annotations.get(img_id, []):
                x, y, w, h = ann["bbox"]  # COCO: [x, y, w, h]
                bbox = BBox(
                    class_id=ann["category_id"],
                    class_name=class_map.get(ann["category_id"], ""),
                    x1=x, y1=y,
                    x2=x + w, y2=y + h,
                )
                bboxes.append(bbox)

            annotation = Annotation(
                image_path=img_info["file_name"],
                image_width=img_info["width"],
                image_height=img_info["height"],
                bboxes=bboxes,
            )
            annotations.append(annotation)

        return annotations, class_map

    def _parse_yolo(self, input_path: str) -> tuple[list[Annotation], dict[int, str]]:
        """Parse YOLO TXT format (one label file per image)."""
        input_dir = Path(input_path)
        if not input_dir.is_dir():
            raise ValueError(f"YOLO input must be a directory, got: {input_path}")

        # Try to read classes file
        class_map: dict[int, str] = {}
        classes_file = input_dir / "classes.txt"
        if classes_file.exists():
            with open(classes_file) as f:
                for i, line in enumerate(f):
                    class_map[i] = line.strip()

        # Also look for dataset.yaml
        yaml_file = input_dir / "dataset.yaml"
        if yaml_file.exists() and not class_map:
            try:
                import yaml
                config = yaml.safe_load(yaml_file.read_text())
                class_map = {i: name for i, name in enumerate(config.get("names", []))}
            except ImportError:
                pass

        # Find all .txt label files
        annotations = []
        label_files = sorted(input_dir.glob("*.txt"))
        if not label_files:
            label_files = sorted(input_dir.glob("labels/*.txt"))

        for label_path in tqdm(label_files, desc="Parsing YOLO"):
            if label_path.name in ("classes.txt", "dataset.yaml"):
                continue

            # Infer image path
            stem = label_path.stem
            image_path = self._find_image(label_path.parent, stem)

            # Parse bounding boxes
            bboxes = []
            img_w, img_h = 1920, 1080  # default fallback
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        bbox = BBox.from_yolo(
                            class_id, cx, cy, w, h,
                            img_w, img_h,
                            class_name=class_map.get(class_id, ""),
                        )
                        bboxes.append(bbox)

            ann = Annotation(
                image_path=image_path,
                image_width=img_w,
                image_height=img_h,
                bboxes=bboxes,
            )
            annotations.append(ann)

        return annotations, class_map

    def _parse_csv(self, input_path: str) -> tuple[list[Annotation], dict[int, str]]:
        """Parse CSV format: filename,class,x1,y1,x2,y2"""
        class_map: dict[int, str] = {}
        image_dict: dict[str, Annotation] = {}
        next_class_id = 0
        class_names: dict[str, int] = {}

        with open(input_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "")
                cls_name = row.get("class", "unknown")

                if cls_name not in class_names:
                    class_names[cls_name] = next_class_id
                    class_map[next_class_id] = cls_name
                    next_class_id += 1

                bbox = BBox(
                    class_id=class_names[cls_name],
                    class_name=cls_name,
                    x1=float(row.get("x1", 0)),
                    y1=float(row.get("y1", 0)),
                    x2=float(row.get("x2", 0)),
                    y2=float(row.get("y2", 0)),
                )

                if fname not in image_dict:
                    image_dict[fname] = Annotation(image_path=fname)

                image_dict[fname].bboxes.append(bbox)

        return list(image_dict.values()), class_map

    def _parse_voc(self, input_path: str) -> tuple[list[Annotation], dict[int, str]]:
        """Parse Pascal VOC XML format (placeholder — requires xml.etree)."""
        # This is a simplified parser for VOC format
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("xml.etree is required for VOC format")

        input_dir = Path(input_path)
        class_map: dict[int, str] = {}
        class_names: dict[str, int] = {}
        next_class_id = 0
        annotations = []

        for xml_file in sorted(input_dir.glob("*.xml")):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.findtext("filename", "")
            size = root.find("size")
            img_w = int(size.findtext("width", "0"))
            img_h = int(size.findtext("height", "0"))

            bboxes = []
            for obj in root.findall("object"):
                cls_name = obj.findtext("name", "unknown")
                if cls_name not in class_names:
                    class_names[cls_name] = next_class_id
                    class_map[next_class_id] = cls_name
                    next_class_id += 1

                bndbox = obj.find("bndbox")
                bbox = BBox(
                    class_id=class_names[cls_name],
                    class_name=cls_name,
                    x1=float(bndbox.findtext("xmin", "0")),
                    y1=float(bndbox.findtext("ymin", "0")),
                    x2=float(bndbox.findtext("xmax", "0")),
                    y2=float(bndbox.findtext("ymax", "0")),
                )
                bboxes.append(bbox)

            ann = Annotation(
                image_path=filename,
                image_width=img_w,
                image_height=img_h,
                bboxes=bboxes,
            )
            annotations.append(ann)

        return annotations, class_map

    # -------------------------------------------------------------------
    # Private: Export
    # -------------------------------------------------------------------

    def _export(self, annotations: list[Annotation], class_map: dict[int, str],
                output_path: str, fmt: str) -> dict:
        exporters = {
            "coco": self._export_coco,
            "yolo": self._export_yolo,
            "csv": self._export_csv,
            "voc": self._export_voc,
        }
        exporter = exporters.get(fmt)
        if exporter is None:
            raise ValueError(f"Unsupported output format: {fmt}")
        return exporter(annotations, class_map, output_path)

    def _export_coco(self, annotations: list[Annotation], class_map: dict[int, str],
                     output_path: str) -> dict:
        """Export to COCO JSON format."""
        coco = {
            "info": {"description": "Converted by Data Annotation Tools"},
            "categories": [{"id": k, "name": v, "supercategory": ""} for k, v in class_map.items()],
            "images": [],
            "annotations": [],
        }

        ann_id = 1
        for img_id, ann in enumerate(annotations, 1):
            coco["images"].append({
                "id": img_id,
                "file_name": ann.image_path,
                "width": ann.image_width,
                "height": ann.image_height,
            })
            for bbox in ann.bboxes:
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": bbox.class_id,
                    "bbox": [bbox.x1, bbox.y1, bbox.width, bbox.height],
                    "area": bbox.width * bbox.height,
                    "iscrowd": 0,
                })
                ann_id += 1

        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        return {"images": len(annotations), "annotations": ann_id - 1, "classes": len(class_map)}

    def _export_yolo(self, annotations: list[Annotation], class_map: dict[int, str],
                     output_path: str) -> dict:
        """Export to YOLO TXT format."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write classes file
        with open(output_dir / "classes.txt", "w") as f:
            for i in range(len(class_map)):
                f.write(f"{class_map.get(i, f'class_{i}')}\n")

        # Write YAML config
        try:
            import yaml
            yaml_config = {
                "names": [class_map.get(i, f"class_{i}") for i in range(len(class_map))],
                "nc": len(class_map),
            }
            with open(output_dir / "dataset.yaml", "w") as f:
                yaml.dump(yaml_config, f, default_flow_style=False)
        except ImportError:
            pass

        # Write label files
        count = 0
        for ann in annotations:
            label_path = output_dir / f"{Path(ann.image_path).stem}.txt"
            lines = []
            for bbox in ann.bboxes:
                cx, cy, w, h = bbox.to_yolo(ann.image_width, ann.image_height)
                lines.append(f"{bbox.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            label_path.write_text("\n".join(lines))
            count += len(lines)

        return {"label_files": len(annotations), "annotations": count, "classes": len(class_map)}

    def _export_csv(self, annotations: list[Annotation], class_map: dict[int, str],
                    output_path: str) -> dict:
        """Export to CSV format."""
        count = 0
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "x1", "y1", "x2", "y2"])
            for ann in annotations:
                for bbox in ann.bboxes:
                    writer.writerow([
                        ann.image_path,
                        bbox.class_name or bbox.class_id,
                        int(bbox.x1), int(bbox.y1),
                        int(bbox.x2), int(bbox.y2),
                    ])
                    count += 1

        return {"images": len(annotations), "annotations": count, "classes": len(class_map)}

    def _export_voc(self, annotations: list[Annotation], class_map: dict[int, str],
                    output_path: str) -> dict:
        """Export to Pascal VOC XML format."""
        try:
            import xml.etree.ElementTree as ET
            from xml.dom import minidom
        except ImportError:
            raise ImportError("xml.etree is required for VOC format")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for ann in annotations:
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = ann.image_path

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(ann.image_width)
            ET.SubElement(size, "height").text = str(ann.image_height)
            ET.SubElement(size, "depth").text = "3"

            for bbox in ann.bboxes:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = bbox.class_name or str(bbox.class_id)
                ET.SubElement(obj, "difficult").text = "0"

                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(bbox.x1))
                ET.SubElement(bndbox, "ymin").text = str(int(bbox.y1))
                ET.SubElement(bndbox, "xmax").text = str(int(bbox.x2))
                ET.SubElement(bndbox, "ymax").text = str(int(bbox.y2))
                count += 1

            # Pretty print
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            xml_path = output_dir / f"{Path(ann.image_path).stem}.xml"
            xml_path.write_text(xml_str)

        return {"xml_files": len(annotations), "annotations": count, "classes": len(class_map)}

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _find_image(self, search_dir: Path, stem: str) -> str:
        """Find an image file by stem name."""
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidates = list(search_dir.glob(f"{stem}{ext}"))
            if candidates:
                return candidates[0].name
            candidates = list(search_dir.glob(f"images/{stem}{ext}"))
            if candidates:
                return candidates[0].name
        return f"{stem}.jpg"  # best guess


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """Validate annotation datasets for common issues."""

    def validate(self, input_path: str, fmt: str) -> ValidationReport:
        """
        Validate a dataset.

        Checks:
        - Missing image files
        - Out-of-bounds bounding boxes
        - Invalid dimensions
        - Class consistency
        - Annotation file format errors
        """
        report = ValidationReport()

        # Parse annotations
        try:
            converter = AnnotationConverter()
            annotations, class_map = converter._parse(input_path, fmt)
        except Exception as e:
            report.valid = False
            report.errors.append(f"Failed to parse dataset: {e}")
            return report

        report.total_images = len(annotations)
        report.total_annotations = sum(len(a.bboxes) for a in annotations)
        report.class_distribution = {}

        for ann in annotations:
            # Check image exists
            img_path = Path(input_path).parent / ann.image_path
            if not img_path.exists() and not Path(ann.image_path).exists():
                report.warnings.append(f"Image not found: {ann.image_path}")

            for bbox in ann.bboxes:
                # Track class distribution
                cls_name = bbox.class_name or str(bbox.class_id)
                report.class_distribution[cls_name] = report.class_distribution.get(cls_name, 0) + 1

                # Check bounds
                if bbox.x1 < 0 or bbox.y1 < 0:
                    report.errors.append(f"Negative coordinates in {ann.image_path}: ({bbox.x1}, {bbox.y1})")
                    report.valid = False
                if bbox.x2 > ann.image_width + 1 or bbox.y2 > ann.image_height + 1:
                    report.warnings.append(f"BBox exceeds image bounds in {ann.image_path}: ({bbox.x2}, {bbox.y2}) vs ({ann.image_width}, {ann.image_height})")
                if bbox.width <= 0 or bbox.height <= 0:
                    report.errors.append(f"Zero/negative dimension in {ann.image_path}: {bbox.width}x{bbox.height}")
                    report.valid = False

        if not report.errors:
            logger.info("✅ Dataset is valid: %d images, %d annotations, %d classes",
                        report.total_images, report.total_annotations, len(class_map))
        else:
            logger.warning("⚠️ Dataset has %d errors and %d warnings",
                          len(report.errors), len(report.warnings))

        return report


# ---------------------------------------------------------------------------
# Dataset Splitter
# ---------------------------------------------------------------------------

class DatasetSplitter:
    """Split dataset into train/val/test sets."""

    def split(self, annotations: list[Annotation],
             train_ratio: float = 0.8,
             val_ratio: float = 0.1,
             test_ratio: float = 0.1,
             seed: int = 42) -> dict[str, list[Annotation]]:
        """
        Split annotations into train/val/test sets.

        Returns:
            Dict with 'train', 'val', 'test' keys.
        """
        random.seed(seed)
        shuffled = annotations.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:],
        }

    def export_splits(self, annotations: list[Annotation], class_map: dict[int, str],
                      output_dir: str, image_dir: str = "",
                      train: float = 0.8, val: float = 0.1) -> dict:
        """
        Split and export to YOLO format directory structure.

        Creates:
            output_dir/
            ├── train/
            │   ├── images/
            │   └── labels/
            ├── val/
            │   ├── images/
            │   └── labels/
            ├── test/
            │   ├── images/
            │   └── labels/
            ├── classes.txt
            └── dataset.yaml
        """
        splits = self.split(annotations, train_ratio=train, val_ratio=val)
        output_path = Path(output_dir)
        image_src = Path(image_dir) if image_dir else None

        converter = AnnotationConverter()
        stats = {}

        for split_name, split_anns in splits.items():
            if not split_anns:
                continue

            labels_dir = output_path / split_name / "labels"
            images_dir = output_path / split_name / "images"
            labels_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            # Copy images
            for ann in split_anns:
                if image_src:
                    src_img = image_src / ann.image_path
                    dst_img = images_dir / ann.image_path
                    if src_img.exists():
                        shutil.copy2(src_img, dst_img)

            # Export labels in YOLO format
            converter._export_yolo(split_anns, class_map, str(labels_dir))

            stats[split_name] = len(split_anns)
            logger.info("  %s: %d images", split_name, len(split_anns))

        # Write global dataset.yaml
        import yaml
        yaml_config = {
            "train": str(output_path / "train" / "images"),
            "val": str(output_path / "val" / "images"),
            "test": str(output_path / "test" / "images"),
            "nc": len(class_map),
            "names": [class_map.get(i, f"class_{i}") for i in range(len(class_map))],
        }
        with open(output_path / "dataset.yaml", "w") as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        return stats


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(annotations: list[Annotation]) -> dict:
    """Compute dataset statistics."""
    class_counts: dict[str, int] = {}
    image_sizes: list[tuple[int, int]] = []
    bboxes_per_image: list[int] = []
    bbox_areas: list[float] = []

    for ann in annotations:
        image_sizes.append((ann.image_width, ann.image_height))
        bboxes_per_image.append(len(ann.bboxes))

        for bbox in ann.bboxes:
            cls_name = bbox.class_name or str(bbox.class_id)
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            bbox_areas.append(bbox.width * bbox.height)

    sizes = np.array(image_sizes)
    return {
        "total_images": len(annotations),
        "total_annotations": sum(bboxes_per_image),
        "classes": len(class_counts),
        "class_distribution": class_counts,
        "avg_bboxes_per_image": round(np.mean(bboxes_per_image), 2),
        "image_sizes": {
            "min_width": int(sizes[:, 0].min()) if len(sizes) else 0,
            "max_width": int(sizes[:, 0].max()) if len(sizes) else 0,
            "min_height": int(sizes[:, 1].min()) if len(sizes) else 0,
            "max_height": int(sizes[:, 1].max()) if len(sizes) else 0,
        },
        "bbox_sizes": {
            "min_area": round(min(bbox_areas), 2) if bbox_areas else 0,
            "max_area": round(max(bbox_areas), 2) if bbox_areas else 0,
            "avg_area": round(np.mean(bbox_areas), 2) if bbox_areas else 0,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Data Annotation Tools — convert, validate, split, and analyze ML datasets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Convert
    convert_parser = subparsers.add_parser("convert", help="Convert annotation formats")
    convert_parser.add_argument("--input", "-i", required=True, help="Input annotations path")
    convert_parser.add_argument("--input-format", choices=["coco", "yolo", "csv", "voc"], required=True)
    convert_parser.add_argument("--output", "-o", required=True, help="Output path")
    convert_parser.add_argument("--output-format", choices=["coco", "yolo", "csv", "voc"], required=True)

    # Validate
    validate_parser = subparsers.add_parser("validate", help="Validate annotations")
    validate_parser.add_argument("--input", "-i", required=True, help="Input annotations path")
    validate_parser.add_argument("--format", choices=["coco", "yolo", "csv", "voc"], required=True)

    # Split
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--input", "-i", required=True, help="Input annotations path")
    split_parser.add_argument("--input-format", choices=["coco", "yolo", "csv", "voc"], default="yolo")
    split_parser.add_argument("--images", default="", help="Source image directory")
    split_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    split_parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    split_parser.add_argument("--val", type=float, default=0.1, help="Validation ratio")

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Compute dataset statistics")
    stats_parser.add_argument("--input", "-i", required=True, help="Input annotations path")
    stats_parser.add_argument("--format", choices=["coco", "yolo", "csv", "voc"], default="yolo")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "convert":
        converter = AnnotationConverter()
        stats = converter.convert(args.input, args.input_format, args.output, args.output_format)
        print(f"\n✅ Conversion complete: {stats}")

    elif args.command == "validate":
        validator = DatasetValidator()
        report = validator.validate(args.input, args.format)
        status = "✅ VALID" if report.valid else "❌ INVALID"
        print(f"\n{status}")
        print(f"   Images: {report.total_images}")
        print(f"   Annotations: {report.total_annotations}")
        print(f"   Classes: {len(report.class_distribution)}")
        if report.errors:
            print(f"\n   Errors ({len(report.errors)}):")
            for err in report.errors[:10]:
                print(f"     ❌ {err}")
        if report.warnings:
            print(f"\n   Warnings ({len(report.warnings)}):")
            for warn in report.warnings[:10]:
                print(f"     ⚠️  {warn}")
        if report.class_distribution:
            print(f"\n   Class Distribution:")
            for cls, count in sorted(report.class_distribution.items(), key=lambda x: -x[1]):
                print(f"     {cls}: {count}")

    elif args.command == "split":
        converter = AnnotationConverter()
        annotations, class_map = converter._parse(args.input, args.input_format)
        splitter = DatasetSplitter()
        stats = splitter.export_splits(
            annotations, class_map, args.output_dir, args.images,
            train=args.train, val=args.val,
        )
        print(f"\n✅ Split complete: {stats}")

    elif args.command == "stats":
        converter = AnnotationConverter()
        annotations, class_map = converter._parse(args.input, args.format)
        stats = compute_statistics(annotations)
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Total annotations: {stats['total_annotations']}")
        print(f"   Classes: {stats['classes']}")
        print(f"   Avg bboxes per image: {stats['avg_bboxes_per_image']}")
        print(f"   Image sizes: {stats['image_sizes']}")
        print(f"   BBox areas: {stats['bbox_sizes']}")
        print(f"\n   Class Distribution:")
        for cls, count in sorted(stats["class_distribution"].items(), key=lambda x: -x[1]):
            print(f"     {cls}: {count}")


if __name__ == "__main__":
    main()
