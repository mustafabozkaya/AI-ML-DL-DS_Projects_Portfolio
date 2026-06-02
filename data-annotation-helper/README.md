# 🏷️ Data Annotation Helper

A versatile toolkit for AI/ML data annotation tasks — format conversion, validation,
dataset splitting, and quality checks for object detection and image classification datasets.

> **Purpose:** Portfolio project for Upwork — demonstrates data annotation, dataset 
> preparation, computer vision pipeline, and Python scripting skills.

---

## ✨ Features

- **Format conversion** — COCO JSON ↔ YOLO TXT ↔ CSV annotation formats
- **Annotation validation** — Check for missing/corrupted labels, out-of-bounds boxes
- **Dataset splitting** — Train/val/test split with stratification
- **Class mapping** — Merge, rename, or filter annotation classes
- **Visual inspection** — Draw bounding boxes to verify annotations
- **Dataset statistics** — Class distribution, image sizes, annotation density
- **CLI interface** — Easy to use from command line

---

## 📦 Installation

```bash
git clone https://github.com/MrBozkay/data-annotation-helper.git
cd data-annotation-helper
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# Convert COCO to YOLO format
python annotools.py convert --input annotations.json --input-format coco \
  --output labels/ --output-format yolo

# Validate annotations
python annotools.py validate --input labels/ --format yolo

# Split dataset (80/10/10)
python annotools.py split --input labels/ --images images/ \
  --output-dir dataset/ --train 0.8 --val 0.1

# Get dataset statistics
python annotools.py stats --input labels/ --format yolo
```

## 🐍 Python API

```python
from annotools import AnnotationConverter, DatasetValidator, DatasetSplitter

# Convert annotations
converter = AnnotationConverter()
converter.convert("annotations.json", "coco", "labels/", "yolo")

# Validate
validator = DatasetValidator()
report = validator.validate("labels/", "yolo")
print(f"Valid: {report.valid}, Errors: {len(report.errors)}")

# Split
splitter = DatasetSplitter()
splits = splitter.split("dataset/", train=0.8, val=0.1, test=0.1)
```

---

## 📋 Supported Annotation Formats

| Format | Description | Common Use |
|---|---|---|
| **COCO JSON** | Standard JSON with images/annotations/categories | Research, Detectron2, MMDetection |
| **YOLO TXT** | One file per image, `class x_center y_center width height` | YOLOv5, YOLOv8, Ultralytics |
| **CSV** | Flat table with `filename,class,x1,y1,x2,y2` | General purpose, spreadsheet editing |
| **Pascal VOC XML** | XML format per image | Older detection pipelines |

---

## 🏗️ Project Structure

```
data-annotation-helper/
├── annotools.py           # Main toolkit
├── requirements.txt       # Dependencies
├── README.md             # This file
└── examples/
    ├── sample_coco.json  # Sample COCO annotations
    └── visualize.py      # Annotation visualization example
```

## 🔧 Requirements

- Python 3.8+
- numpy
- Pillow
- opencv-python (for visualization)
- tqdm (for progress bars)

---

## 📋 Use Cases on Upwork

| Job Type | How This Helps |
|---|---|
| Data Annotation | Format conversion, validation, quality control |
| Bounding Box Labeling | Validate and convert client annotation exports |
| Dataset Preparation | Split, shuffle, prepare for model training |
| YOLOv8 Training Prep | Convert any annotation format to YOLO format |
| Computer Vision | Full dataset management pipeline |
| AI/ML Data Collection | End-to-end dataset quality assurance |

---

## 📄 License

MIT — Free to use and modify for your projects.
