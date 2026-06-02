#!/usr/bin/env python3
"""
Example: Visualize annotations by drawing bounding boxes on images.

Usage:
    python examples/visualize.py --image path/to/image.jpg --labels path/to/labels.txt
    python examples/visualize.py --coco annotations.json --output-dir viz/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Colors for different classes (BGR format)
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Dark Red
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Dark Blue
]


def draw_yolo_bboxes(image_path: str, label_path: str, class_names: list[str]) -> np.ndarray:
    """Draw YOLO format bounding boxes on an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # Denormalize
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                color = COLORS[cls_id % len(COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bounding box annotations")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--labels", help="Path to YOLO label file")
    parser.add_argument("--classes", default="", help="Path to classes.txt")
    parser.add_argument("--output", "-o", default="", help="Output image path")
    args = parser.parse_args()

    # Load class names
    class_names = []
    if args.classes:
        with open(args.classes) as f:
            class_names = [line.strip() for line in f]

    if args.image and args.labels:
        img = draw_yolo_bboxes(args.image, args.labels, class_names)
        if args.output:
            cv2.imwrite(args.output, img)
            print(f"✅ Saved visualization to {args.output}")
        else:
            cv2.imshow("Annotations", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
