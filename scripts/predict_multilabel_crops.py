#!/usr/bin/env python3
"""
Predict multi-label behaviors on cropped-person images using the custom YOLOv8 pipeline.
Outputs per-box scores for all classes via boxes_all.
"""

import argparse
import json
import numpy as np
from types import MethodType
from pathlib import Path
from ultralytics import YOLO
from YOLO.postprocess import postprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Predict multi-label behaviors on person crops.")
    parser.add_argument("--weights", required=True, help="Path to trained weights (e.g., best.pt).")
    parser.add_argument("--source", required=True, help="Image path, directory, or glob.")
    parser.add_argument("--imgsz", type=int, default=224, help="Input size (must match training, e.g., 224).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold.")
    parser.add_argument("--device", default="0", help="Device id, e.g., '0' or 'cpu'.")
    parser.add_argument("--save", action="store_true", help="Save annotated images to runs/predict-multilabel/")
    parser.add_argument("--jsonl-output", default=None, help="Optional JSONL output path with all per-box scores.")
    return parser.parse_args()


def warmup_and_patch(model, imgsz):
    """Initialize predictor and attach multi-label postprocess."""
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=imgsz, verbose=False, save=False)
    model.predictor.postprocess = MethodType(postprocess, model.predictor)


def run_inference(args):
    model = YOLO(args.weights)
    warmup_and_patch(model, args.imgsz)

    results_stream = model.predict(
        args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        max_det=100,
        verbose=True,
        stream=True,
        project="runs/predict-multilabel",
        name="exp"
    )

    jsonl_path = Path(args.jsonl_output) if args.jsonl_output else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file = jsonl_path.open("w") if jsonl_path else None

    for res in results_stream:
        names = res.names
        boxes = []
        for idx in range(len(res.boxes)):
            boxes.append({
                "xyxy": res.boxes.xyxy[idx].tolist(),
                "score": float(res.boxes.conf[idx]),
                "top1_class": names[int(res.boxes.cls[idx])],
                "all_scores": res.boxes_all[idx, 4:].tolist(),  # full per-class scores
            })

        record = {"image": res.path, "boxes": boxes}
        if jsonl_file:
            jsonl_file.write(json.dumps(record) + "\n")

    if jsonl_file:
        jsonl_file.close()
        print(f"Saved JSONL predictions to {jsonl_path}")


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
