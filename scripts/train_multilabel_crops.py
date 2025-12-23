#!/usr/bin/env python3
"""
Train the multi-label YOLOv8 model on cropped-person images (224px).
Each crop should contain a single person; the box spans the full image.
"""

import argparse
from YOLO.trainer import YOLOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label YOLOv8 on person crops.")
    parser.add_argument("--data", default="./your_dataset.yaml", help="Path to dataset YAML.")
    parser.add_argument("--weights", default="YOLO_models/yolov8n.pt", help="Base model weights.")
    parser.add_argument("--project", default="YOLO_multilabel_224", help="Project/output folder name.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", default="0", help="Device id, e.g., '0' or 'cpu'.")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size.")
    parser.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate.")
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam", "AdamW"], help="Optimizer type.")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--use-constraints", action="store_true", help="Enable constraint loss (requires YOLO/constraints.npy).")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience (epochs).")
    parser.add_argument("--save-period", type=int, default=50, help="Checkpoint save period (epochs).")
    return parser.parse_args()


def main():
    args = parse_args()

    close_mosaic = max(1, args.epochs // 4)  # disable mosaic in the final quarter of training

    overrides = {
        "device": args.device,
        "project": args.project,
        "data": args.data,
        "task": "detect",
        "model": args.weights,
        "epochs": args.epochs,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "patience": args.patience,
        "save_period": args.save_period,
        # Light augmentations for 224px person crops
        "hsv_h": 0.012, "hsv_s": 0.5, "hsv_v": 0.4,   # mild color jitter to keep skin/cloth tones realistic
        "degrees": 2.5, "translate": 0.04, "scale": 0.2, "shear": 1.0,  # gentle geo jitter; scale limited because box spans crop
        "fliplr": 0.5, "flipud": 0.0,
        "mosaic": 0.5, "mixup": 0.0, "copy_paste": 0.0,  # higher mosaic for more spatial variety; mixup off for single-person crops
        "close_mosaic": close_mosaic,
    }

    trainer = YOLOTrainer(overrides=overrides, req_loss=args.use_constraints)
    trainer.train()


if __name__ == "__main__":
    main()
