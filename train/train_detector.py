#!/usr/bin/env python3
"""
Training script for License Plate Detection & Classification (YOLO).
Optimized for Russian non-standard license plates (type1, type1a, type1b, other).
"""

import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO


def train_detector(
    data_yaml: str = "dataset/dataset.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 10,
    batch_size: int = 32,
    imgsz: int = 640,
    device: str = "0",
    output_weights: str = "models/detector_best.pt"
):
    print("=" * 60)
    print(" Starting Plate Detector Training (YOLO)")
    print("=" * 60)
    print(f"Data config:     {data_yaml}")
    print(f"Base model:      {base_model}")
    print(f"Epochs:          {epochs}")
    print(f"Batch size:      {batch_size}")
    print(f"Target weights:  {output_weights}")
    print("=" * 60)

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=4,
        save=True,
        project="train_runs",
        name="plate_detector",
        exist_ok=True,
        verbose=True
    )

    best_pt = os.path.join("train_runs", "plate_detector", "weights", "best.pt")
    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(output_weights), exist_ok=True)
        shutil.copy2(best_pt, output_weights)
        print(f"\nSuccessfully saved best detector weights to: {output_weights}")
    else:
        print(f"\nWarning: {best_pt} not found. Saving last.pt if available.")
        last_pt = os.path.join("train_runs", "plate_detector", "weights", "last.pt")
        if os.path.exists(last_pt):
            shutil.copy2(last_pt, output_weights)
            print(f"Saved {last_pt} to {output_weights}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Plate Detector")
    parser.add_argument('--data', default="dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--output', default="models/detector_best.pt", help="Output path for best weights")
    args = parser.parse_args()

    train_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_weights=args.output
    )
