#!/usr/bin/env python3
"""
Main execution script for Volga-IT Semi-Final:
Recognition of Non-Standard License Plates.

Usage:
    python run.py --input-dir <path/to/images> --output <results.csv> [--device cuda/cpu] [--benchmark]
"""

import os
import sys
import argparse
import torch
from src.pipeline import LicensePlatePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Volga-IT: Non-Standard License Plate Recognition Pipeline"
    )
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help="Path to folder containing test images (.jpg, .png)"
    )
    parser.add_argument(
        '--output', '-o',
        default="results.csv",
        help="Path to save output CSV file (default: results.csv)"
    )
    parser.add_argument(
        '--detector-weights',
        default="models/detector_best.pt",
        help="Path to trained detector weights"
    )
    parser.add_argument(
        '--ocr-weights',
        default="models/ocr_best.pt",
        help="Path to trained OCR weights"
    )
    parser.add_argument(
        '--device',
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help="Print real-time throughput and latency metrics"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" Volga-IT: Non-Standard License Plate Recognition Pipeline")
    print("=" * 60)
    print(f"Input Directory:   {args.input_dir}")
    print(f"Output CSV:        {args.output}")
    print(f"Detector Weights:  {args.detector_weights}")
    print(f"OCR Weights:       {args.ocr_weights}")
    print(f"Device:            {args.device}")
    print("=" * 60)

    pipeline = LicensePlatePipeline(
        detector_weights=args.detector_weights,
        ocr_weights=args.ocr_weights,
        device=args.device
    )

    pipeline.process_directory(
        input_dir=args.input_dir,
        output_csv_path=args.output,
        benchmark=args.benchmark
    )


if __name__ == '__main__':
    main()
