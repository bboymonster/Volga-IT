#!/usr/bin/env python3
"""
Dataset Validator for Volga-IT: Recognition of Non-Standard License Plates.
Conforms strictly to Sections 6-8 of the competition specification.
Validates structure, meta.csv, labels, image integrity, bounding boxes, quads,
plate syntax, and generates a detailed validation report.
"""

import os
import sys
import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configuration & constraints
ALLOWED_CHARS_REGEX = re.compile(r'^[ABEKMHOPCTYX\d#]+$')
PLATE_REGEX = re.compile(r'^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}(\d{2}|[127]\d{2})$')
VALID_PLATE_TYPES = {'type1', 'type1a', 'type1b', 'other'}
VALID_CONDITIONS = {'day', 'night', 'rain', 'snow', 'dirt', 'glare', 'motion_blur', 'angle'}

REQUIRED_META_FIELDS = [
    'image', 'plate_num', 'plate_type', 'bbox', 'quad',
    'is_vehicle', 'is_synthetic', 'source', 'license', 'conditions'
]

REQUIRED_PATHS = [
    'images/real',
    'images/synthetic',
    'labels',
    'meta.csv',
    'generator',
    'README.md',
    'LICENSE'
]


class DatasetValidator:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats = {
            'total_rows': 0,
            'valid_rows': 0,
            'real_images': 0,
            'synthetic_images': 0,
            'plates_by_type': {'type1': 0, 'type1a': 0, 'type1b': 0, 'other': 0},
            'unique_plates_by_type': {'type1': set(), 'type1a': set(), 'type1b': set(), 'other': set()},
            'vehicles_count': 0,
            'non_vehicles_count': 0,
            'conditions_count': {c: 0 for c in VALID_CONDITIONS}
        }

    def validate_structure(self) -> bool:
        """Validate existence of mandatory files and directories."""
        all_ok = True
        for rel_path in REQUIRED_PATHS:
            full_path = self.dataset_dir / rel_path
            if not full_path.exists():
                self.errors.append(f"Missing required path: {rel_path}")
                all_ok = False
        return all_ok

    def validate_meta_csv(self) -> bool:
        """Validate meta.csv format, headers, and row content."""
        meta_path = self.dataset_dir / 'meta.csv'
        if not meta_path.exists():
            self.errors.append("meta.csv does not exist.")
            return False

        unique_images_real = set()
        unique_images_synth = set()

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                # Delimiter MUST be ';'
                reader = csv.DictReader(f, delimiter=';')
                header = reader.fieldnames or []

                # Check headers
                for field in REQUIRED_META_FIELDS:
                    if field not in header:
                        self.errors.append(f"meta.csv missing required column: '{field}'")
                        return False

                row_idx = 1
                for row in reader:
                    row_idx += 1
                    self.stats['total_rows'] += 1
                    row_errors = []

                    # 1. image
                    img_rel = row.get('image', '').strip()
                    if not img_rel:
                        row_errors.append("empty 'image'")
                    else:
                        img_path = self.dataset_dir / img_rel
                        if not img_path.exists():
                            row_errors.append(f"image file not found: {img_rel}")
                        else:
                            if 'real' in img_rel:
                                unique_images_real.add(img_rel)
                            else:
                                unique_images_synth.add(img_rel)

                    # 2. plate_num
                    plate_num = row.get('plate_num', '').strip().upper()
                    plate_type = row.get('plate_type', '').strip().lower()

                    if plate_type != 'other':
                        if not ALLOWED_CHARS_REGEX.match(plate_num):
                            row_errors.append(f"invalid characters in plate_num '{plate_num}'")
                        if len(plate_num) < 8 or len(plate_num) > 9:
                            if '#' not in plate_num:
                                row_errors.append(f"unexpected length ({len(plate_num)}) for plate_num '{plate_num}'")

                    # 3. plate_type
                    if plate_type not in VALID_PLATE_TYPES:
                        row_errors.append(f"invalid plate_type '{plate_type}'. Must be one of {VALID_PLATE_TYPES}")
                    else:
                        self.stats['plates_by_type'][plate_type] += 1
                        if '#' not in plate_num:
                            self.stats['unique_plates_by_type'][plate_type].add(plate_num)

                    # 4. bbox (x,y,w,h)
                    bbox_str = row.get('bbox', '').strip()
                    bbox_parts = bbox_str.split(',')
                    if len(bbox_parts) != 4:
                        row_errors.append(f"invalid bbox '{bbox_str}', expected 4 values (x,y,w,h)")
                    else:
                        try:
                            x, y, w, h = map(float, bbox_parts)
                            if w <= 0 or h <= 0:
                                row_errors.append(f"bbox w and h must be positive in '{bbox_str}'")
                        except ValueError:
                            row_errors.append(f"bbox contains non-numeric values in '{bbox_str}'")

                    # 5. quad (x1,y1,x2,y2,x3,y4,x5,x6,x7,x8)
                    quad_str = row.get('quad', '').strip()
                    quad_parts = quad_str.split(',')
                    if len(quad_parts) != 8:
                        row_errors.append(f"invalid quad '{quad_str}', expected 8 coordinates")
                    else:
                        try:
                            coords = list(map(float, quad_parts))
                        except ValueError:
                            row_errors.append(f"quad contains non-numeric values in '{quad_str}'")

                    # 6. is_vehicle
                    is_vehicle = str(row.get('is_vehicle', '')).strip()
                    if is_vehicle not in ('0', '1'):
                        row_errors.append(f"is_vehicle must be 0 or 1, got '{is_vehicle}'")
                    else:
                        if is_vehicle == '1':
                            self.stats['vehicles_count'] += 1
                        else:
                            self.stats['non_vehicles_count'] += 1

                    # 7. is_synthetic
                    is_synth = str(row.get('is_synthetic', '')).strip()
                    if is_synth not in ('0', '1'):
                        row_errors.append(f"is_synthetic must be 0 or 1, got '{is_synth}'")

                    # 8. source & license
                    source = row.get('source', '').strip()
                    lic = row.get('license', '').strip()
                    if not source:
                        row_errors.append("source cannot be empty")
                    if not lic:
                        row_errors.append("license cannot be empty")

                    # 9. conditions
                    conds = [c.strip().lower() for c in row.get('conditions', '').split(',') if c.strip()]
                    for c in conds:
                        if c in self.stats['conditions_count']:
                            self.stats['conditions_count'][c] += 1
                        else:
                            self.warnings.append(f"Row {row_idx}: unknown condition '{c}'")

                    # Label file check
                    if img_rel:
                        stem = Path(img_rel).stem
                        label_file = self.dataset_dir / 'labels' / f"{stem}.txt"
                        if not label_file.exists():
                            self.warnings.append(f"Missing label file: labels/{stem}.txt")

                    if row_errors:
                        self.errors.append(f"Row {row_idx} ({img_rel}): " + "; ".join(row_errors))
                    else:
                        self.stats['valid_rows'] += 1

            self.stats['real_images'] = len(unique_images_real)
            self.stats['synthetic_images'] = len(unique_images_synth)

        except Exception as e:
            self.errors.append(f"Error reading meta.csv: {e}")
            return False

        return len(self.errors) == 0

    def generate_report(self) -> str:
        """Generate human-readable validation summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("     DATASET VALIDATION REPORT (Volga-IT semi-final)")
        lines.append("=" * 60)
        lines.append(f"Dataset root: {self.dataset_dir}")
        lines.append(f"Status: {'PASSED' if not self.errors else 'FAILED'}")
        lines.append(f"Errors count: {len(self.errors)}")
        lines.append(f"Warnings count: {len(self.warnings)}")
        lines.append("-" * 60)
        lines.append("STATISTICS:")
        lines.append(f"  Total records in meta.csv: {self.stats['total_rows']}")
        lines.append(f"  Valid records:             {self.stats['valid_rows']}")
        lines.append(f"  Real images:               {self.stats['real_images']}")
        lines.append(f"  Synthetic images:          {self.stats['synthetic_images']}")
        lines.append(f"  Total images:              {self.stats['real_images'] + self.stats['synthetic_images']}")
        lines.append(f"  Vehicles (is_vehicle=1):   {self.stats['vehicles_count']}")
        lines.append(f"  Non-vehicles (is_veh=0):   {self.stats['non_vehicles_count']}")
        lines.append("")
        lines.append("Plates breakdown by type:")
        for t, count in self.stats['plates_by_type'].items():
            uniq = len(self.stats['unique_plates_by_type'][t])
            lines.append(f"  - {t:<8}: {count:>6} plates ({uniq:>6} unique)")

        lines.append("")
        lines.append("Conditions breakdown:")
        for c, count in self.stats['conditions_count'].items():
            lines.append(f"  - {c:<12}: {count:>6}")

        if self.errors:
            lines.append("-" * 60)
            lines.append("ERRORS (first 20):")
            for err in self.errors[:20]:
                lines.append(f"  [ERROR] {err}")
            if len(self.errors) > 20:
                lines.append(f"  ... and {len(self.errors) - 20} more errors.")

        if self.warnings:
            lines.append("-" * 60)
            lines.append("WARNINGS (first 10):")
            for w in self.warnings[:10]:
                lines.append(f"  [WARN] {w}")

        lines.append("=" * 60)
        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Volga-IT License Plate Dataset")
    parser.add_argument('--dataset-dir', default='dataset', help="Path to dataset directory")
    parser.add_argument('--report', default=None, help="Path to save text report")
    args = parser.parse_args()

    validator = DatasetValidator(args.dataset_dir)
    validator.validate_structure()
    validator.validate_meta_csv()
    report = validator.generate_report()
    print(report)

    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {args.report}")

    sys.exit(0 if not validator.errors else 1)


if __name__ == '__main__':
    main()
