"""
Automated Test Suite for Volga-IT License Plate Recognition.
Verifies config, preprocessor, OCR forward pass, postprocessor syntax validation,
dataset integrity, and inference speed (< 100 ms).
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import cv2
import torch

from src.config import (
    ALLOWED_LETTERS,
    ALLOWED_DIGITS,
    TARGET_CLASSES,
    PLATE_REGEX,
    CYRILLIC_TO_LATIN
)
from src.preprocessor import rectify_plate, split_type1a_lines, enhance_plate
from src.postprocessor import validate_and_format_plate, normalize_characters, format_csv_row
from src.ocr import CRNN, LicensePlateOCR
from src.pipeline import LicensePlatePipeline
from dataset.validate_dataset import DatasetValidator


def test_config_rules():
    """Verify alphabet and regex constraints."""
    assert len(ALLOWED_LETTERS) == 12
    assert 'A' in ALLOWED_LETTERS and 'X' in ALLOWED_LETTERS
    assert len(TARGET_CLASSES) == 4
    assert set(TARGET_CLASSES) == {'type1', 'type1a', 'type1b', 'other'}

    # Regex tests
    assert PLATE_REGEX.match("A123BC777")
    assert PLATE_REGEX.match("T777TT77")
    assert PLATE_REGEX.match("K555OP25")
    assert not PLATE_REGEX.match("D123BC77")   # 'D' is not allowed in Russian plates
    assert not PLATE_REGEX.match("A123BC5555") # Too many digits


def test_normalization_and_cyrillic_mapping():
    """Verify Cyrillic to Latin mapping."""
    cyrillic_sample = "А123ВС77"
    normalized = normalize_characters(cyrillic_sample)
    assert normalized == "A123BC77"
    assert all(ord(c) < 128 for c in normalized)


def test_postprocessor_mask_and_hash():
    """Verify mask validation and '#' placeholder substitution."""
    valid_plate, conf = validate_and_format_plate("A123BC777")
    assert valid_plate == "A123BC777"
    assert conf >= 0.7

    # Bad characters substituted by '#'
    corrupted = "A1Z3BC77"  # 'Z' is not allowed
    formatted, _ = validate_and_format_plate(corrupted)
    assert formatted[2] == '#'

    # Low confidence character substituted by '#'
    plate_with_low_conf, _ = validate_and_format_plate(
        "A123BC777",
        char_confs=[0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        conf_threshold=0.4
    )
    assert plate_with_low_conf == "A1#3BC777"


def test_csv_formatting():
    """Verify exact CSV row format: image;plate_num;plate_type;confidence."""
    row = format_csv_row("path/to/img_0042.jpg", "A123BC716", "type1a", 0.93)
    assert row == "img_0042.jpg;A123BC716;type1a;0.93"
    parts = row.split(';')
    assert len(parts) == 4
    assert parts[0] == "img_0042.jpg"
    assert parts[1] == "A123BC716"
    assert parts[2] == "type1a"
    assert parts[3] == "0.93"


def test_preprocessor_rectification():
    """Verify rectification and type1a line split."""
    dummy_img = np.ones((500, 500, 3), dtype=np.uint8) * 128
    rect_1 = rectify_plate(dummy_img, bbox=(50, 50, 200, 50), plate_type='type1')
    assert rect_1.shape == (64, 256, 3)

    rect_1a = rectify_plate(dummy_img, bbox=(50, 50, 150, 100), plate_type='type1a')
    assert rect_1a.shape == (100, 160, 3)

    top, bot = split_type1a_lines(rect_1a)
    assert top.shape == (48, 256, 3)
    assert bot.shape == (48, 256, 3)


def test_crnn_forward_pass():
    """Verify CRNN architecture forward pass."""
    model = CRNN(num_classes=24)
    x = torch.randn(2, 3, 48, 256)
    logits = model(x)
    # Output: (T, B, C)
    assert logits.dim() == 3
    assert logits.size(1) == 2
    assert logits.size(2) == 24


def test_dataset_validator():
    """Run validator on the dataset."""
    validator = DatasetValidator("dataset")
    assert validator.validate_structure()
    assert validator.validate_meta_csv()
    assert len(validator.errors) == 0


def test_inference_latency_benchmark():
    """Verify pipeline latency is below 100 ms limit."""
    pipeline = LicensePlatePipeline(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy test image
    dummy_scene = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    test_path = "tests/test_latency.jpg"
    cv2.imwrite(test_path, dummy_scene)

    # Warmup
    _ = pipeline.process_image(test_path)

    # Benchmark 10 runs
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = pipeline.process_image(test_path)
        times.append(time.perf_counter() - t0)

    avg_latency_ms = (sum(times) / len(times)) * 1000.0
    print(f"\nAverage inference latency: {avg_latency_ms:.2f} ms")
    if os.path.exists(test_path):
        os.remove(test_path)

    assert avg_latency_ms < 100.0, f"Inference exceeded 100 ms: {avg_latency_ms:.2f} ms"
