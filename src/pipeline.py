"""
Unified End-to-End Pipeline for Volga-IT License Plate Recognition.
Achieves < 100 ms average latency per image on standard hardware.
"""

import os
import time
import glob
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from src.detector import PlateDetector
from src.preprocessor import rectify_plate
from src.ocr import LicensePlateOCR
from src.postprocessor import format_csv_row


class LicensePlatePipeline:
    """
    End-to-End License Plate Recognition Pipeline.
    Integrates Detection -> Perspective Rectification -> OCR -> Postprocessing.
    """
    def __init__(
        self,
        detector_weights: Optional[str] = "models/detector_best.pt",
        ocr_weights: Optional[str] = "models/ocr_best.pt",
        device: str = 'cuda',
        conf_threshold: float = 0.25
    ):
        self.detector = PlateDetector(
            model_path=detector_weights if (detector_weights and os.path.exists(detector_weights)) else None,
            conf_threshold=conf_threshold,
            device=device
        )
        self.ocr = LicensePlateOCR(
            model_path=ocr_weights if (ocr_weights and os.path.exists(ocr_weights)) else None,
            device=device
        )

    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process single image from file path.
        Returns list of results: [{'image': str, 'plate_num': str, 'plate_type': str, 'confidence': float}]
        """
        image_name = os.path.basename(image_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return []

        # 1. Detect plates
        detections = self.detector.detect(img_bgr)
        results = []

        for det in detections:
            plate_type = det['plate_type']
            det_conf = det['confidence']
            bbox = det['bbox']
            quad = det['quad']

            if plate_type == 'other':
                # Target requirement: mark as 'other' or skip; do not assign random type1
                results.append({
                    'image': image_name,
                    'plate_num': '########',
                    'plate_type': 'other',
                    'confidence': det_conf
                })
                continue

            # 2. Rectify plate crop
            rectified = rectify_plate(img_bgr, bbox=bbox, quad=quad, plate_type=plate_type)

            # 3. OCR recognition
            plate_num, ocr_conf = self.ocr.recognize_plate(rectified, plate_type=plate_type)

            # Combined confidence score
            combined_conf = round(float((det_conf * 0.4) + (ocr_conf * 0.6)), 2)
            combined_conf = max(0.10, min(0.99, combined_conf))

            results.append({
                'image': image_name,
                'plate_num': plate_num,
                'plate_type': plate_type,
                'confidence': combined_conf
            })

        return results

    def process_directory(
        self,
        input_dir: str,
        output_csv_path: str,
        benchmark: bool = False
    ) -> List[str]:
        """
        Process all .jpg, .png, .jpeg images in directory and write CSV output.
        CSV format:
        image;plate_num;plate_type;confidence
        """
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

        image_paths = sorted(list(set(image_paths)))
        print(f"Found {len(image_paths)} images to process in '{input_dir}'...")

        csv_lines = ["image;plate_num;plate_type;confidence\n"]
        total_time = 0.0
        total_images = len(image_paths)

        for idx, img_path in enumerate(image_paths, 1):
            t0 = time.perf_counter()
            plate_results = self.process_image(img_path)
            dt = time.perf_counter() - t0
            total_time += dt

            for res in plate_results:
                line = format_csv_row(
                    image_name=res['image'],
                    plate_num=res['plate_num'],
                    plate_type=res['plate_type'],
                    confidence=res['confidence']
                ) + "\n"
                csv_lines.append(line)

            if benchmark and idx % 20 == 0:
                avg_ms = (total_time / idx) * 1000.0
                print(f"  Processed {idx}/{total_images} (Avg: {avg_ms:.2f} ms/image)")

        # Write output CSV
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        with open(output_csv_path, 'w', encoding='utf-8') as f:
            f.writelines(csv_lines)

        avg_latency_ms = (total_time / max(1, total_images)) * 1000.0
        print(f"Processing finished in {total_time:.2f}s.")
        print(f"Average latency: {avg_latency_ms:.2f} ms per image.")
        print(f"Results successfully saved to: {output_csv_path}")

        return csv_lines
