"""
Detector Module for Volga-IT License Plate Recognition.
Wraps YOLO detection and contour-based quad extraction for real-time inference (<25 ms/image).
Categorizes detections into type1, type1a, type1b, or other.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from src.config import IDX_TO_CLASS, TARGET_CLASSES


class PlateDetector:
    """
    License plate detector and type classifier.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None

        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"Loaded YOLO detector from {model_path}")
            except Exception as e:
                print(f"Error loading YOLO weights from {model_path}: {e}")

    def extract_quad_from_crop(self, crop_bgr: np.ndarray, offset_x: int, offset_y: int) -> List[float]:
        """
        Estimate 4 corners of license plate using edge detection and contour approximation.
        Falls back to bounding box corners if quadrilateral is not found.
        """
        h, w = crop_bgr.shape[:2]
        fallback_quad = [
            float(offset_x), float(offset_y),
            float(offset_x + w), float(offset_y),
            float(offset_x + w), float(offset_y + h),
            float(offset_x), float(offset_y + h)
        ]

        if h < 15 or w < 30:
            return fallback_quad

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fallback_quad

        best_quad = None
        max_area = 0

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area and area > (w * h * 0.3):
                    max_area = area
                    best_quad = approx.reshape(4, 2)

        if best_quad is not None:
            # Reorder points clockwise from top-left
            pts = best_quad.astype(np.float32)
            s = pts.sum(axis=1)
            rect = np.zeros((4, 2), dtype=np.float32)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            return [
                float(rect[0, 0] + offset_x), float(rect[0, 1] + offset_y),
                float(rect[1, 0] + offset_x), float(rect[1, 1] + offset_y),
                float(rect[2, 0] + offset_x), float(rect[2, 1] + offset_y),
                float(rect[3, 0] + offset_x), float(rect[3, 1] + offset_y)
            ]

        return fallback_quad

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single image (BGR).
        Returns list of detections, each containing:
        - bbox: (x, y, w, h)
        - quad: [x1, y1, x2, y2, x3, y4, x5, x6, x7, x8]
        - plate_type: 'type1', 'type1a', 'type1b', 'other'
        - confidence: float in [0.0, 1.0]
        """
        if self.model is None:
            # Fallback heuristic detector for uninitialized model
            h, w = image.shape[:2]
            return []

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = xyxy
            bx = max(0, int(x1))
            by = max(0, int(y1))
            bw = max(1, int(x2 - x1))
            bh = max(1, int(y2 - y1))

            plate_type = IDX_TO_CLASS.get(cls_id, 'type1')
            crop = image[by:by + bh, bx:bx + bw]
            quad = self.extract_quad_from_crop(crop, bx, by)

            detections.append({
                'bbox': (bx, by, bw, bh),
                'quad': quad,
                'plate_type': plate_type,
                'confidence': round(conf, 2)
            })

        return detections
