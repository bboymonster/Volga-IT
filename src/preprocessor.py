"""
Preprocessor for Volga-IT license plate recognition pipeline.
Handles perspective rectification, contrast adjustment, and plate line segmentation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.config import RECTIFIED_DIMS, GOST_DIMENSIONS


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order coordinates clockwise starting from top-left:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, quad_pts: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Apply perspective transform to warp quadrilateral into a target_w x target_h rectangle.
    """
    rect = order_points(quad_pts.astype(np.float32))
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
    return warped


def rectify_plate(image: np.ndarray, bbox: Tuple[int, int, int, int], quad: Optional[List[float]] = None, plate_type: str = 'type1') -> np.ndarray:
    """
    Rectify license plate image into canonical orientation and fixed dimensions.
    """
    target_w, target_h = RECTIFIED_DIMS.get(plate_type, (256, 64))

    if quad is not None and len(quad) == 8:
        quad_arr = np.array(quad, dtype=np.float32).reshape(4, 2)
        # Check if quad points are reasonable
        w1 = np.linalg.norm(quad_arr[0] - quad_arr[1])
        w2 = np.linalg.norm(quad_arr[3] - quad_arr[2])
        h1 = np.linalg.norm(quad_arr[0] - quad_arr[3])
        h2 = np.linalg.norm(quad_arr[1] - quad_arr[2])
        if min(w1, w2) > 10 and min(h1, h2) > 5:
            return four_point_transform(image, quad_arr, target_w, target_h)

    # Fallback to bbox crop
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def enhance_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE and contrast normalization to mitigate glares, shadows, and darkness.
    """
    if plate_bgr.size == 0:
        return plate_bgr

    # Convert to LAB color space
    lab = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    merged = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def split_type1a_lines(plate_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split square 2-line plate (Type 1A) into:
    - Top line: 1 letter + 3 digits
    - Bottom line: 2 letters + 2-3 digits region
    Uses adaptive horizontal projection profile to find the exact gap between rows.
    """
    h, w = plate_bgr.shape[:2]
    if h < 20 or w < 20:
        split_y = int(h * 0.50)
    else:
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        proj = inv.mean(axis=1)
        y_start, y_end = int(h * 0.38), int(h * 0.62)
        split_y = y_start + int(np.argmin(proj[y_start:y_end]))

    top_crop = plate_bgr[0:split_y, :]
    bot_crop = plate_bgr[split_y:h, :]

    # Normalize sizes for OCR
    top_resized = cv2.resize(top_crop, (256, 48), interpolation=cv2.INTER_LINEAR)
    bot_resized = cv2.resize(bot_crop, (256, 48), interpolation=cv2.INTER_LINEAR)

    return top_resized, bot_resized

