"""
Postprocessor for Volga-IT license plate recognition pipeline.
Handles syntax validation, region validation, confidence calibration,
and '#' placeholder replacement for low-confidence characters.
"""

import re
from typing import Tuple, List, Dict, Optional
from src.config import (
    ALLOWED_LETTERS_SET,
    ALLOWED_DIGITS_SET,
    CYRILLIC_TO_LATIN,
    VALID_REGION_CODES,
    PLATE_REGEX
)


def normalize_characters(raw_text: str) -> str:
    """Map Cyrillic letters to Latin equivalents, replace unrecognized symbols with '#'."""
    result = []
    for ch in raw_text.strip().upper():
        if ch in CYRILLIC_TO_LATIN:
            result.append(CYRILLIC_TO_LATIN[ch])
        elif ch in ALLOWED_LETTERS_SET or ch in ALLOWED_DIGITS_SET or ch == '#':
            result.append(ch)
        else:
            result.append('#')
    return "".join(result)


def validate_and_format_plate(
    raw_plate: str,
    char_confs: Optional[List[float]] = None,
    conf_threshold: float = 0.40
) -> Tuple[str, float]:
    """
    Validate plate string against Russian plate mask:
    [Letter][3 Digits][2 Letters][2-3 Digits Region]
    Replaces ambiguous / low-confidence characters with '#'.
    Returns (formatted_plate, overall_confidence).
    """
    clean_text = normalize_characters(raw_plate)
    n = len(clean_text)

    if n < 8 or n > 9:
        # If length doesn't match standard plate, pad or truncate
        if n > 9:
            clean_text = clean_text[:9]
        elif n < 8:
            clean_text = clean_text.ljust(8, '#')
        n = len(clean_text)

    if char_confs is None or len(char_confs) != n:
        char_confs = [0.85] * n

    formatted_chars = list(clean_text)
    adjusted_confs = list(char_confs)

    # Position 0: Must be Letter
    if formatted_chars[0] not in ALLOWED_LETTERS_SET:
        formatted_chars[0] = '#'
        adjusted_confs[0] = min(adjusted_confs[0], 0.3)
    elif adjusted_confs[0] < conf_threshold:
        formatted_chars[0] = '#'

    # Positions 1, 2, 3: Must be Digits
    for i in (1, 2, 3):
        if formatted_chars[i] not in ALLOWED_DIGITS_SET:
            formatted_chars[i] = '#'
            adjusted_confs[i] = min(adjusted_confs[i], 0.3)
        elif adjusted_confs[i] < conf_threshold:
            formatted_chars[i] = '#'

    # Positions 4, 5: Must be Letters
    for i in (4, 5):
        if formatted_chars[i] not in ALLOWED_LETTERS_SET:
            formatted_chars[i] = '#'
            adjusted_confs[i] = min(adjusted_confs[i], 0.3)
        elif adjusted_confs[i] < conf_threshold:
            formatted_chars[i] = '#'

    # Positions 6..end: Region digits (2 or 3 digits)
    region_str = "".join(formatted_chars[6:])
    for idx, c in enumerate(region_str):
        global_idx = 6 + idx
        if c not in ALLOWED_DIGITS_SET:
            formatted_chars[global_idx] = '#'
            adjusted_confs[global_idx] = min(adjusted_confs[global_idx], 0.3)
        elif adjusted_confs[global_idx] < conf_threshold:
            formatted_chars[global_idx] = '#'

    # 3-digit region check: must start with 1, 2, or 7
    if len(region_str) == 3 and formatted_chars[6] != '#':
        if formatted_chars[6] not in ('1', '2', '7'):
            formatted_chars[6] = '#'

    final_plate = "".join(formatted_chars)
    overall_conf = round(float(sum(adjusted_confs) / len(adjusted_confs)), 2)
    overall_conf = max(0.10, min(0.99, overall_conf))

    return final_plate, overall_conf


def format_csv_row(image_name: str, plate_num: str, plate_type: str, confidence: float) -> str:
    """Format single result row: image;plate_num;plate_type;confidence."""
    # Ensure image is basename only
    import os
    base_name = os.path.basename(image_name)
    conf_str = f"{confidence:.2f}"
    return f"{base_name};{plate_num};{plate_type};{conf_str}"
