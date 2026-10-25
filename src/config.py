"""
Configuration and constants for Volga-IT License Plate Recognition pipeline.
Conforms strictly to GOST R 50577-2018.
"""

import re
from typing import Dict, List, Set

# Allowed Latin characters that match Cyrillic glyphs in GOST R 50577-2018
ALLOWED_LETTERS: List[str] = ['A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X']
ALLOWED_LETTERS_SET: Set[str] = set(ALLOWED_LETTERS)

# Digits
ALLOWED_DIGITS: List[str] = [str(i) for i in range(10)]
ALLOWED_DIGITS_SET: Set[str] = set(ALLOWED_DIGITS)

# Total vocabulary for character recognition (including unknown placeholder '#')
VOCABULARY: List[str] = ALLOWED_LETTERS + ALLOWED_DIGITS + ['#']
CHAR_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(VOCABULARY)}
IDX_TO_CHAR: Dict[int, str] = {i: c for i, c in enumerate(VOCABULARY)}

# Mapping Cyrillic letters to Latin equivalents
CYRILLIC_TO_LATIN: Dict[str, str] = {
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H',
    'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X',
    'а': 'A', 'в': 'B', 'е': 'E', 'к': 'K', 'м': 'M', 'н': 'H',
    'о': 'O', 'р': 'P', 'с': 'C', 'т': 'T', 'у': 'Y', 'х': 'X',
    'a': 'A', 'b': 'B', 'e': 'E', 'k': 'K', 'm': 'M', 'h': 'H',
    'o': 'O', 'p': 'P', 'c': 'C', 't': 'T', 'y': 'Y', 'x': 'X'
}

# Target plate classes
TARGET_CLASSES: List[str] = ['type1', 'type1a', 'type1b', 'other']
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(TARGET_CLASSES)}

# Regular expression pattern for standard Russian license plate:
# 1 letter, 3 digits, 2 letters, 2-3 digits region (3-digit region starts with 1, 2, or 7)
PLATE_REGEX = re.compile(r'^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}(\d{2}|[127]\d{2})$')

# Pattern with optional '#' placeholders
PLATE_PATTERN_WITH_HASH = re.compile(
    r'^[ABEKMHOPCTYX#][\d#]{3}[ABEKMHOPCTYX#]{2}([\d#]{2}|[127#][\d#]{2})$'
)

# Common Russian region codes
VALID_REGION_CODES: Set[str] = {
    # 2-digit regions (01 to 99)
    f"{i:02d}" for i in range(1, 100)
} | {
    # 3-digit common regions
    "102", "113", "116", "121", "123", "124", "125", "126", "134", "136", "138",
    "142", "150", "152", "154", "159", "161", "163", "164", "173", "174", "177",
    "178", "186", "190", "196", "197", "198", "199", "216", "277", "702", "716",
    "725", "750", "761", "763", "774", "777", "790", "797", "799"
}

# Physical Dimensions (GOST R 50577-2018) in mm
GOST_DIMENSIONS = {
    'type1': {'width': 520, 'height': 112, 'aspect_ratio': 520 / 112},    # ~4.64
    'type1a': {'width': 290, 'height': 170, 'aspect_ratio': 290 / 170},   # ~1.71
    'type1b': {'width': 520, 'height': 112, 'aspect_ratio': 520 / 112},   # ~4.64
}

# Standard rectified image resolution for OCR
RECTIFIED_DIMS = {
    'type1': (256, 64),
    'type1a': (160, 100),
    'type1b': (256, 64),
    'other': (256, 64)
}

# Colors (RGB)
COLORS = {
    'plate_white': (245, 245, 245),
    'plate_yellow': (245, 195, 25),
    'text_black': (15, 15, 15),
    'flag_white': (255, 255, 255),
    'flag_blue': (0, 57, 166),
    'flag_red': (213, 43, 30)
}
