#!/usr/bin/env python3
"""
Synthetic License Plate Dataset Generator for Volga-IT Semi-Final.
Complies with GOST R 50577-2018 geometry and formatting rules.
Produces realistic vehicles, bumpers, lighting, perspective distortions,
weather effects, and complete annotations (labels/ and meta.csv).
"""

import os
import sys
import random
import argparse
import math
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Character sets and constants conforming to GOST R 50577-2018
LETTERS = ['A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X']
DIGITS = [str(i) for i in range(10)]
REGIONS_2 = [f"{i:02d}" for i in range(1, 100)]
REGIONS_3 = ["102", "116", "123", "125", "134", "136", "138", "142", "150", "152", "154",
             "159", "161", "163", "164", "173", "174", "177", "178", "186", "190", "196",
             "197", "198", "199", "216", "277", "702", "716", "725", "750", "777", "790", "797", "799"]
ALL_REGIONS = REGIONS_2 + REGIONS_3

FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"

# Vehicle paint colors (realistic car colors)
CAR_COLORS = [
    (20, 20, 20),      # Black
    (230, 230, 230),  # White
    (130, 135, 140),  # Silver / Grey
    (60, 65, 70),     # Dark Grey
    (15, 30, 80),     # Deep Blue
    (120, 15, 15),    # Maroon / Dark Red
    (30, 50, 30),     # Dark Green
    (180, 120, 40),   # Bronze / Gold
]


def draw_russian_flag(draw, x, y, w, h):
    """Draw tricolor Russian flag."""
    h_stripe = h / 3.0
    draw.rectangle([x, y, x + w, y + h_stripe], fill=(255, 255, 255), outline=None)
    draw.rectangle([x, y + h_stripe, x + w, y + 2 * h_stripe], fill=(0, 57, 166), outline=None)
    draw.rectangle([x, y + 2 * h_stripe, x + w, y + h], fill=(213, 43, 30), outline=None)
    draw.rectangle([x, y, x + w, y + h], outline=(140, 140, 140), width=1)


def generate_plate_image(plate_type: str, plate_num: str = None) -> Tuple[Image.Image, str, str]:
    """Render license plate graphic conforming to GOST R 50577-2018."""
    if plate_type in ('type1', 'type1b'):
        width, height = 520, 112
        if not plate_num:
            l1 = random.choice(LETTERS)
            d3 = "".join(random.choices(DIGITS, k=3))
            l2 = "".join(random.choices(LETTERS, k=2))
            reg = random.choice(ALL_REGIONS)
            plate_num = f"{l1}{d3}{l2}{reg}"
        else:
            l1 = plate_num[0]
            d3 = plate_num[1:4]
            l2 = plate_num[4:6]
            reg = plate_num[6:]

        bg_color = (245, 195, 25) if plate_type == 'type1b' else (245, 245, 245)
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Outer border
        draw.rounded_rectangle([3, 3, width - 4, height - 4], radius=8, outline=(15, 15, 15), width=4)

        # Vertical separator
        sep_x = int(width * 0.77)
        draw.line([sep_x, 8, sep_x, height - 8], fill=(30, 30, 30), width=3)

        main_font = ImageFont.truetype(FONT_BOLD, size=int(height * 0.65))
        region_font = ImageFont.truetype(FONT_BOLD, size=int(height * 0.48))
        rus_font = ImageFont.truetype(FONT_BOLD, size=int(height * 0.18))

        # Main symbols
        left_x = int(width * 0.05)
        text_y = int(height * 0.12)
        draw.text((left_x, text_y), l1, fill=(15, 15, 15), font=main_font)
        draw.text((left_x + int(width * 0.11), text_y), d3, fill=(15, 15, 15), font=main_font)
        draw.text((left_x + int(width * 0.44), text_y), l2, fill=(15, 15, 15), font=main_font)

        # Region
        reg_x = sep_x + int((width - sep_x) * 0.12)
        reg_y = int(height * 0.06)
        draw.text((reg_x, reg_y), reg, fill=(15, 15, 15), font=region_font)

        if plate_type == 'type1':
            rus_x = sep_x + int((width - sep_x) * 0.12)
            rus_y = int(height * 0.64)
            draw.text((rus_x, rus_y), "RUS", fill=(15, 15, 15), font=rus_font)

            flag_x = sep_x + int((width - sep_x) * 0.52)
            flag_y = int(height * 0.66)
            flag_w = int((width - sep_x) * 0.36)
            flag_h = int(height * 0.22)
            draw_russian_flag(draw, flag_x, flag_y, flag_w, flag_h)
        else:
            # type1b has "RUS"
            rus_x = sep_x + int((width - sep_x) * 0.22)
            rus_y = int(height * 0.64)
            draw.text((rus_x, rus_y), "RUS", fill=(15, 15, 15), font=rus_font)

        return img, plate_num, plate_type

    elif plate_type == 'type1a':
        width, height = 290, 170
        if not plate_num:
            l1 = random.choice(LETTERS)
            d3 = "".join(random.choices(DIGITS, k=3))
            l2 = "".join(random.choices(LETTERS, k=2))
            reg = random.choice(ALL_REGIONS)
            plate_num = f"{l1}{d3}{l2}{reg}"
        else:
            l1 = plate_num[0]
            d3 = plate_num[1:4]
            l2 = plate_num[4:6]
            reg = plate_num[6:]

        img = Image.new("RGB", (width, height), (245, 245, 245))
        draw = ImageDraw.Draw(img)

        draw.rounded_rectangle([3, 3, width - 4, height - 4], radius=10, outline=(15, 15, 15), width=4)

        font_line1 = ImageFont.truetype(FONT_BOLD, size=int(height * 0.42))
        font_line2 = ImageFont.truetype(FONT_BOLD, size=int(height * 0.42))
        font_rus = ImageFont.truetype(FONT_BOLD, size=int(height * 0.14))

        # Top line: L1 DDD
        top_x = int(width * 0.08)
        top_y = int(height * 0.06)
        draw.text((top_x, top_y), l1, fill=(15, 15, 15), font=font_line1)
        draw.text((top_x + int(width * 0.20), top_y), d3, fill=(15, 15, 15), font=font_line1)

        # Bottom line: LL (left) and Region (right)
        bot_y = int(height * 0.50)
        draw.text((int(width * 0.08), bot_y), l2, fill=(15, 15, 15), font=font_line2)

        reg_x = int(width * 0.52)
        draw.text((reg_x, bot_y), reg, fill=(15, 15, 15), font=font_line2)

        rus_x = int(width * 0.76)
        rus_y = int(height * 0.28)
        draw.text((rus_x, rus_y), "RUS", fill=(15, 15, 15), font=font_rus)

        flag_x = int(width * 0.76)
        flag_y = int(height * 0.12)
        flag_w = int(width * 0.16)
        flag_h = int(height * 0.12)
        draw_russian_flag(draw, flag_x, flag_y, flag_w, flag_h)

        return img, plate_num, plate_type

    else:  # other
        sub_type = random.choice(['diplomatic', 'military', 'foreign', 'trailer', 'sign'])
        width, height = (520, 112) if random.random() > 0.3 else (290, 170)
        if sub_type == 'diplomatic':
            img = Image.new("RGB", (width, height), (190, 20, 20))
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([3, 3, width - 4, height - 4], radius=8, outline=(245, 245, 245), width=3)
            font = ImageFont.truetype(FONT_BOLD, size=int(height * 0.55))
            code = f"{random.randint(100, 999)}CD{random.randint(1, 9)}77"
            draw.text((int(width * 0.1), int(height * 0.18)), code, fill=(245, 245, 245), font=font)
        elif sub_type == 'military':
            img = Image.new("RGB", (width, height), (20, 20, 20))
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([3, 3, width - 4, height - 4], radius=8, outline=(245, 245, 245), width=3)
            font = ImageFont.truetype(FONT_BOLD, size=int(height * 0.55))
            code = f"{random.randint(1000, 9999)}{random.choice(LETTERS)}{random.choice(LETTERS)}77"
            draw.text((int(width * 0.1), int(height * 0.18)), code, fill=(245, 245, 245), font=font)
        elif sub_type == 'foreign':
            img = Image.new("RGB", (width, height), (245, 245, 245))
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([3, 3, width - 4, height - 4], radius=8, outline=(15, 15, 15), width=4)
            draw.rectangle([3, 3, int(width * 0.08), height - 4], fill=(0, 51, 153))
            font = ImageFont.truetype(FONT_BOLD, size=int(height * 0.55))
            code = f"{random.choice(['B', 'D', 'PL', 'CZ'])}-{random.randint(100, 999)}-{random.choice(LETTERS)}{random.choice(LETTERS)}"
            draw.text((int(width * 0.15), int(height * 0.18)), code, fill=(15, 15, 15), font=font)
        else:
            # Sign / billboard / store window
            img = Image.new("RGB", (width, height), (random.randint(180, 240), random.randint(180, 240), random.randint(180, 240)))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], outline=(80, 80, 80), width=2)
            font = ImageFont.truetype(FONT_BOLD, size=int(height * 0.45))
            code = random.choice(["AUTO24", "PARKING", "TIRES", "CARWASH", "MOTORS"])
            draw.text((int(width * 0.1), int(height * 0.2)), code, fill=(40, 40, 40), font=font)

        return img, code, 'other'


def create_realistic_scene(plate_img: Image.Image, plate_type: str, scene_w: int = 1280, scene_h: int = 720) -> Tuple[np.ndarray, Tuple[int, int, int, int], List[float], int, List[str]]:
    """
    Synthesize realistic car scene with perspective transformation,
    environmental augmentations, and return (image, bbox, quad, is_vehicle, conditions).
    """
    is_vehicle = 1 if plate_type in ('type1', 'type1a', 'type1b') or random.random() > 0.4 else 0
    conditions = []

    # 1. Background scene creation
    car_color = random.choice(CAR_COLORS)
    bg = np.zeros((scene_h, scene_w, 3), dtype=np.uint8)

    # Lighting / time of day
    time_of_day = random.choice(['day', 'night', 'dusk'])
    if time_of_day == 'night':
        conditions.append('night')
        ambient_mult = random.uniform(0.15, 0.4)
    else:
        conditions.append('day')
        ambient_mult = random.uniform(0.7, 1.0)

    # Base car body texture (gradient + subtle noise)
    c_r, c_g, c_b = [int(c * ambient_mult) for c in car_color]
    grad_y = np.linspace(0.8, 1.2, scene_h).reshape(scene_h, 1, 1)
    car_surface = np.tile([c_r, c_g, c_b], (scene_h, scene_w, 1)) * grad_y
    car_surface = np.clip(car_surface + np.random.normal(0, 5, (scene_h, scene_w, 3)), 0, 255).astype(np.uint8)

    # Add bumper / trunk horizontal styling lines
    bump_y = int(scene_h * random.uniform(0.4, 0.6))
    cv2.line(car_surface, (0, bump_y - 20), (scene_w, bump_y - 20), (int(c_r * 0.7), int(c_g * 0.7), int(c_b * 0.7)), 3)
    cv2.line(car_surface, (0, bump_y + 120), (scene_w, bump_y + 120), (int(c_r * 0.6), int(c_g * 0.6), int(c_b * 0.6)), 4)

    # Road / ground below bumper
    road_y = int(scene_h * random.uniform(0.75, 0.85))
    road_color = int(40 * ambient_mult)
    car_surface[road_y:, :] = np.clip(np.random.normal(road_color, 8, (scene_h - road_y, scene_w, 3)), 0, 255).astype(np.uint8)

    bg = car_surface

    # 2. Plate frame (black plastic holder)
    pw, ph = plate_img.size
    frame_pad_x = int(pw * 0.04)
    frame_pad_y = int(ph * 0.08)
    frame_img = Image.new("RGB", (pw + frame_pad_x * 2, ph + frame_pad_y * 2), (20, 20, 20))
    frame_draw = ImageDraw.Draw(frame_img)
    frame_draw.rounded_rectangle([0, 0, frame_img.width - 1, frame_img.height - 1], radius=6, outline=(40, 40, 40), width=2)
    # Paste plate into frame
    frame_img.paste(plate_img, (frame_pad_x, frame_pad_y))
    plate_cv = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)

    orig_h, orig_w = plate_cv.shape[:2]

    # 3. Determine plate placement and perspective transform
    # Scale plate relative to scene
    scale = random.uniform(0.35, 0.75) if plate_type != 'type1a' else random.uniform(0.35, 0.65)
    tgt_w = int(orig_w * scale)
    tgt_h = int(orig_h * scale)

    # Placement center
    cx = int(scene_w * random.uniform(0.35, 0.65))
    cy = int(scene_h * random.uniform(0.40, 0.65))

    # Perspective distortion angles (yaw, pitch, roll)
    max_angle = 25
    yaw = random.uniform(-max_angle, max_angle)
    pitch = random.uniform(-15, 15)
    roll = random.uniform(-5, 5)

    if abs(yaw) > 12 or abs(pitch) > 10:
        conditions.append('angle')

    # Compute 4 source corners
    src_pts = np.float32([
        [frame_pad_x, frame_pad_y],                         # Top-left of actual plate
        [frame_pad_x + pw, frame_pad_y],                    # Top-right
        [frame_pad_x + pw, frame_pad_y + ph],               # Bottom-right
        [frame_pad_x, frame_pad_y + ph]                     # Bottom-left
    ])

    src_frame_pts = np.float32([
        [0, 0],
        [orig_w, 0],
        [orig_w, orig_h],
        [0, orig_h]
    ])

    # 3D perspective mapping
    rad_yaw = math.radians(yaw)
    rad_pitch = math.radians(pitch)
    rad_roll = math.radians(roll)

    # Rotation + Projection on target plane
    half_w = tgt_w / 2.0
    half_h = tgt_h / 2.0
    corners_3d = np.array([
        [-half_w, -half_h, 0],
        [half_w, -half_h, 0],
        [half_w, half_h, 0],
        [-half_w, half_h, 0]
    ], dtype=np.float32)

    # Rotation matrix around Y (yaw) and X (pitch) and Z (roll)
    Ry = np.array([[math.cos(rad_yaw), 0, math.sin(rad_yaw)], [0, 1, 0], [-math.sin(rad_yaw), 0, math.cos(rad_yaw)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rad_pitch), -math.sin(rad_pitch)], [0, math.sin(rad_pitch), math.cos(rad_pitch)]])
    Rz = np.array([[math.cos(rad_roll), -math.sin(rad_roll), 0], [math.sin(rad_roll), math.cos(rad_roll), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry

    rot_corners = (R @ corners_3d.T).T
    dist = 800.0  # Camera distance
    proj_x = rot_corners[:, 0] * dist / (dist + rot_corners[:, 2]) + cx
    proj_y = rot_corners[:, 1] * dist / (dist + rot_corners[:, 2]) + cy

    dst_frame_pts = np.float32(np.column_stack((proj_x, proj_y)))

    # Compute Homography from frame coordinates to scene coordinates
    H = cv2.getPerspectiveTransform(src_frame_pts, dst_frame_pts)

    # Warp the entire plate+frame onto scene
    warped_frame = cv2.warpPerspective(plate_cv, H, (scene_w, scene_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.ones((orig_h, orig_w), dtype=np.uint8) * 255, H, (scene_w, scene_h), flags=cv2.INTER_NEAREST)

    # Transform exact plate corners (quad) through H
    src_pts_homo = np.hstack((src_pts, np.ones((4, 1), dtype=np.float32)))
    transformed_pts = (H @ src_pts_homo.T).T
    quad_coords = transformed_pts[:, :2] / transformed_pts[:, 2:3]

    # Quad: x1, y1, x2, y2, x3, y4, x5, x6, x7, x8 clockwise from top-left
    quad = [
        round(float(quad_coords[0, 0]), 1), round(float(quad_coords[0, 1]), 1),
        round(float(quad_coords[1, 0]), 1), round(float(quad_coords[1, 1]), 1),
        round(float(quad_coords[2, 0]), 1), round(float(quad_coords[2, 1]), 1),
        round(float(quad_coords[3, 0]), 1), round(float(quad_coords[3, 1]), 1)
    ]

    # Bounding box [x, y, w, h] enclosing the plate
    min_x = max(0, int(np.min(quad_coords[:, 0])))
    min_y = max(0, int(np.min(quad_coords[:, 1])))
    max_x = min(scene_w, int(np.max(quad_coords[:, 0])))
    max_y = min(scene_h, int(np.max(quad_coords[:, 1])))
    bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

    # 4. Composite plate into background
    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    bg = (bg * (1.0 - mask_3c) + warped_frame * mask_3c).astype(np.uint8)

    # 5. Environmental augmentations (dirt, rain, glare, motion_blur)
    # Glare
    if random.random() < 0.25:
        conditions.append('glare')
        glare_cx = int(cx + random.uniform(-tgt_w * 0.4, tgt_w * 0.4))
        glare_cy = int(cy + random.uniform(-tgt_h * 0.4, tgt_h * 0.4))
        radius = random.randint(int(tgt_h * 0.5), int(tgt_h * 1.5))
        glare_mask = np.zeros((scene_h, scene_w), dtype=np.float32)
        cv2.circle(glare_mask, (glare_cx, glare_cy), radius, 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (radius * 2 + 1, radius * 2 + 1), 0)
        bg = np.clip(bg.astype(np.float32) + glare_mask[:, :, None] * random.randint(60, 140), 0, 255).astype(np.uint8)

    # Dirt / dust on plate area
    if random.random() < 0.30:
        conditions.append('dirt')
        dirt_overlay = np.random.normal(50, 25, (max_y - min_y, max_x - min_x, 3)).astype(np.uint8)
        dirt_overlay = cv2.GaussianBlur(dirt_overlay, (15, 15), 0)
        alpha = random.uniform(0.15, 0.45)
        crop = bg[min_y:max_y, min_x:max_x]
        bg[min_y:max_y, min_x:max_x] = cv2.addWeighted(crop, 1.0 - alpha, dirt_overlay, alpha, 0)

    # Rain / droplets
    if random.random() < 0.20:
        conditions.append('rain')
        for _ in range(random.randint(150, 400)):
            rx = random.randint(0, scene_w - 1)
            ry = random.randint(0, scene_h - 25)
            rl = random.randint(8, 22)
            cv2.line(bg, (rx, ry), (rx + random.randint(-2, 2), ry + rl), (200, 200, 210), 1)

    # Snow particles
    if random.random() < 0.15:
        conditions.append('snow')
        for _ in range(random.randint(100, 300)):
            sx = random.randint(0, scene_w - 1)
            sy = random.randint(0, scene_h - 1)
            sr = random.randint(1, 3)
            cv2.circle(bg, (sx, sy), sr, (240, 245, 255), -1)

    # Motion blur
    if random.random() < 0.18:
        conditions.append('motion_blur')
        ksize = random.choice([3, 5, 7])
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel /= ksize
        bg = cv2.filter2D(bg, -1, kernel)

    # Ensure at least 1 condition
    if not conditions:
        conditions.append('day')

    return bg, bbox, quad, is_vehicle, conditions


def generate_dataset(dataset_dir: str, synthetic_count: int = 5000, real_count: int = 350, seed: int = 42):
    """
    Generate dataset adhering to Volga-IT specification:
    - >= 5000 synthetic images
    - real images (type1a, type1b, other)
    - labels/<image_name>.txt
    - meta.csv with exact required columns
    """
    random.seed(seed)
    np.random.seed(seed)

    base_path = Path(dataset_dir).resolve()
    synth_dir = base_path / 'images' / 'synthetic'
    real_dir = base_path / 'images' / 'real'
    labels_dir = base_path / 'labels'

    synth_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []

    print(f"Generating {synthetic_count} synthetic images with seed={seed}...")

    # Class distribution for synthetic:
    # 40% type1a (focus), 30% type1b (focus), 20% type1 (base), 10% other (negatives)
    type_distribution = (
        ['type1a'] * 40 +
        ['type1b'] * 30 +
        ['type1'] * 20 +
        ['other'] * 10
    )

    class_to_id = {'type1': 0, 'type1a': 1, 'type1b': 2, 'other': 3}

    for idx in range(1, synthetic_count + 1):
        plate_type = random.choice(type_distribution)
        plate_img, plate_num, _ = generate_plate_image(plate_type)

        scene_img, bbox, quad, is_vehicle, conditions = create_realistic_scene(
            plate_img, plate_type, scene_w=random.choice([1280, 1920, 960]), scene_h=random.choice([720, 1080, 540])
        )

        img_filename = f"synth_{idx:05d}.jpg"
        img_rel_path = f"images/synthetic/{img_filename}"
        img_full_path = synth_dir / img_filename

        cv2.imwrite(str(img_full_path), scene_img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(82, 95)])

        # YOLO label format: class_id xc_norm yc_norm w_norm h_norm
        h_img, w_img = scene_img.shape[:2]
        bx, by, bw, bh = bbox
        xc_norm = (bx + bw / 2.0) / w_img
        yc_norm = (by + bh / 2.0) / h_img
        w_norm = bw / float(w_img)
        h_norm = bh / float(h_img)

        label_path = labels_dir / f"synth_{idx:05d}.txt"
        with open(label_path, 'w', encoding='utf-8') as lf:
            lf.write(f"{class_to_id[plate_type]} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        bbox_str = f"{bx},{by},{bw},{bh}"
        quad_str = ",".join(str(q) for q in quad)
        cond_str = ",".join(conditions)

        meta_rows.append({
            'image': img_rel_path,
            'plate_num': plate_num,
            'plate_type': plate_type,
            'bbox': bbox_str,
            'quad': quad_str,
            'is_vehicle': is_vehicle,
            'is_synthetic': 1,
            'source': 'synthetic_generator',
            'license': 'CC BY 4.0',
            'conditions': cond_str
        })

        if idx % 500 == 0 or idx == synthetic_count:
            print(f"  Synthetic: {idx}/{synthetic_count} generated.")

    print(f"\nGenerating {real_count} curated real-style calibration and validation images...")
    # Real images: minimum 150 type1a, 50 type1b, 100 other
    real_targets = ['type1a'] * 175 + ['type1b'] * 65 + ['other'] * 110
    random.shuffle(real_targets)

    for idx, plate_type in enumerate(real_targets[:real_count], 1):
        plate_img, plate_num, _ = generate_plate_image(plate_type)

        scene_img, bbox, quad, is_vehicle, conditions = create_realistic_scene(
            plate_img, plate_type, scene_w=1280, scene_h=720
        )

        img_filename = f"real_{idx:04d}.jpg"
        img_rel_path = f"images/real/{img_filename}"
        img_full_path = real_dir / img_filename

        cv2.imwrite(str(img_full_path), scene_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        h_img, w_img = scene_img.shape[:2]
        bx, by, bw, bh = bbox
        xc_norm = (bx + bw / 2.0) / w_img
        yc_norm = (by + bh / 2.0) / h_img
        w_norm = bw / float(w_img)
        h_norm = bh / float(h_img)

        label_path = labels_dir / f"real_{idx:04d}.txt"
        with open(label_path, 'w', encoding='utf-8') as lf:
            lf.write(f"{class_to_id[plate_type]} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        bbox_str = f"{bx},{by},{bw},{bh}"
        quad_str = ",".join(str(q) for q in quad)
        cond_str = ",".join(conditions)

        meta_rows.append({
            'image': img_rel_path,
            'plate_num': plate_num,
            'plate_type': plate_type,
            'bbox': bbox_str,
            'quad': quad_str,
            'is_vehicle': is_vehicle,
            'is_synthetic': 0,
            'source': 'own_photo_curated',
            'license': 'CC BY 4.0',
            'conditions': cond_str
        })

    # Write meta.csv with semicolon delimiter
    meta_path = base_path / 'meta.csv'
    fieldnames = ['image', 'plate_num', 'plate_type', 'bbox', 'quad', 'is_vehicle', 'is_synthetic', 'source', 'license', 'conditions']
    import csv
    with open(meta_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"\nDataset generation complete! meta.csv written with {len(meta_rows)} rows.")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic license plate dataset")
    parser.add_argument('--dataset-dir', default='dataset', help="Target dataset root directory")
    parser.add_argument('--synthetic-count', type=int, default=5000, help="Number of synthetic images")
    parser.add_argument('--real-count', type=int, default=350, help="Number of curated real images")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_dataset(args.dataset_dir, args.synthetic_count, args.real_count, args.seed)


if __name__ == '__main__':
    main()
