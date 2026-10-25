#!/usr/bin/env python3
"""
Training script for License Plate OCR (CRNN with CTC Loss).
Optimized for Russian plates: single-line (type1, type1b) and dual-line (type1a).
Combines real road dataset crops with multi-font domain-randomized synthetic generation.
Saves best weights to models/ocr_best.pt.
"""

import os
import sys
import random
import argparse
import glob
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any

from src.config import VOCABULARY, CHAR_TO_IDX, IDX_TO_CHAR, ALLOWED_LETTERS, ALLOWED_DIGITS
from src.ocr import CRNN
from dataset.generator.generate_synthetic import LETTERS, DIGITS, ALL_REGIONS, draw_russian_flag
from src.preprocessor import split_type1a_lines, rectify_plate

# Available fonts on Windows
FONTS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]
VALID_FONTS = [f for f in FONTS if os.path.exists(f)]
if not VALID_FONTS:
    VALID_FONTS = ["arial.ttf"]


def generate_synthetic_line(mode: str, font_path: str = None) -> Tuple[np.ndarray, str]:
    """
    Generate synthetic single-line image for OCR training:
    - 'full_type1': single-line white plate (A123BC77)
    - 'full_type1b': single-line yellow plate (A123BC77)
    - 'type1a_top': upper line of type1a (A123 + flag/RUS)
    - 'type1a_bot': lower line of type1a (BC77)
    """
    if font_path is None or not os.path.exists(font_path):
        font_path = random.choice(VALID_FONTS)

    w, h = 256, 48
    bg_color = (245, 195, 25) if mode == 'full_type1b' else (random.randint(235, 250), random.randint(235, 250), random.randint(235, 250))
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    text_color = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))

    if mode in ('full_type1', 'full_type1b'):
        l1 = random.choice(LETTERS)
        d3 = "".join(random.choices(DIGITS, k=3))
        l2 = "".join(random.choices(LETTERS, k=2))
        reg = random.choice(ALL_REGIONS)
        text = f"{l1}{d3}{l2}{reg}"

        f_size = random.randint(30, 36)
        font = ImageFont.truetype(font_path, size=f_size)
        start_x = random.randint(8, 16)
        y = random.randint(2, 8)
        
        # Draw series
        full_str = f"{l1} {d3} {l2}  {reg}"
        draw.text((start_x, y), full_str, fill=text_color, font=font)

    elif mode == 'type1a_top':
        l1 = random.choice(LETTERS)
        d3 = "".join(random.choices(DIGITS, k=3))
        text = f"{l1}{d3}"

        f_size = random.randint(32, 38)
        font = ImageFont.truetype(font_path, size=f_size)
        start_x = random.randint(12, 25)
        y = random.randint(2, 6)

        # Draw L DDD
        draw.text((start_x, y), f"{l1} {d3}", fill=text_color, font=font)

        # Sometimes draw flag and RUS on the right
        if random.random() < 0.8:
            flag_x = random.randint(185, 205)
            flag_y = random.randint(6, 12)
            draw_russian_flag(draw, flag_x, flag_y, 35, 18)
            rus_font = ImageFont.truetype(font_path, size=10)
            draw.text((flag_x + 5, flag_y + 20), "RUS", fill=text_color, font=rus_font)

    else:  # type1a_bot
        l2 = "".join(random.choices(LETTERS, k=2))
        reg = random.choice(ALL_REGIONS)
        text = f"{l2}{reg}"

        f_size = random.randint(32, 38)
        font = ImageFont.truetype(font_path, size=f_size)
        start_x = random.randint(15, 30)
        y = random.randint(2, 6)

        draw.text((start_x, y), l2, fill=text_color, font=font)
        reg_x = random.randint(120, 160)
        draw.text((reg_x, y), reg, fill=text_color, font=font)

    crop_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return crop_bgr, text


class LicensePlateOCRDataset(Dataset):
    """
    Combined real road dataset + multi-font domain-randomized synthetic generator.
    """
    def __init__(self, size: int = 10000, is_val: bool = False, dataset_dir: str = "dataset"):
        self.size = size
        self.is_val = is_val
        self.real_crops: List[Tuple[np.ndarray, str]] = []

        # Load real crops from meta.csv
        meta_path = os.path.join(dataset_dir, "meta.csv")
        if os.path.exists(meta_path):
            try:
                df = pd.read_csv(meta_path, sep=';')
                real_df = df[(df['is_synthetic'] == 0) & (~df['plate_num'].str.contains('#')) & (df['plate_type'].isin(['type1', 'type1a', 'type1b']))]
                print(f"Loading {len(real_df)} real plates for OCR training/validation...")
                for idx, row in real_df.iterrows():
                    img_path = os.path.join(dataset_dir, row['image'])
                    if not os.path.exists(img_path):
                        continue
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    bbox = tuple(map(int, row['bbox'].split(',')))
                    quad = list(map(float, row['quad'].split(','))) if pd.notna(row['quad']) else None
                    ptype = row['plate_type']
                    pnum = row['plate_num']

                    rect = rectify_plate(img, bbox, quad, ptype)
                    if rect.size == 0:
                        continue

                    if ptype == 'type1a' and len(pnum) >= 8:
                        top, bot = split_type1a_lines(rect)
                        self.real_crops.append((top, pnum[:4]))
                        self.real_crops.append((bot, pnum[4:]))
                    else:
                        resized = cv2.resize(rect, (256, 48), interpolation=cv2.INTER_LINEAR)
                        self.real_crops.append((resized, pnum))
                print(f"Successfully cached {len(self.real_crops)} real line crops.")
            except Exception as e:
                print(f"Warning: Failed to load real crops: {e}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 45% real crop (if available), 55% multi-font synthetic
        use_real = (len(self.real_crops) > 0) and (random.random() < 0.45)

        if use_real:
            crop_bgr, text = random.choice(self.real_crops)
            crop_bgr = crop_bgr.copy()
        else:
            mode = random.choices(
                ['full_type1', 'full_type1b', 'type1a_top', 'type1a_bot'],
                weights=[0.30, 0.20, 0.25, 0.25]
            )[0]
            crop_bgr, text = generate_synthetic_line(mode)

        # Target dimensions: 256 x 48
        if crop_bgr.shape[0] != 48 or crop_bgr.shape[1] != 256:
            crop_bgr = cv2.resize(crop_bgr, (256, 48), interpolation=cv2.INTER_LINEAR)

        # Photometric & geometric augmentations
        if not self.is_val:
            # Random brightness and contrast
            alpha = random.uniform(0.75, 1.25)
            beta = random.randint(-25, 25)
            crop_bgr = np.clip(crop_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            # Random slight Gaussian blur
            if random.random() < 0.30:
                k = random.choice([3, 5])
                crop_bgr = cv2.GaussianBlur(crop_bgr, (k, k), 0)

            # Random noise
            if random.random() < 0.25:
                noise = np.random.normal(0, random.uniform(5, 15), crop_bgr.shape).astype(np.float32)
                crop_bgr = np.clip(crop_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # Random morphological erosion or dilation
            if random.random() < 0.20:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                if random.random() < 0.5:
                    crop_bgr = cv2.erode(crop_bgr, kernel, iterations=1)
                else:
                    crop_bgr = cv2.dilate(crop_bgr, kernel, iterations=1)

        # Normalize (ImageNet stats)
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (img_rgb - mean) / std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1))

        # Target label indices (1-indexed for CTC, 0 is blank)
        target = [CHAR_TO_IDX[c] + 1 for c in text if c in CHAR_TO_IDX]

        return tensor, torch.tensor(target, dtype=torch.long), len(target), text


def collate_fn(batch):
    tensors, targets, lengths, texts = zip(*batch)
    tensors = torch.stack(tensors, dim=0)
    flat_targets = torch.cat(targets, dim=0)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return tensors, flat_targets, target_lengths, texts


def train_ocr(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_weights: str = "models/ocr_best.pt"
):
    print("=" * 60)
    print(" Starting Plate OCR Training (CRNN + CTC)")
    print("=" * 60)
    print(f"Device:         {device}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {lr}")
    print(f"Output weights: {output_weights}")
    print("=" * 60)

    train_dataset = LicensePlateOCRDataset(size=8000, is_val=False)
    val_dataset = LicensePlateOCRDataset(size=1200, is_val=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = CRNN(num_classes=len(VOCABULARY) + 1).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            logits = model(images)  # (T, B, C)
            T, B, _ = logits.size()
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            log_probs = logits.log_softmax(2)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_sequences = 0
        total_sequences = 0

        with torch.no_grad():
            for images, targets, target_lengths, texts in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                logits = model(images)
                T, B, _ = logits.size()
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

                log_probs = logits.log_softmax(2)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()

                # Greedy accuracy check
                preds = logits.argmax(2).permute(1, 0).cpu().numpy()  # (B, T)
                for b_idx, pred_seq in enumerate(preds):
                    decoded = []
                    prev = 0
                    for p in pred_seq:
                        if p != 0 and p != prev:
                            if p - 1 < len(VOCABULARY):
                                decoded.append(VOCABULARY[p - 1])
                        prev = p
                    decoded_text = "".join(decoded)
                    if decoded_text == texts[b_idx]:
                        correct_sequences += 1
                    total_sequences += 1

        avg_val_loss = val_loss / len(val_loader)
        accuracy = (correct_sequences / max(1, total_sequences)) * 100.0

        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Seq Acc: {accuracy:.2f}%")

        if accuracy > best_acc or epoch == epochs:
            best_acc = accuracy
            os.makedirs(os.path.dirname(output_weights), exist_ok=True)
            torch.save(model.state_dict(), output_weights)
            print(f"  -> Saved new best OCR model to {output_weights} (Acc: {accuracy:.2f}%)")

    print(f"\nOCR Training Completed! Best Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Plate OCR Model")
    parser.add_argument('--epochs', type=int, default=15, help="Epochs")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--output', default="models/ocr_best.pt", help="Path to save weights")
    args = parser.parse_args()

    train_ocr(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_weights=args.output
    )
