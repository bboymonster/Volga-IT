#!/usr/bin/env python3
"""
Training script for License Plate OCR (CRNN with CTC Loss).
Optimized for Russian plates: single-line (type1, type1b) and dual-line (type1a).
Trains with online synthetic augmentation and saves best weights to models/ocr_best.pt.
"""

import os
import sys
import random
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from src.config import VOCABULARY, CHAR_TO_IDX, IDX_TO_CHAR
from src.ocr import CRNN
from dataset.generator.generate_synthetic import generate_plate_image, LETTERS, DIGITS, ALL_REGIONS
from src.preprocessor import split_type1a_lines, enhance_plate


class LicensePlateOCRDataset(Dataset):
    """
    Dynamic OCR dataset generating realistic license plate line crops.
    Generates full single lines (type1, type1b) and split lines (type1a top & bottom).
    """
    def __init__(self, size: int = 10000, is_val: bool = False):
        self.size = size
        self.is_val = is_val

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Sample mode: 40% full single line, 30% type1a top (L DDD), 30% type1a bottom (LL RRR)
        mode = random.choices(['full', 'type1a_top', 'type1a_bot'], weights=[0.4, 0.3, 0.3])[0]

        if mode == 'full':
            ptype = 'type1' if random.random() > 0.3 else 'type1b'
            plate_img, text, _ = generate_plate_image(ptype)
            crop_bgr = cv2.cvtColor(np.array(plate_img), cv2.COLOR_RGB2BGR)
        elif mode == 'type1a_top':
            l1 = random.choice(LETTERS)
            d3 = "".join(random.choices(DIGITS, k=3))
            text = f"{l1}{d3}"
            plate_img, full_text, _ = generate_plate_image('type1a', plate_num=f"{text}AA77")
            crop_bgr = cv2.cvtColor(np.array(plate_img), cv2.COLOR_RGB2BGR)
            top_line, _ = split_type1a_lines(crop_bgr)
            crop_bgr = top_line
        else:  # type1a_bot
            l2 = "".join(random.choices(LETTERS, k=2))
            reg = random.choice(ALL_REGIONS)
            text = f"{l2}{reg}"
            plate_img, full_text, _ = generate_plate_image('type1a', plate_num=f"A123{text}")
            crop_bgr = cv2.cvtColor(np.array(plate_img), cv2.COLOR_RGB2BGR)
            _, bot_line = split_type1a_lines(crop_bgr)
            crop_bgr = bot_line

        # Resize to standard OCR input: 256x48
        resized = cv2.resize(crop_bgr, (256, 48), interpolation=cv2.INTER_LINEAR)

        # Photometric augmentations if training
        if not self.is_val:
            # Random brightness & contrast
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-25, 25)
            resized = np.clip(resized.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            # Random slight blur
            if random.random() < 0.25:
                ksize = random.choice([3, 5])
                resized = cv2.GaussianBlur(resized, (ksize, ksize), 0)

            # Random noise
            if random.random() < 0.25:
                noise = np.random.normal(0, 10, resized.shape).astype(np.float32)
                resized = np.clip(resized.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Normalize (ImageNet stats)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
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
    val_dataset = LicensePlateOCRDataset(size=1000, is_val=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = CRNN(num_classes=len(VOCABULARY) + 1).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(output_weights), exist_ok=True)
            torch.save(model.state_dict(), output_weights)
            print(f"  -> Saved new best OCR model to {output_weights} (Acc: {accuracy:.2f}%)")

    print("\nOCR Training Completed!")


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
