"""
OCR Module for Volga-IT License Plate Recognition.
Features a lightweight, high-speed CRNN (CNN + BiLSTM + CTC) architecture
optimized for real-time inference (<10 ms/plate) on standard GPUs and CPUs.
Supports single-line (type1, type1b) and dual-line (type1a) Russian license plates.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional
from src.config import VOCABULARY, CHAR_TO_IDX, IDX_TO_CHAR
from src.preprocessor import split_type1a_lines, enhance_plate
from src.postprocessor import validate_and_format_plate


class CRNN(nn.Module):
    """
    Lightweight CRNN (CNN + BiLSTM + Linear) for sequence character recognition.
    Input: (B, 3, 48, 256)
    Output: (T, B, num_classes)
    """
    def __init__(self, num_classes: int = len(VOCABULARY) + 1):
        super(CRNN, self).__init__()
        self.num_classes = num_classes  # Includes CTC blank at index 0

        # CNN Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 32, 24, 128)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 64, 12, 64)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (B, 128, 6, 64)

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (B, 256, 3, 64)

            # Block 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))   # (B, 256, 1, 64)
        )

        # Bidirectional LSTM Sequence Encoder
        self.rnn = nn.Sequential(
            nn.LSTM(256, 128, bidirectional=True, batch_first=True, num_layers=2),
        )

        # Linear Classifier to Character Vocabulary
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        conv = self.features(x)  # (B, 256, 1, W')
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0, 2, 1)  # (B, W', 256)

        # LSTM
        rnn_out, _ = self.rnn[0](conv)  # (B, W', 256)

        # Logits
        logits = self.fc(rnn_out)  # (B, W', num_classes)
        # CTC expects (T, B, C)
        logits = logits.permute(1, 0, 2)
        return logits


class LicensePlateOCR:
    """
    Inference and decoding engine for Russian license plates.
    """
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = CRNN(num_classes=len(VOCABULARY) + 1).to(self.device)
        self.blank_idx = 0

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded OCR weights from {model_path}")
        else:
            print("OCR model initialized with random weights (call train_ocr or load checkpoint).")

        self.model.eval()

    def preprocess_crop(self, crop_bgr: np.ndarray, target_w: int = 256, target_h: int = 48) -> torch.Tensor:
        """Resize, normalize, and convert crop to tensor (1, 3, H, W)."""
        resized = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (img_rgb - mean) / std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor

    def decode_greedy(self, logits: torch.Tensor) -> Tuple[str, List[float]]:
        """
        Greedy CTC decoder with character confidence estimation.
        logits: (T, 1, C)
        """
        probs = F.softmax(logits.squeeze(1), dim=-1)  # (T, C)
        max_probs, preds = torch.max(probs, dim=-1)   # (T,)

        preds = preds.cpu().numpy()
        max_probs = max_probs.detach().cpu().numpy()

        decoded_chars = []
        char_confs = []
        prev_idx = self.blank_idx

        for p_idx, prob in zip(preds, max_probs):
            if p_idx != self.blank_idx:
                if p_idx != prev_idx:
                    # Index in VOCABULARY is p_idx - 1 (since 0 is blank)
                    vocab_idx = p_idx - 1
                    if 0 <= vocab_idx < len(VOCABULARY):
                        decoded_chars.append(VOCABULARY[vocab_idx])
                        char_confs.append(float(prob))
            prev_idx = p_idx

        text = "".join(decoded_chars)
        return text, char_confs

    @torch.no_grad()
    def recognize_single_line(self, crop_bgr: np.ndarray) -> Tuple[str, List[float]]:
        """Recognize single-line crop."""
        inp = self.preprocess_crop(crop_bgr, target_w=256, target_h=48)
        logits = self.model(inp)
        return self.decode_greedy(logits)

    @torch.no_grad()
    def recognize_plate(self, plate_crop: np.ndarray, plate_type: str = 'type1') -> Tuple[str, float]:
        """
        End-to-end recognition of rectified plate crop.
        Handles single-line and dual-line (type1a) plates.
        Returns (plate_num, confidence).
        """
        enhanced = enhance_plate(plate_crop)

        if plate_type == 'type1a':
            # Split square plate into line 1 (L DDD) and line 2 (LL RRR)
            top_line, bot_line = split_type1a_lines(enhanced)
            text_top, confs_top = self.recognize_single_line(top_line)
            text_bot, confs_bot = self.recognize_single_line(bot_line)
            raw_text = text_top + text_bot
            raw_confs = confs_top + confs_bot
        else:
            raw_text, raw_confs = self.recognize_single_line(enhanced)

        # Validate against Russian license plate mask and replace low confidence with '#'
        formatted_plate, overall_conf = validate_and_format_plate(raw_text, raw_confs)
        return formatted_plate, overall_conf
