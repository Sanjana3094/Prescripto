# backend/pipeline.py
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import easyocr

from .dictionary_fix import clean_and_correct_prescription_lines
from .reminder_parser import medicines_from_lines

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model_cer_hybrid_balancedwork.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[pipeline] Using device: {device}")

# ------------------- load checkpoint -------------------
ckpt = torch.load(MODEL_PATH, map_location=device)

ctc_char_to_idx = ckpt["ctc_char_to_idx"]
ctc_idx_to_char = ckpt["ctc_idx_to_char"]
attn_char_to_idx = ckpt["attn_char_to_idx"]
attn_idx_to_char = ckpt["attn_idx_to_char"]

attn_pad_idx = 0
attn_sos_idx = 1
attn_eos_idx = 2


# ------------------- Model B (your CRNN+Attention) -------------------
class EncoderCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256,512,3,1,1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,1,1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            512, 256, num_layers=2, bidirectional=True,
            dropout=0.3, batch_first=True
        )

    def forward(self, x):
        feat = self.cnn(x)
        B,C,H,W = feat.size()
        feat = feat.permute(0,3,1,2).contiguous()
        feat = feat.view(B, W, C*H)
        out,_ = self.rnn(feat)
        return out


class AttentionDecoder(nn.Module):
    def __init__(self, encoder_dim=512, embed_dim=256, dec_hidden=512,
                 vocab_size=len(attn_char_to_idx), dropout=0.3):
        super().__init__()
        self.dec_hidden = dec_hidden
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=attn_pad_idx)
        self.attn_proj_enc = nn.Linear(encoder_dim, dec_hidden)
        self.attn_proj_dec = nn.Linear(dec_hidden, dec_hidden)
        # IMPORTANT: bias=False to match your trained weights
        self.attn_v = nn.Linear(dec_hidden, 1, bias=False)
        self.rnn_cell = nn.LSTMCell(embed_dim + encoder_dim, dec_hidden)
        self.out = nn.Linear(dec_hidden, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch: int):
        h = torch.zeros(batch, self.dec_hidden, device=device)
        c = torch.zeros(batch, self.dec_hidden, device=device)
        return h, c

    def compute_attention(self, encoder_out, h):
        proj_enc = self.attn_proj_enc(encoder_out)
        proj_dec = self.attn_proj_dec(h).unsqueeze(1)
        energy = torch.tanh(proj_enc + proj_dec)
        scores = self.attn_v(energy).squeeze(2)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_out).squeeze(1)
        return context

    def forward_step(self, encoder_out, prev_token, h, c):
        emb = self.embedding(prev_token)
        context = self.compute_attention(encoder_out, h)
        rnn_input = torch.cat([emb, context], dim=1)
        h_new, c_new = self.rnn_cell(rnn_input, (h, c))
        logits = self.out(self.dropout(h_new))
        return logits, (h_new, c_new)


encoder = EncoderCRNN().to(device)
decoder = AttentionDecoder().to(device)
encoder.load_state_dict(ckpt["encoder_state"])
decoder.load_state_dict(ckpt["decoder_state"])
encoder.eval()
decoder.eval()
print("[pipeline] Model B loaded successfully.")


# ------------------- beam search & word recognition -------------------
max_width = 192
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def beam_search(decoder, encoder_out, beam_width=10, max_len=25):
    B = encoder_out.size(0)
    results: list[str] = []
    for b in range(B):
        enc = encoder_out[b:b+1]
        h, c = decoder.init_hidden(1)
        start = torch.tensor([attn_sos_idx], device=device)
        beams = [(0.0, [start.item()], h, c)]

        for _ in range(max_len):
            new_beams = []
            for log_prob, seq, h_i, c_i in beams:
                last = torch.tensor([seq[-1]], device=device)
                if last.item() == attn_eos_idx:
                    new_beams.append((log_prob, seq, h_i, c_i))
                    continue
                logits, (h_new, c_new) = decoder.forward_step(enc, last, h_i, c_i)
                probs = F.log_softmax(logits, dim=-1)
                topk_lp, topk_ids = probs.topk(beam_width, dim=-1)
                for k in range(beam_width):
                    new_lp = log_prob + topk_lp[0, k].item()
                    new_seq = seq + [topk_ids[0, k].item()]
                    new_beams.append((new_lp, new_seq, h_new, c_new))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        best = beams[0][1]
        best = [t for t in best if t not in (attn_sos_idx, attn_eos_idx, attn_pad_idx)]
        results.append("".join(attn_idx_to_char[i] for i in best))
    return results


def _preprocess_word_image(path: str):
    img = Image.open(path).convert("L")
    w, h = img.size
    new_h = 32
    new_w = max(32, min(int(w * (new_h / h)), max_width))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img)
    img_t = transform(img)
    pad_w = max_width - new_w
    img_t = F.pad(img_t, (0, pad_w, 0, 0), value=1.0)
    return img_t.unsqueeze(0)


def recognize_word(path: str) -> str:
    img_t = _preprocess_word_image(path).to(device)
    with torch.no_grad():
        enc_out = encoder(img_t)
        out = beam_search(decoder, enc_out)
    return out[0]


# ------------------- EasyOCR detection -------------------
reader = easyocr.Reader(["en"], gpu=(device.type == "cuda"))


def _load_image(path: str):
    return cv2.imread(path)


def _preprocess_for_detection(img):
    return cv2.bilateralFilter(img, 5, 50, 50)


# ------------------- main pipeline -------------------
def run_ocr_on_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Entry point for the API.
    Takes raw image bytes, runs detection + CRNN + dictionary + parsing,
    and returns Option-C style JSON.
    """
    # write image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    img = _load_image(tmp_path)
    if img is None:
        raise ValueError("Could not load image")

    H, W = img.shape[:2]
    proc = _preprocess_for_detection(img)
    proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

    results = reader.readtext(proc_rgb, detail=1, paragraph=False)

    tokens: list[dict[str, float | str]] = []

    for (box, _, _) in results:
        pts = np.array(box).astype(int)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

        crop = img[y_min:y_max, x_min:x_max]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as ctmp:
            crop_path = ctmp.name
            cv2.imwrite(crop_path, crop)

        text = recognize_word(crop_path)

        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        h = y_max - y_min

        tokens.append(
            {
                "text": text,
                "x_center": x_c,
                "y_center": y_c,
                "height": h,
            }
        )

    # group into lines (same as notebook)
    tokens.sort(key=lambda t: (t["y_center"], t["x_center"]))
    lines = []
    current = [tokens[0]]
    for tok in tokens[1:]:
        prev = current[-1]
        same = abs(tok["y_center"] - prev["y_center"]) < 0.6 * max(
            tok["height"], prev["height"]
        )
        if same:
            current.append(tok)
        else:
            lines.append(current)
            current = [tok]
    lines.append(current)

    raw_lines: List[str] = []
    for li, line in enumerate(lines, 1):
        line_sorted = sorted(line, key=lambda t: t["x_center"])
        joined = " ".join(t["text"] for t in line_sorted)
        raw_lines.append(f"line_{li}: {joined}")

    corrected_lines = clean_and_correct_prescription_lines(raw_lines)
    medicines = medicines_from_lines(corrected_lines)

    # build Option-C style JSON (no fixed schedule yet – that needs start date from UI)
    meds_json = []
    for m in medicines:
        meds_json.append(
            {
                "raw_line": m.raw_line,
                "form": m.form,
                "name": m.name,
                "strength": m.strength,
                "pattern": m.pattern,
                "days": m.days,
            }
        )

    return {
        "raw_lines": raw_lines,
        "corrected_lines": corrected_lines,
        "medicines": meds_json,
    }
