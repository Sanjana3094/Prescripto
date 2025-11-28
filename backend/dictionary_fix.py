from pathlib import Path
import re
import difflib
from typing import List, Set

import pandas as pd

# --------- locate data folder ---------
BASE_DIR = Path(__file__).resolve().parent          # .../backend
PROJECT_ROOT = BASE_DIR.parent                      # .../prescripto_final

CANDIDATE_DIRS = [
    BASE_DIR / "data",          # prescripto_final/backend/data
    PROJECT_ROOT / "data",      # prescripto_final/data
]

DATA_DIR = None
for d in CANDIDATE_DIRS:
    if d.exists():
        DATA_DIR = d
        break

if DATA_DIR is None:
    print("[dictionary_fix] WARNING: No data directory found.")
    MEDICINE_LABELS: Set[str] = set()
else:
    print(f"[dictionary_fix] Looking for label files in: {DATA_DIR}")
    MEDICINE_LABELS: Set[str] = set()

    # support .csv and .xlsx
    files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))

    if not files:
        print("[dictionary_fix] WARNING: No csv/xlsx files found in data directory.")
    else:
        for fp in files:
            try:
                if fp.suffix.lower() == ".csv":
                    # handle normal CSV + UTF-8 with BOM
                    df = pd.read_csv(fp, encoding="utf-8-sig")
                else:
                    df = pd.read_excel(fp)

                # collect all string-like values from object columns
                for col in df.columns:
                    if df[col].dtype == "object":
                        for v in df[col].dropna().astype(str).tolist():
                            v = v.strip()
                            if v:
                                MEDICINE_LABELS.add(v)
                print(f"[dictionary_fix] Loaded {len(df)} rows from {fp.name}")
            except Exception as e:
                print(f"[dictionary_fix] ERROR reading {fp.name}: {e}")

print(f"[dictionary_fix] Loaded {len(MEDICINE_LABELS)} unique medicine labels from CSV/XLSX.")


def _normalize_token(tok: str) -> str:
    """Lowercase and strip non-letters for matching."""
    tok = tok.strip()
    tok = re.sub(r"[^A-Za-z]", "", tok)
    return tok.lower()


def _build_lookup_map():
    """Map normalized tokens -> list of full labels that contain them."""
    lookup = {}
    for label in MEDICINE_LABELS:
        norm = _normalize_token(label)
        if not norm:
            continue
        lookup.setdefault(norm, set()).add(label)
    return lookup


LOOKUP_MAP = _build_lookup_map()
LOOKUP_KEYS = list(LOOKUP_MAP.keys())


def _correct_word(word: str, cutoff: float = 0.83) -> str:
    """
    Try to correct a single word using fuzzy matching on the label dictionary.
    Returns the corrected medicine name (first candidate) or the original word.
    """
    norm = _normalize_token(word)
    if not norm:
        return word

    matches = difflib.get_close_matches(norm, LOOKUP_KEYS, n=1, cutoff=cutoff)
    if not matches:
        return word

    best_key = matches[0]
    # pick any full label that maps to this normalized token
    candidates = LOOKUP_MAP.get(best_key)
    if not candidates:
        return word

    # choose the shortest candidate (usually the clean brand name)
    best_label = sorted(candidates, key=len)[0]
    return best_label


def clean_and_correct_prescription_lines(lines: List[str]) -> List[str]:
    """
    Take raw OCR lines and return cleaned + dictionary-corrected lines.
    This is what pipeline.py calls.
    """
    if not MEDICINE_LABELS:
        # dictionary not loaded; just return original lines
        print("[dictionary_fix] WARNING: No medicine labels loaded; skipping correction.")
        return lines

    corrected_lines: List[str] = []

    for line in lines:
        parts = line.split()
        new_parts = []

        for p in parts:
            # only try to correct word-like tokens with letters
            if re.search(r"[A-Za-z]", p):
                corrected = _correct_word(p)
                new_parts.append(corrected)
            else:
                new_parts.append(p)

        corrected_line = " ".join(new_parts)
        corrected_lines.append(corrected_line)

    return corrected_lines
