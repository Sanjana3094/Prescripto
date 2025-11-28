# backend/reminder_parser.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re

# import dictionary + fuzzy matcher from dictionary_fix
from .dictionary_fix import MEDICINE_LABELS, _correct_word


@dataclass
class Medicine:
    raw_line: str
    name: str
    form: str = ""
    strength: str = ""
    pattern: str = ""
    days: Optional[int] = None


FORM_KEYWORDS = {
    "tab": "Tab",
    "tablet": "Tab",
    "tabs": "Tab",
    "tb": "Tab",          # OCR often gives Tb
    "cap": "Cap",
    "caps": "Cap",
    "capsule": "Cap",
    "capsules": "Cap",
    "syr": "Syrup",
    "syrup": "Syrup",
    "inj": "Injection",
    "injection": "Injection",
    "drop": "Drops",
    "drops": "Drops",
}

DAY_WORDS = {"day", "days", "d", "ddys", "dys"}
UNIT_WORDS = {"mg", "mcg", "g", "ml"}

# things that strongly suggest header/address lines
HEADER_KEYWORDS = [
    "hospital",
    "clinic",
    "nursing home",
    "speciality",
    "specialty",
    "road",
    "block",
    "layout",
    "street",
    "bengaluru",
    "bangalore",
    "koramangala",
    "tel:",
    "phone",
    "fax",
    "date",
    "dr.",
    "dr ",
    "doctor",
    "age",
    "sex",
]


def _strip_line_prefix(line: str) -> str:
    """
    Convert 'line_1: Tab Crocin 1-1-1 3 days'
    to 'Tab Crocin 1-1-1 3 days'.
    """
    if ":" in line:
        prefix, rest = line.split(":", 1)
        if prefix.strip().lower().startswith("line_"):
            return rest.strip()
    return line.strip()


def _looks_like_header(clean: str) -> bool:
    """
    Very rough check for hospital/address/header lines.
    """
    low = clean.lower()
    # lot of digits + punctuation + no mg/ml etc → probably phone/address
    digit_ratio = sum(ch.isdigit() for ch in low) / max(len(low), 1)
    if digit_ratio > 0.3 and not any(u in low for u in ("mg", "mcg", "ml", "g")):
        return True

    if any(k in low for k in HEADER_KEYWORDS):
        return True

    return False


def medicines_from_lines(corrected_lines: List[str]) -> List[Medicine]:
    """
    Heuristic parser:
    - ONE Medicine object per corrected line (no merging).
    - Extract (name, form, strength, pattern, days) if possible.
    - Handles patterns split across tokens (1 - 1 - 1) and typos like "ddys".
    - Skips very likely non-medicine header/address lines,
      and short annotations like '(B/F)'.
    - Tries to build the medicine *name* using only dictionary labels when possible.
    """
    medicines: List[Medicine] = []

    for line in corrected_lines:
        clean = _strip_line_prefix(line)
        if not clean:
            continue

        tokens = clean.split()
        if not tokens:
            continue

        name_tokens: List[str] = []
        form = ""
        strength = ""
        pattern = ""
        days: Optional[int] = None

        i = 0
        n = len(tokens)
        while i < n:
            t = tokens[i]
            low = t.lower()

            # --- FORM (Tab, Cap, Syrup, etc.) ---
            if low in FORM_KEYWORDS and not form:
                form = FORM_KEYWORDS[low]
                i += 1
                continue

            # --- PATTERN: 1-0-1 OR 1 - 0 - 1 ---
            m_pat_single = re.fullmatch(r"(\d)\s*-\s*(\d)\s*-\s*(\d)", low)
            if not pattern and m_pat_single:
                g1, g2, g3 = m_pat_single.groups()
                pattern = f"{g1}-{g2}-{g3}"
                i += 1
                continue

            if (
                not pattern
                and i + 4 < n
                and tokens[i].isdigit()
                and tokens[i + 1] == "-"
                and tokens[i + 2].isdigit()
                and tokens[i + 3] == "-"
                and tokens[i + 4].isdigit()
            ):
                pattern = f"{tokens[i]}-{tokens[i+2]}-{tokens[i+4]}"
                i += 5
                continue

            # --- DAYS: "3 days", "3 d", "3 ddys", etc. ---
            if (
                days is None
                and t.isdigit()
                and i + 1 < n
                and tokens[i + 1].lower() in DAY_WORDS
            ):
                try:
                    days = int(t)
                except ValueError:
                    pass
                i += 2
                continue

            if days is None:
                m_days = re.fullmatch(r"(\d+)(day|days|d|ddys|dys)", low)
                if m_days:
                    try:
                        days = int(m_days.group(1))
                    except ValueError:
                        pass
                    i += 1
                    continue

            # --- STRENGTH: "5 mg" or "250mg" ---
            if (
                not strength
                and t.replace(".", "", 1).isdigit()
                and i + 1 < n
                and tokens[i + 1].lower() in UNIT_WORDS
            ):
                strength = f"{t} {tokens[i+1]}"
                i += 2
                continue

            m_strength_single = re.fullmatch(r"(\d+(?:\.\d+)?)(mg|mcg|g|ml)", low)
            if not strength and m_strength_single:
                num, unit = m_strength_single.groups()
                strength = f"{num} {unit}"
                i += 1
                continue

            # --- SKIP pure numbers / dashes / day words / units from the name ---
            if t.isdigit() or t == "-" or low in DAY_WORDS or low in UNIT_WORDS:
                i += 1
                continue

            # Otherwise, treat as part of the free-text name
            name_tokens.append(t)
            i += 1

        # ------------------------------------------------------------------
        # Decide whether this line *really* looks like a medicine line
        # ------------------------------------------------------------------
        has_medicine_signal = bool(form or strength or pattern or days)

        # 1) obvious header / address / doctor line → skip
        if not has_medicine_signal and _looks_like_header(clean):
            continue

        low_clean = clean.lower()
        # 2) very short annotations like "(B/F)", "B/F", "(/F)" etc. → skip
        short_tokens = low_clean.split()
        if (
            len(short_tokens) <= 3
            and not has_medicine_signal
            and any(ch.isalpha() for ch in low_clean)
            and all(re.fullmatch(r"[\(\)\/bf\.]+", tok) for tok in short_tokens)
        ):
            continue

        # 3) generic safety: if line is tiny (len < 4 chars) and no signal → skip
        if len(clean) < 4 and not has_medicine_signal:
            continue

        # ------------------------------------------------------------------
        # Build the medicine NAME using dictionary labels when possible
        # ------------------------------------------------------------------
        dict_hits: List[str] = []
        for t in tokens:
            if not any(ch.isalpha() for ch in t):
                continue
            # use a fairly strict cutoff so we don't hallucinate wrong drugs
            corrected = _correct_word(t, cutoff=0.9)
            if corrected in MEDICINE_LABELS and corrected not in dict_hits:
                dict_hits.append(corrected)

        if dict_hits:
            name = " ".join(dict_hits)
        else:
            # fall back to cleaned free-text name
            name = " ".join(name_tokens).strip()
            if not name:
                name = clean

        # Skip truly empty garbage
        if not name and not form and not strength and not pattern and days is None:
            continue

        medicines.append(
            Medicine(
                raw_line=line,
                name=name,
                form=form,
                strength=strength,
                pattern=pattern,
                days=days,
            )
        )

    return medicines
