# SPDX-License-Identifier: GPL-3.0-or-later
"""MRZ reading with Tesseract and the OCR-B model, validated by check digits."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from mrz.checker.td1 import TD1CodeChecker

# OCR-B model from https://github.com/DoubangoTelecom/tesseractMRZ;
# the default "eng" model keeps reading the "<" filler as "K" or "L".
TESSDATA_DIR = Path(__file__).parent / "tessdata"
WINDOWS_TESSERACT = Path(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR")

TD1_LINE_LENGTH = 30
MIN_WIDTH = 1600


TESSERACT_OPTIONS = [
    "--tessdata-dir", str(TESSDATA_DIR),
    "-l", "mrz",
    "--psm", "6",
    "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
]  # fmt: skip


@lru_cache(maxsize=1)
def find_tesseract() -> str:
    model = TESSDATA_DIR / "mrz.traineddata"
    if not model.exists():
        raise RuntimeError(f"Missing {model}")
    executable = shutil.which("tesseract") or shutil.which("tesseract", path=WINDOWS_TESSERACT)
    if executable is None:
        raise RuntimeError("Tesseract not found, install it and add it to PATH")
    return executable


def _tesseract(gray: np.ndarray) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        image, output = Path(tmp, "image.png"), Path(tmp, "output")
        cv2.imwrite(str(image), gray)
        result = subprocess.run(
            [find_tesseract(), str(image), str(output), *TESSERACT_OPTIONS],
            check=False,
            capture_output=True,
            # Tesseract's own threading only adds overhead on a small MRZ crop
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Tesseract failed: {result.stderr.decode(errors='replace')}")
        return output.with_suffix(".txt").read_text(encoding="utf-8")


def _ocr_lines(gray: np.ndarray) -> list[str]:
    text = _tesseract(gray)
    lines = (re.sub(r"[^A-Z0-9<]", "", line.upper()) for line in text.splitlines())
    return [_fix_filler(line) for line in lines if len(line) >= 25 and "<" in line]


def _fix_filler(line: str) -> str:
    # A lone K/L/C/E/X between "<<" pairs is a misread filler: single-letter name
    # parts are separated by one "<", and check digits can't catch it (K = 20 ≡ "<").
    return re.sub(r"(?<=<<)[KLCEX](?=<<)", "<", line)


def validate(lines: list[str]) -> str | None:
    """Returns the MRZ if the last three rows form a valid TD1 code, else None."""
    if len(lines) < 3 or any(len(line) > 34 for line in lines[-3:]):
        return None
    td1 = [line[:TD1_LINE_LENGTH].ljust(TD1_LINE_LENGTH, "<") for line in lines[-3:]]
    return "".join(td1) if TD1CodeChecker("\n".join(td1)) else None


def _variants(image: np.ndarray) -> tuple[np.ndarray, ...]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] < MIN_WIDTH:
        scale = MIN_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad = gray.shape[0] // 10
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    flattened = cv2.divide(gray, cv2.GaussianBlur(gray, (0, 0), 25), scale=200)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return blurred, flattened, binary


def read_mrz(image: np.ndarray) -> tuple[str | None, list[str]]:
    """
    Reads the MRZ from a BGR image of the code area.

    Returns (mrz, lines): the validated 90-character TD1 MRZ or None, and the
    best rows read, for feedback.
    """
    best: list[str] = []
    for variant in _variants(image):
        lines = _ocr_lines(variant)
        mrz = validate(lines)
        if mrz is not None:
            return mrz, lines
        if len(lines) > len(best):
            best = lines
    return None, best


def scan_image(path: str | os.PathLike[str]) -> str:
    """Reads the MRZ from a photo or scan of the document; raises ValueError on failure."""
    image = cv2.imread(os.fspath(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    mrz, lines = read_mrz(image[image.shape[0] // 2 :])
    if mrz is None:
        raise ValueError(f"Could not read the MRZ, best read: {lines}")
    return mrz
