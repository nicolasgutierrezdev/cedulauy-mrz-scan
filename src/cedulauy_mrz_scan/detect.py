# SPDX-License-Identifier: GPL-3.0-or-later
"""Locates the MRZ lines in a camera frame without OCR and judges the card placement."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

CARD_ASPECT = 85.6 / 54  # ID-1 card, width / height
MRZ_WIDTH = 0.86  # MRZ width as a fraction of the card width
MRZ_CENTER_Y = 0.78  # MRZ center as a fraction of the card height

SIZE_TOLERANCE = 0.10  # MRZ width vs expected
CENTER_TOLERANCE = 0.06  # center offset, fraction of box width
MAX_ANGLE = 4.0  # degrees
MAX_MOVEMENT = 0.006  # between frames, fraction of box width

PLACE_CARD = "Place the card inside the box"

Box = tuple[int, int, int, int]


class Line(NamedTuple):
    center: np.ndarray
    length: float
    angle: float


def guide_box(frame_shape: tuple[int, ...], fill: float = 0.7) -> Box:
    """Card-shaped box centered in the frame, as (x1, y1, x2, y2)."""
    h, w = frame_shape[:2]
    box_w = int(w * fill)
    box_h = int(box_w / CARD_ASPECT)
    if box_h > h * fill:
        box_h = int(h * fill)
        box_w = int(box_h * CARD_ASPECT)
    x1, y1 = (w - box_w) // 2, (h - box_h) // 2
    return x1, y1, x1 + box_w, y1 + box_h


def mrz_band(box: Box) -> Box:
    """Area of the guide box where the MRZ should be."""
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    band_h = int(0.22 * box_h)
    band_y = y1 + int(MRZ_CENTER_Y * box_h) - band_h // 2
    margin = int((1 - MRZ_WIDTH) / 2 * box_w) - 10
    return x1 + margin, band_y, x2 - margin, band_y + band_h


def lines_center(lines: list[Line]) -> np.ndarray:
    return np.mean([line.center for line in lines], axis=0)


def find_mrz(frame: np.ndarray, box: Box) -> tuple[list[Line], np.ndarray]:
    """
    Finds up to 3 MRZ text lines around the guide box, sorted top to bottom.

    Dark text becomes bright blobs and each row of characters merges into one
    long thin blob, so detection ignores fingers on the card edges and the
    background. Also returns the blob mask, for debugging.
    """
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    rx1, ry1 = max(x1 - box_w // 10, 0), max(y1 - box_h // 10, 0)
    rx2, ry2 = min(x2 + box_w // 10, frame.shape[1]), min(y2 + box_h // 10, frame.shape[0])
    gray = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (box_w // 30, box_h // 8))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (box_w // 35, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    expected_length = MRZ_WIDTH * box_w
    lines = []
    for contour in contours:
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(contour)
        length, thickness = max(rw, rh), min(rw, rh)
        if length < 0.5 * expected_length or thickness > 0.1 * box_h or length < 10 * thickness:
            continue
        vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle = float((np.degrees(np.arctan2(vy, vx)) + 90) % 180 - 90)
        if abs(angle) <= 30:
            lines.append(Line(np.float32([cx + rx1, cy + ry1]), length, angle))
    lines.sort(key=lambda line: line.center[1])

    if len(lines) > 3:
        # Keep the 3 consecutive lines with the most similar lengths
        def spread(i: int) -> float:
            lengths = [line.length for line in lines[i : i + 3]]
            return max(lengths) - min(lengths)

        best = min(range(len(lines) - 2), key=spread)
        lines = lines[best : best + 3]

    full_mask = np.zeros(frame.shape[:2], np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = mask
    return lines, full_mask


def check_placement(lines: list[Line], box: Box, previous_center: np.ndarray | None) -> str | None:
    """Returns an instruction for the user, or None if the card is well placed."""
    if not lines:
        return PLACE_CARD
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    center = lines_center(lines)

    if center[1] < y1 + box_h / 2:
        return "Turn the card around (code at the bottom)"
    if len(lines) < 3:
        return "Keep the code (bottom lines) visible"

    length = np.median([line.length for line in lines])
    expected_length = MRZ_WIDTH * box_w
    if length < expected_length * (1 - SIZE_TOLERANCE):
        return "Move the card closer"
    if length > expected_length * (1 + SIZE_TOLERANCE):
        return "Move the card back"

    tolerance = box_w * CENTER_TOLERANCE
    if (
        abs(center[0] - (x1 + x2) / 2) > tolerance
        or abs(center[1] - (y1 + MRZ_CENTER_Y * box_h)) > tolerance
    ):
        return "Center the card in the box"

    if abs(np.mean([line.angle for line in lines])) > MAX_ANGLE:
        return "Straighten the card"

    if (
        previous_center is not None
        and np.linalg.norm(center - previous_center) > box_w * MAX_MOVEMENT
    ):
        return "Hold still"
    return None


def crop_mrz(frame: np.ndarray, lines: list[Line], box: Box) -> np.ndarray:
    """Cuts out the MRZ with a margin, rotated so the lines are horizontal."""
    x1, y1, x2, y2 = box
    cx, cy = (float(v) for v in lines_center(lines))
    angle = float(np.mean([line.angle for line in lines]))
    width = int(max(line.length for line in lines) + 0.1 * (x2 - x1))
    height = int(abs(lines[-1].center[1] - lines[0].center[1]) + 0.15 * (y2 - y1))
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    matrix[:, 2] += (width / 2 - cx, height / 2 - cy)
    return cv2.warpAffine(
        frame, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
