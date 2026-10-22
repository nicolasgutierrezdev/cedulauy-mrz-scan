# SPDX-License-Identifier: GPL-3.0-or-later
"""Scanner overlay drawn on the camera frames."""

from __future__ import annotations

import os
import time
from functools import cache, lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detect import PLACE_CARD, Box, Line, mrz_band
from .i18n import translate

# BGR
WHITE = (245, 245, 245)
GRAY = (170, 170, 170)
DARK_GRAY = (90, 90, 90)
AMBER = (40, 180, 255)
GREEN = (110, 215, 90)
BLUE = (255, 170, 70)
PANEL = (28, 28, 28)

STATE_COLORS = {"adjust": AMBER, "hold": GREEN, "scanning": BLUE, "done": GREEN}

FONT_PATHS = {
    False: [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ],
    True: [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ],
}

TIP = "Use good, even lighting and avoid reflections"


@cache
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = next((path for path in FONT_PATHS[bold] if os.path.exists(path)), None)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


@lru_cache(maxsize=256)
def _text_mask(text: str, size: int, bold: bool = False) -> np.ndarray:
    """Anti-aliased text as a float alpha mask."""
    font = _font(size, bold)
    ascent, descent = font.getmetrics()
    image = Image.new("L", (max(int(font.getlength(text)) + 2, 1), ascent + descent))
    ImageDraw.Draw(image).text((1, 0), text, font=font, fill=255)
    return np.asarray(image, np.float32)[..., None] / 255


def draw_text(frame, text, x, y, size=20, color=WHITE, bold=False, center=False) -> None:
    """Draws text with its top at y; x is the left edge, or the center if center=True."""
    mask = _text_mask(text, size, bold)
    h, w = mask.shape[:2]
    if center:
        x -= w // 2
    fx1, fy1 = max(x, 0), max(y, 0)
    fx2, fy2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if fx1 >= fx2 or fy1 >= fy2:
        return
    alpha = mask[fy1 - y : fy2 - y, fx1 - x : fx2 - x]
    region = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (region * (1 - alpha) + np.float32(color) * alpha).astype(np.uint8)


def _rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1) -> None:
    radius = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        for p1, p2 in (
            ((x1 + radius, y1), (x2 - radius, y1)),
            ((x1 + radius, y2), (x2 - radius, y2)),
            ((x1, y1 + radius), (x1, y2 - radius)),
            ((x2, y1 + radius), (x2, y2 - radius)),
        ):
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    for cx, cy, angle in (
        (x1 + radius, y1 + radius, 180),
        (x2 - radius, y1 + radius, 270),
        (x2 - radius, y2 - radius, 0),
        (x1 + radius, y2 - radius, 90),
    ):
        cv2.ellipse(img, (cx, cy), (radius, radius), angle, 0, 90, color, thickness, cv2.LINE_AA)


def _blend_rounded_rect(frame, x1, y1, x2, y2, radius, color, alpha) -> None:
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1] - 1), min(y2, frame.shape[0] - 1)
    if x1 >= x2 or y1 >= y2:
        return
    region = frame[y1 : y2 + 1, x1 : x2 + 1]
    overlay = region.copy()
    _rounded_rect(overlay, 0, 0, x2 - x1, y2 - y1, radius, color)
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, dst=region)


def _corner_brackets(frame, box: Box, color, radius, length, thickness=5) -> None:
    x1, y1, x2, y2 = box
    for cx, cy, angle, sx, sy in (
        (x1 + radius, y1 + radius, 180, 1, 1),
        (x2 - radius, y1 + radius, 270, -1, 1),
        (x2 - radius, y2 - radius, 0, -1, -1),
        (x1 + radius, y2 - radius, 90, 1, -1),
    ):
        edge_x = x1 if sx > 0 else x2
        edge_y = y1 if sy > 0 else y2
        cv2.ellipse(frame, (cx, cy), (radius, radius), angle, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, edge_y), (cx + sx * length, edge_y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (edge_x, cy), (edge_x, cy + sy * length), color, thickness, cv2.LINE_AA)


@lru_cache(maxsize=4)
def _outside_mask(shape: tuple[int, ...], box: Box, radius: int) -> np.ndarray:
    mask = np.full(shape[:2], 255, np.uint8)
    _rounded_rect(mask, *box, radius, 0)
    return mask


def _pill(frame, text, x, y, size, dot_color, height, center=False) -> None:
    """Dark rounded label with a colored dot; x is the left edge, or the center."""
    mask = _text_mask(text, size, True)
    width = mask.shape[1] + height + 20
    if center:
        x -= width // 2
    _blend_rounded_rect(frame, x, y, x + width, y + height, height // 2, PANEL, 0.85)
    cv2.circle(
        frame, (x + height // 2 + 4, y + height // 2), height // 7, dot_color, -1, cv2.LINE_AA
    )
    draw_text(frame, text, x + height, y + (height - mask.shape[0]) // 2, size, WHITE, True)


def _draw_band(frame, box: Box, state: str, accent) -> None:
    bx1, by1, bx2, by2 = mrz_band(box)
    done = state == "done"
    _blend_rounded_rect(
        frame, bx1, by1, bx2, by2, 10, GREEN if done else WHITE, 0.35 if done else 0.10
    )
    _rounded_rect(frame, bx1, by1, bx2, by2, 10, accent, 1)
    draw_text(frame, "MRZ", bx1 + 4, by1 - 22, 15, accent, bold=True)
    if state == "scanning":
        phase = (np.sin(time.monotonic() * 4) + 1) / 2
        sy = int(by1 + 4 + phase * (by2 - by1 - 8))
        _blend_rounded_rect(frame, bx1, sy - 6, bx2, sy + 6, 6, BLUE, 0.25)
        cv2.line(frame, (bx1 + 6, sy), (bx2 - 6, sy), BLUE, 2, cv2.LINE_AA)


def draw_ui(
    frame: np.ndarray,
    box: Box,
    lines: list[Line],
    state: str,
    message: str,
    progress: float = 0.0,
    note: str = "",
    lang: str = "en",
) -> None:
    """
    Draws the scanner overlay in place.

    state is "adjust", "hold", "scanning" or "done"; progress (0..1)
    fills the hold bar. Texts are shown in lang ("en" or "es").
    """
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    mid_x = (x1 + x2) // 2
    radius = box_w // 28
    accent = WHITE if message == PLACE_CARD else STATE_COLORS[state]

    def tr(text: str) -> str:
        return translate(text, lang)

    cv2.copyTo(
        cv2.convertScaleAbs(frame, alpha=0.35), _outside_mask(frame.shape, box, radius), frame
    )
    _draw_band(frame, box, state, accent)

    for (cx, cy), length, angle in lines:
        dx = length / 2 * np.cos(np.radians(angle))
        dy = length / 2 * np.sin(np.radians(angle))
        for ex, ey in ((cx - dx, cy - dy), (cx + dx, cy + dy)):
            cv2.circle(frame, (int(ex), int(ey)), 4, accent, -1, cv2.LINE_AA)

    _rounded_rect(frame, x1, y1, x2, y2, radius, DARK_GRAY, 1)
    _corner_brackets(frame, box, accent, radius, box_w // 10)

    draw_text(frame, tr("Scan your ID card"), mid_x, max(y1 - 62, 8), 26, WHITE, True, center=True)
    draw_text(
        frame,
        tr("Back side, code at the bottom, inside the frame"),
        mid_x,
        max(y1 - 30, 40),
        16,
        GRAY,
        center=True,
    )

    bar_y = y2 + 14
    if state == "hold":
        _blend_rounded_rect(frame, x1, bar_y, x2, bar_y + 5, 3, (80, 80, 80), 0.8)
        filled = x1 + int(min(progress, 1.0) * box_w)
        if filled > x1 + 6:
            _rounded_rect(frame, x1, bar_y, filled, bar_y + 5, 3, GREEN)

    pill_h = 44
    pill_y = min(bar_y + 18, frame.shape[0] - pill_h - 34)
    _pill(frame, tr(message), mid_x, pill_y, 20, accent, pill_h, center=True)
    if note:
        draw_text(frame, tr(note), mid_x, pill_y + pill_h + 8, 15, GRAY, center=True)

    _pill(frame, tr(TIP), 16, 16, 15, AMBER, 32)
    hint = tr("Q  quit")
    hint_w = _text_mask(hint, 14).shape[1]
    draw_text(frame, hint, frame.shape[1] - 16 - hint_w, frame.shape[0] - 26, 14, GRAY)


def draw_debug_mask(frame: np.ndarray, mask: np.ndarray, box: Box) -> None:
    """Detector blobs inside the guide box, as a thumbnail in the bottom-left corner."""
    x1, y1, x2, y2 = box
    thumb_w = max(x1 - 32, 140)
    thumb_h = thumb_w * (y2 - y1) // (x2 - x1)
    thumb = cv2.cvtColor(cv2.resize(mask[y1:y2, x1:x2], (thumb_w, thumb_h)), cv2.COLOR_GRAY2BGR)
    ty = frame.shape[0] - thumb_h - 16
    frame[ty : ty + thumb_h, 16 : 16 + thumb_w] = thumb
    cv2.rectangle(frame, (16, ty), (16 + thumb_w, ty + thumb_h), GRAY, 1)
    draw_text(frame, "Detection", 16, ty - 20, 14, GRAY)
