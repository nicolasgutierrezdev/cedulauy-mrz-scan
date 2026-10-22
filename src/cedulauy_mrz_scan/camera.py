# SPDX-License-Identifier: GPL-3.0-or-later
"""Live camera scanning with placement guidance."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import detect, ocr, ui
from .i18n import system_language

HOLD_TIME = 0.5  # seconds well placed before the first picture
CAPTURE_GAP = 0.7  # seconds, still holding, before the second picture
NOTE_DURATION = 3.0
WINDOW = "MRZ scanner"
DEBUG_DIR = Path("debug")


def _save_debug(name: str, image: np.ndarray) -> Path:
    DEBUG_DIR.mkdir(exist_ok=True)
    path = DEBUG_DIR / f"{name}_{int(time.time())}.png"
    cv2.imwrite(str(path), image)
    return path


class _Reader:
    """OCRs picture pairs in a background thread so the preview stays live."""

    def __init__(self, debug: bool) -> None:
        self.debug = debug
        self.mrz: str | None = None
        self.done = threading.Event()
        self.ready = threading.Event()
        self.ready.set()
        self._note = ""
        self._note_time = 0.0
        self._pairs: queue.Queue[tuple[np.ndarray, np.ndarray] | None] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, pair: tuple[np.ndarray, np.ndarray]) -> None:
        self.ready.clear()
        self._pairs.put(pair)

    def note(self, now: float) -> str:
        return self._note if now - self._note_time < NOTE_DURATION else ""

    def stop(self) -> None:
        if not self.done.is_set():
            try:
                self._pairs.get_nowait()
            except queue.Empty:
                pass
            self._pairs.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while (pair := self._pairs.get()) is not None:
            reads = []
            for i, picture in enumerate(pair, 1):
                mrz, lines = ocr.read_mrz(picture)
                if self.debug:
                    print(f"Picture {i}: {'valid' if mrz else 'invalid'} {lines}")
                reads.append(mrz)
                if mrz is None:
                    break
            # Both reads must agree: names are not covered by check digits
            if len(reads) == 2 and reads[0] == reads[1]:
                self.mrz = reads[0]
                self.done.set()
                return
            if self.debug:
                for i, picture in enumerate(pair, 1):
                    print(f"Saved {_save_debug(f'failed_{i}', picture)}")
            self._note = (
                "The two pictures gave different reads, try again"
                if len(reads) == 2
                else "Could not read the code, try again (better light helps)"
            )
            self._note_time = time.monotonic()
            self.ready.set()


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")
    # MJPG gives 1280x720 at 30 fps on most webcams; YUYV drops to 10 fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def scan_camera(
    camera_index: int = 0,
    *,
    show: bool = True,
    min_sharpness: float = 15.0,
    debug: bool = False,
    lang: str | None = None,
) -> str | None:
    """
    Guides the user to place the card in a guide box and reads its MRZ.

    Once the card has been well placed for HOLD_TIME, two pictures are taken
    CAPTURE_GAP apart; moving in between restarts the hold. Returns the MRZ
    when both pictures pass the check digits with identical reads, or None if
    the user quits (q / ESC).

    With debug=True the detector mask is shown, "s" saves the current frame
    and failed pictures are saved to ./debug/. lang is "en" or "es"; by default
    it follows the system language.
    """
    lang = lang or system_language()
    ocr.find_tesseract()
    cap = _open_camera(camera_index)
    reader = _Reader(debug)
    if show:
        if sys.platform == "linux":
            # opencv-python's bundled Qt only ships the X11 plugin; skip its Wayland attempt
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    previous_center = None
    placed_since = None
    first_picture = None
    try:
        while not reader.done.is_set():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            box = detect.guide_box(frame.shape)
            lines, mask = detect.find_mrz(frame, box)
            instruction = detect.check_placement(lines, box, previous_center)
            previous_center = detect.lines_center(lines) if lines else None

            crop = None
            if instruction is None:
                crop = detect.crop_mrz(frame, lines, box)
                if detect.sharpness(crop) < min_sharpness:
                    instruction = "Image blurry, hold still"

            now = time.monotonic()
            if instruction is not None or not reader.ready.is_set():
                placed_since, first_picture = None, None
            elif placed_since is None:
                placed_since = now
            elif first_picture is None and now - placed_since >= HOLD_TIME:
                first_picture = crop
            elif first_picture is not None and now - placed_since >= HOLD_TIME + CAPTURE_GAP:
                reader.submit((first_picture, crop))
                placed_since, first_picture = None, None

            if not show:
                continue

            raw_frame = frame.copy() if debug else None
            progress = 0.0
            note = reader.note(now)
            if not reader.ready.is_set():
                state, message = "scanning", "Reading the code..."
                note = "Pictures taken, you can lower the card"
            elif instruction is not None:
                state, message = "adjust", instruction
            else:
                state, message = "hold", "Hold still"
                if placed_since is not None:
                    progress = (now - placed_since) / (HOLD_TIME + CAPTURE_GAP)
            ui.draw_ui(frame, box, lines, state, message, progress, note, lang)
            if debug:
                ui.draw_debug_mask(frame, mask, box)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if debug and key == ord("s"):
                print(f"Saved {_save_debug('frame', raw_frame)} (instruction: {instruction})")

        if reader.done.is_set() and show:
            ok, fresh = cap.read()
            frame = fresh if ok else frame
            ui.draw_ui(frame, box, [], "done", "MRZ read successfully", lang=lang)
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(900)
    finally:
        reader.stop()
        cap.release()
        if show:
            cv2.destroyAllWindows()

    return reader.mrz
