# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .camera import scan_camera
from .i18n import LANGUAGES
from .ocr import scan_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cedulauy-mrz-scan", description="Read the MRZ of an ID card or passport."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--image", help="path to a photo or scan of the document")
    source.add_argument("-c", "--camera", type=int, metavar="INDEX", help="camera to scan from")
    parser.add_argument("-o", "--output", type=Path, help="write the MRZ to this file")
    parser.add_argument(
        "--lang", choices=LANGUAGES, help="language of the camera texts (default: system)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="show the MRZ detection; press s to save a frame"
    )
    args = parser.parse_args(argv)

    try:
        mrz = (
            scan_camera(args.camera, debug=args.debug, lang=args.lang)
            if args.image is None
            else scan_image(args.image)
        )
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    if mrz is None:
        return 1
    if args.output is None:
        print(mrz)
    else:
        args.output.write_text(mrz + "\n")
    return 0
