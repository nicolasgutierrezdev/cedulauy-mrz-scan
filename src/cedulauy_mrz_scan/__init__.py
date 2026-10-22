# SPDX-License-Identifier: GPL-3.0-or-later
"""Scan and read the MRZ of Uruguayan identity cards."""

from importlib.metadata import version

from .camera import scan_camera
from .ocr import read_mrz, scan_image

__version__ = version("cedulauy-mrz-scan")
__all__ = ["__version__", "read_mrz", "scan_camera", "scan_image"]
