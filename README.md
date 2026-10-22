cedulauy-mrz-scan
=================

[![CI](https://github.com/nicolasgutierrezdev/cedulauy-mrz-scan/actions/workflows/ci.yml/badge.svg)](https://github.com/nicolasgutierrezdev/cedulauy-mrz-scan/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cedulauy-mrz-scan.svg)](https://pypi.org/project/cedulauy-mrz-scan/)
[![Python versions](https://img.shields.io/pypi/pyversions/cedulauy-mrz-scan.svg)](https://pypi.org/project/cedulauy-mrz-scan/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<p align="center">
  <img src="https://raw.githubusercontent.com/nicolasgutierrezdev/cedulauy-mrz-scan/main/assets/banner.svg" alt="cedulauy-mrz-scan: MRZ reader for Uruguayan ID cards" width="100%">
</p>

*Read the MRZ of Uruguayan identity cards (cédula de identidad) from a webcam or a photo.*

> **Alpha.** The API may change between releases.

Install
-------

The package needs Python 3.10+ and the Tesseract OCR engine, which is a separate program.

**1. Install Tesseract**

| System          | Command                                                             |
|-----------------|---------------------------------------------------------------------|
| Debian / Ubuntu | `sudo apt install tesseract-ocr`                                    |
| Fedora          | `sudo dnf install tesseract`                                        |
| Arch            | `sudo pacman -S tesseract`                                          |
| macOS           | `brew install tesseract`                                            |
| Windows         | [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) |

Any language data works: the MRZ model (OCR-B) ships with this package, so no `tessdata` setup is
needed. On Windows the default install folder (`C:\Program Files\Tesseract-OCR`) is found even if
it is not on `PATH`.

Check it with:

```bash
tesseract --version
```

**2. Install the package**

```bash
uv add cedulauy-mrz-scan
# or
pip install cedulauy-mrz-scan
```

Usage
-----

Point a webcam at the back of the card:

```bash
cedulauy-mrz-scan --camera 0
```

A window shows a card-shaped guide and tells you how to hold the card: closer, straighter, still.
When it is well placed, two pictures are taken and the MRZ is printed. Press `q` or `Esc` to quit.

From a photo or scan:

```bash
cedulauy-mrz-scan --image card.jpg
cedulauy-mrz-scan --image card.jpg -o mrz.txt
```

| Option               | Description                                              |
|----------------------|----------------------------------------------------------|
| `-c, --camera INDEX` | Scan from a camera (`0` is usually the built-in one)     |
| `-i, --image PATH`   | Read a photo or scan                                     |
| `-o, --output FILE`  | Write the MRZ to a file instead of printing it           |
| `--lang {en,es}`     | Language of the on-screen texts (default: system's)      |
| `--debug`            | Show the line detection; `s` saves the current frame     |

From Python:

```python
from cedulauy_mrz_scan import scan_camera, scan_image

mrz = scan_camera(0)          # None if the user quits
mrz = scan_image("card.jpg")  # raises ValueError if unreadable
```

The result is the raw TD1 MRZ as one 90-character string (3 lines of 30). `read_mrz(image)`
reads an already cropped image of the code and returns `(mrz or None, lines read)`.

Features
--------

- **Guided capture**: finds the MRZ lines in the live image and tells the user how to place the
  card, before any OCR runs.
- **Verified**: a read is only returned if it passes the check digits.
- **Double capture**: two pictures must give the same read, which also covers the name (check
  digits don't).
- **OCR-B model**: bundled, trained for MRZ text; the default English model misreads `<` as `K`.
- **Offline**: everything runs locally, nothing is sent anywhere.
- **Spanish / English**: on-screen texts follow the system language.

Tips
----

- Even light without reflections matters most; a shiny card under a ceiling lamp is the usual
  failure.
- Hold the card steady once the bar starts filling; moving restarts it.

Development
-----------

```bash
uv sync            # create .venv and install dev dependencies
uv run pytest      # run tests
uv run ruff check  # lint
uv build           # build sdist and wheel into dist/
```

License
-------

GPL-3.0-or-later. The bundled OCR-B model (`tessdata/mrz.traineddata`) is from
[DoubangoTelecom/tesseractMRZ](https://github.com/DoubangoTelecom/tesseractMRZ), BSD-3-Clause.
