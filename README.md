# cedulauy-mrz-scan

Scan and read the MRZ (Machine Readable Zone) of Uruguayan identity cards (cédula de identidad).
Passports (TD3) are read too.

A live camera view guides the user to place the card, takes two pictures and returns the MRZ
only when both pass the check digits and agree.

## Requirements

[Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) must be installed
(`apt install tesseract-ocr`, `pacman -S tesseract`, `brew install tesseract`, or the
[Windows installer](https://github.com/UB-Mannheim/tesseract/wiki)). It is looked up on `PATH`,
and on Windows also in `C:\Program Files\Tesseract-OCR`. The OCR-B model is bundled.

## Installation

```bash
pip install cedulauy-mrz-scan
# or
uv add cedulauy-mrz-scan
```

## Usage

```bash
cedulauy-mrz-scan --camera 0
cedulauy-mrz-scan --camera 0 --lang es  # on-screen texts in Spanish
cedulauy-mrz-scan --image card.jpg
```

```python
from cedulauy_mrz_scan import scan_camera, scan_image

mrz = scan_camera(0)         # None if the user quits; lang="es" or "en"
mrz = scan_image("card.jpg") # raises ValueError if unreadable
```

`read_mrz(image)` reads an already cropped BGR image of the code and returns
`(mrz or None, lines read)`.

The camera texts follow the system language (Spanish for `es` locales, English
otherwise) unless `--lang` / `lang=` is given.

## Development

```bash
uv sync            # create .venv and install dev dependencies
uv run pytest      # run tests
uv run ruff check  # lint
uv build           # build sdist and wheel into dist/
```

## License

GPL-3.0-or-later. The bundled OCR-B model (`tessdata/mrz.traineddata`) is from
[DoubangoTelecom/tesseractMRZ](https://github.com/DoubangoTelecom/tesseractMRZ), BSD-3-Clause.
