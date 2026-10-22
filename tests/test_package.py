import os
import shutil

import numpy as np
import pytest

import cedulauy_mrz_scan
from cedulauy_mrz_scan import detect, i18n, ocr

# ICAO 9303 part 5 specimen
TD1 = [
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
]


def test_version() -> None:
    assert cedulauy_mrz_scan.__version__


def test_model_is_packaged() -> None:
    assert (ocr.TESSDATA_DIR / "mrz.traineddata").is_file()


def test_validate_td1() -> None:
    assert ocr.validate(["NOISE<<LINE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", *TD1]) == "".join(TD1)


def test_validate_pads_short_lines() -> None:
    assert ocr.validate([TD1[0], TD1[1], TD1[2].rstrip("<")]) == "".join(TD1)


def test_validate_rejects_bad_check_digit() -> None:
    assert ocr.validate([TD1[0].replace("7<", "8<"), TD1[1], TD1[2]]) is None


def test_fix_filler() -> None:
    assert ocr._fix_filler("ERIKSSON<<ANNA<<K<<") == "ERIKSSON<<ANNA<<<<<"
    assert ocr._fix_filler("DOE<<J<SMITH") == "DOE<<J<SMITH"


def test_guide_box_keeps_card_aspect() -> None:
    x1, y1, x2, y2 = detect.guide_box((720, 1280, 3))
    assert (x2 - x1) / (y2 - y1) == pytest.approx(detect.CARD_ASPECT, rel=0.01)


def test_empty_frame_asks_for_card() -> None:
    frame = np.full((720, 1280, 3), 128, np.uint8)
    box = detect.guide_box(frame.shape)
    lines, _ = detect.find_mrz(frame, box)
    assert detect.check_placement(lines, box, None) == detect.PLACE_CARD


def test_tesseract_handles_path_with_spaces(tmp_path, monkeypatch) -> None:
    try:
        ocr.find_tesseract()
    except RuntimeError:
        pytest.skip("tesseract not installed")
    tessdata = tmp_path / "tess data ñ"
    shutil.copytree(ocr.TESSDATA_DIR, tessdata)
    options = ocr.TESSERACT_OPTIONS.copy()
    options[options.index("--tessdata-dir") + 1] = str(tessdata)
    monkeypatch.setattr(ocr, "TESSERACT_OPTIONS", options)
    assert ocr.read_mrz(np.full((100, 600, 3), 255, np.uint8)) == (None, [])


@pytest.mark.parametrize(
    ("name", "lang"),
    [("es_UY", "es"), ("es-ES", "es"), ("en_US", "en"), ("", "en")],
)
def test_system_language(monkeypatch, name, lang) -> None:
    monkeypatch.setattr(i18n, "_locale_name", lambda: name)
    assert i18n.system_language() == lang


def test_real_system_language() -> None:
    lang = i18n.system_language()
    assert lang in i18n.LANGUAGES
    assert lang == os.environ.get("EXPECTED_LANG", lang)


def test_translate() -> None:
    assert i18n.translate(detect.PLACE_CARD, "es") == "Coloque la cédula dentro del recuadro"
    assert i18n.translate(detect.PLACE_CARD, "en") == detect.PLACE_CARD
    assert i18n.translate("", "es") == ""


def test_every_translation_is_used() -> None:
    package = ocr.TESSDATA_DIR.parent
    source = "".join(p.read_text("utf-8") for p in package.glob("*.py") if p.name != "i18n.py")
    assert [text for text in i18n.SPANISH if f'"{text}"' not in source] == []
