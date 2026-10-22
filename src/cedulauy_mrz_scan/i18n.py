# SPDX-License-Identifier: GPL-3.0-or-later
"""Language of the on-screen texts: English, or Spanish for "es" locales."""

from __future__ import annotations

import ctypes
import locale
import sys

LANGUAGES = ("en", "es")

SPANISH = {
    "Place the card inside the box": "Coloque la cédula dentro del recuadro",
    "Turn the card around (code at the bottom)": "Dé vuelta la cédula (código abajo)",
    "Keep the code (bottom lines) visible": "Mantenga visible el código (líneas de abajo)",
    "Move the card closer": "Acerque la cédula",
    "Move the card back": "Aleje la cédula",
    "Center the card in the box": "Centre la cédula en el recuadro",
    "Straighten the card": "Enderece la cédula",
    "Hold still": "No se mueva",
    "Image blurry, hold still": "Imagen borrosa, no se mueva",
    "The two pictures gave different reads, try again": (
        "Las dos fotos dieron lecturas distintas, intente de nuevo"
    ),
    "Could not read the code, try again (better light helps)": (
        "No se pudo leer el código, intente de nuevo (con mejor luz)"
    ),
    "Reading the code...": "Leyendo el código...",
    "Pictures taken, you can lower the card": "Fotos tomadas, puede bajar la cédula",
    "MRZ read successfully": "Código leído correctamente",
    "Scan your ID card": "Escanee su cédula",
    "Back side, code at the bottom, inside the frame": "Dorso, código abajo, dentro del recuadro",
    "Use good, even lighting and avoid reflections": "Use buena luz, pareja y sin reflejos",
    "Q  quit": "Q  salir",
}


def _locale_name() -> str:
    if sys.platform == "win32":
        buffer = ctypes.create_unicode_buffer(85)  # LOCALE_NAME_MAX_LENGTH
        ctypes.windll.kernel32.GetUserDefaultLocaleName(buffer, len(buffer))
        return buffer.value
    return locale.getlocale()[0] or ""


def system_language() -> str:
    return "es" if _locale_name().lower().startswith("es") else "en"


def translate(text: str, lang: str) -> str:
    return SPANISH.get(text, text) if lang == "es" else text
