"""TesseractOcrEngine: the OcrEngine implementation backed by Tesseract."""

from __future__ import annotations

import shutil

import numpy as np
import pytesseract

from app.extraction.ocr_engine import OcrEngine, OcrPage, OcrWord

# Tesseract's image_to_data reports one row per layout element, at these levels.
# Only level 5 rows are actual words; the rest are pages, blocks, paragraphs
# and lines with no text of their own.
WORD_LEVEL = 5


class TesseractNotInstalledError(RuntimeError):
    """Raised when the Tesseract program can't be found on this machine."""


class TesseractOcrEngine(OcrEngine):
    """Runs Tesseract locally. Documents never leave the machine."""

    def __init__(self, language: str = "eng", page_segmentation_mode: int = 6) -> None:
        # Page segmentation mode 6 = "assume a single uniform block of text".
        # It keeps each table row on one line, which suits statements better
        # than the default (3), which tends to split columns into separate blocks.
        self.language = language
        self.config = f"--psm {page_segmentation_mode}"
        self._check_installed()

    def recognize(self, image: np.ndarray, page_number: int = 1) -> OcrPage:
        """Run Tesseract on one page and return its words with boxes and scores."""
        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=self.config,
            output_type=pytesseract.Output.DICT,
        )
        return OcrPage(page_number=page_number, words=self.parse_data(data))

    @staticmethod
    def parse_data(data: dict[str, list]) -> list[OcrWord]:
        """Convert Tesseract's column-oriented output into a list of OcrWords.

        `data` maps each column name (text, conf, left, ...) to a list with one
        entry per row. Kept separate from `recognize` so it can be tested
        without running Tesseract.
        """
        words: list[OcrWord] = []
        for i in range(len(data["text"])):
            text = str(data["text"][i]).strip()
            confidence = float(data["conf"][i])
            # Skip non-word rows, and the empty rows Tesseract emits with
            # confidence -1 for gaps between words.
            if data["level"][i] != WORD_LEVEL or not text or confidence < 0:
                continue
            words.append(
                OcrWord(
                    text=text,
                    confidence=confidence,
                    left=int(data["left"][i]),
                    top=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                    line_id=(
                        int(data["block_num"][i]),
                        int(data["par_num"][i]),
                        int(data["line_num"][i]),
                    ),
                )
            )
        return words

    @staticmethod
    def _check_installed() -> None:
        """Fail early with install instructions instead of a cryptic error later."""
        if shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
            raise TesseractNotInstalledError(
                "Tesseract is not installed or not on your PATH.\n"
                "  macOS:   brew install tesseract\n"
                "  Windows: install from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "See scaffold.md for details."
            )
