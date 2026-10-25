"""ImagePreprocessor: pipeline step 2.

Turns each page of a PDF into a clean black-and-white image that Tesseract can
read well: render -> grayscale -> deskew -> threshold.

Statement PDFs are usually clean digital documents, so these steps stay
minimal. They matter more for scanned PDFs and, later, phone photos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pymupdf
import numpy as np

# Tesseract is tuned for text around 300 DPI. Lower DPI loses small digits;
# higher DPI is slower without improving accuracy much.
DEFAULT_DPI = 300

# Deskew searches for the page's tilt within +/- MAX_SKEW_DEGREES. A scanned
# page is rarely tilted more than that. Tilts smaller than MIN_SKEW_DEGREES are
# left alone, since rotating blurs the image slightly for no real gain.
MIN_SKEW_DEGREES = 0.2
MAX_SKEW_DEGREES = 10.0

# Skew is estimated on a downscaled copy of the page to keep it fast.
SKEW_ESTIMATE_WIDTH = 1000


@dataclass(frozen=True)
class PageImage:
    """One preprocessed PDF page, ready for OCR."""

    page_number: int  # 1-based, matching how people refer to pages.
    image: np.ndarray  # 2D uint8 array: 0 = black ink, 255 = white paper.
    skew_degrees: float  # Rotation that was corrected (0.0 if none).


class ImagePreprocessor:
    """Renders PDF pages to images and cleans them up for OCR."""

    def __init__(self, dpi: int = DEFAULT_DPI, deskew: bool = True) -> None:
        self.dpi = dpi
        self.deskew_enabled = deskew

    def process_pdf(self, pdf_path: Path | str) -> list[PageImage]:
        """Render and preprocess every page of the PDF, in page order."""
        pages: list[PageImage] = []
        with pymupdf.open(pdf_path) as doc:
            for index, page in enumerate(doc):
                gray = self.render_page(page)
                pages.append(self.process_image(gray, page_number=index + 1))
        return pages

    def render_page(self, page: pymupdf.Page) -> np.ndarray:
        """Rasterize one PDF page straight to an 8-bit grayscale array."""
        # PDF coordinates are in points (1/72 inch), so scale by dpi / 72.
        zoom = self.dpi / 72
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom),
            colorspace=pymupdf.csGRAY,
            alpha=False,
        )
        # The pixmap's raw bytes are row-major with one byte per pixel.
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width
        )

    def process_image(self, gray: np.ndarray, page_number: int = 1) -> PageImage:
        """Deskew and threshold an already-grayscale image."""
        skew = 0.0
        if self.deskew_enabled:
            skew = self.estimate_skew(gray)
            if abs(skew) >= MIN_SKEW_DEGREES:
                gray = self.rotate(gray, skew)
            else:
                skew = 0.0
        return PageImage(
            page_number=page_number,
            image=self.threshold(gray),
            skew_degrees=skew,
        )

    @classmethod
    def estimate_skew(cls, gray: np.ndarray) -> float:
        """Estimate how many degrees the text is tilted counter-clockwise.

        Uses a projection profile: try rotating the page by candidate angles
        and sum the ink in each pixel row. When the text lines are level, ink
        piles up in some rows (the lines) and others are empty (the gaps), so
        the row sums vary the most. The angle with the highest variance wins.
        A coarse pass finds the area, then a fine pass pins it down.
        """
        # Downscale for speed, then invert so ink is 1 and paper is 0.
        scale = min(1.0, SKEW_ESTIMATE_WIDTH / gray.shape[1])
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if not ink.any():  # Blank page: nothing to straighten.
            return 0.0

        def score(correction: float) -> float:
            # Rotate the ink by `correction` and measure how "striped" it is.
            rotated = cls._rotate_raw(ink, correction, border=0)
            return float(np.var(rotated.sum(axis=1)))

        coarse = np.arange(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES + 0.01, 0.5)
        best = max(coarse, key=score)
        fine = np.arange(best - 0.5, best + 0.51, 0.1)
        best = max(fine, key=score)
        # `best` is the rotation that levels the text, so the skew is its opposite.
        return round(-float(best), 2)

    @classmethod
    def rotate(cls, gray: np.ndarray, skew_degrees: float) -> np.ndarray:
        """Rotate the image to undo `skew_degrees` of tilt, filling corners white."""
        return cls._rotate_raw(gray, -skew_degrees, border=255)

    @staticmethod
    def _rotate_raw(image: np.ndarray, degrees: float, border: int) -> np.ndarray:
        """Rotate counter-clockwise by `degrees` about the center, same size."""
        height, width = image.shape
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    @staticmethod
    def threshold(gray: np.ndarray) -> np.ndarray:
        """Convert to pure black and white using Otsu's method.

        Otsu picks the cut-off between ink and paper automatically from the
        image's histogram, so it handles grey backgrounds and faded text
        without a hand-tuned value.
        """
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary
