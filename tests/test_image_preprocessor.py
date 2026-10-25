from __future__ import annotations

import cv2
import numpy as np
import pymupdf
import pytest

from app.extraction.image_preprocessor import ImagePreprocessor


def _render_first_page(pdf_path, preprocessor):
    with pymupdf.open(pdf_path) as doc:
        return preprocessor.render_page(doc[0])


def test_one_image_per_page_in_order(make_pdf):
    pdf = make_pdf([["page one"], ["page two"], ["page three"]])
    pages = ImagePreprocessor(dpi=100).process_pdf(pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]


def test_render_size_follows_dpi(make_pdf):
    # A US Letter page is 8.5 x 11 inches.
    pdf = make_pdf([["x"]])
    page = ImagePreprocessor(dpi=200).process_pdf(pdf)[0]
    assert page.image.shape == (2200, 1700)


def test_output_is_pure_black_and_white(make_pdf):
    pdf = make_pdf([["Opening balance 5,000.00"]])
    image = ImagePreprocessor(dpi=150).process_pdf(pdf)[0].image
    assert image.dtype == np.uint8
    assert set(np.unique(image)) <= {0, 255}


def test_straight_page_is_not_rotated(make_pdf):
    pdf = make_pdf([[f"Line {i} of a statement with some text" for i in range(20)]])
    assert ImagePreprocessor(dpi=150).process_pdf(pdf)[0].skew_degrees == 0.0


@pytest.mark.parametrize("tilt", [-6.0, -2.5, 1.5, 4.0])
def test_estimates_and_corrects_skew(make_pdf, tilt):
    pdf = make_pdf([[f"Line {i} of a statement with some text" for i in range(20)]])
    pre = ImagePreprocessor(dpi=150)
    gray = _render_first_page(pdf, pre)

    # Simulate a crooked scan by tilting the page counter-clockwise by `tilt`.
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
    tilted = cv2.warpAffine(gray, matrix, (w, h), borderValue=255)

    assert pre.estimate_skew(tilted) == pytest.approx(tilt, abs=0.2)
    page = pre.process_image(tilted)
    assert page.skew_degrees == pytest.approx(tilt, abs=0.2)


def test_blank_page_has_no_skew():
    blank = np.full((500, 400), 255, dtype=np.uint8)
    assert ImagePreprocessor.estimate_skew(blank) == 0.0
