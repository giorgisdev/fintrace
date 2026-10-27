from __future__ import annotations

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image

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


def test_jpeg_becomes_one_black_and_white_page(make_jpeg):
    jpeg = make_jpeg(["Opening balance 5,000.00"], dpi=250)  # 2125 px wide
    pages = ImagePreprocessor().process_jpeg(jpeg)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].image.shape == (2750, 2125)  # Already wide enough: not resized.
    assert set(np.unique(pages[0].image)) <= {0, 255}


def test_small_jpeg_is_upscaled(make_jpeg):
    jpeg = make_jpeg(["Opening balance 5,000.00"], dpi=100)  # 850 x 1100 px
    page = ImagePreprocessor().process_jpeg(jpeg)[0]
    assert page.image.shape[1] == 2000
    assert page.image.shape[0] == pytest.approx(1100 * 2000 / 850, abs=1)


def test_jpeg_exif_rotation_is_applied(tmp_path):
    # A landscape-shaped image tagged "rotate 90 degrees", like a phone photo.
    path = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # EXIF Orientation tag: rotate 90 degrees clockwise to view.
    Image.new("L", (300, 200), color=255).save(path, exif=exif)

    gray = ImagePreprocessor.load_image(path)

    assert gray.shape == (300, 200)  # Now portrait: height 300, width 200.


def test_process_file_picks_method_by_extension(make_pdf, make_jpeg):
    pre = ImagePreprocessor(dpi=100)
    assert len(pre.process_file(make_pdf([["a"], ["b"]]))) == 2
    assert len(pre.process_file(make_jpeg(["a"]))) == 1
    with pytest.raises(ValueError):
        pre.process_file("notes.txt")


def test_blank_page_has_no_skew():
    blank = np.full((500, 400), 255, dtype=np.uint8)
    assert ImagePreprocessor.estimate_skew(blank) == 0.0
