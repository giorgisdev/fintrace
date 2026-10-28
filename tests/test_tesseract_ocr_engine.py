from __future__ import annotations

import shutil

import pytest

from app.extraction.image_preprocessor import ImagePreprocessor
from app.extraction.ocr_engine import OcrPage, OcrWord
from app.extraction.tesseract_ocr_engine import TesseractOcrEngine
from app.pipeline import OcrPipeline
from app.storage.image_store import ImageStore

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Tesseract is not installed"
)


def _tesseract_rows(rows):
    """Build the column-oriented dict image_to_data returns, from row tuples."""
    keys = ["level", "block_num", "par_num", "line_num", "left", "top", "width", "height", "conf", "text"]
    return {key: [row[i] for row in rows] for i, key in enumerate(keys)}


def test_parse_data_keeps_only_real_words():
    data = _tesseract_rows(
        [
            (1, 0, 0, 0, 0, 0, 2550, 3300, -1, ""),  # page row
            (4, 1, 1, 1, 10, 10, 300, 40, -1, ""),  # line row
            (5, 1, 1, 1, 10, 10, 120, 40, 96.5, "Opening"),
            (5, 1, 1, 1, 140, 10, 150, 40, 91.0, "balance"),
            (5, 1, 1, 1, 300, 10, 10, 40, -1, " "),  # empty gap
            (5, 1, 1, 2, 10, 60, 160, 40, 88.25, "5,000.00"),
        ]
    )
    words = TesseractOcrEngine.parse_data(data)

    assert [w.text for w in words] == ["Opening", "balance", "5,000.00"]
    assert words[2] == OcrWord("5,000.00", 88.25, 10, 60, 160, 40, line_id=(1, 1, 2))


def test_page_text_joins_words_into_lines():
    page = OcrPage(
        page_number=1,
        words=[
            OcrWord("Opening", 90, 0, 0, 1, 1, line_id=(1, 1, 1)),
            OcrWord("balance", 90, 0, 0, 1, 1, line_id=(1, 1, 1)),
            OcrWord("5,000.00", 60, 0, 0, 1, 1, line_id=(1, 1, 2)),
        ],
    )
    assert page.text == "Opening balance\n5,000.00"
    assert page.mean_confidence == pytest.approx(80)


@needs_tesseract
def test_end_to_end_reads_text_from_pdf(tmp_path, make_pdf):
    pdf = make_pdf([["Deposit Square payout 1,240.50"], ["Closing balance 5,557.43"]])
    pipeline = OcrPipeline(ImageStore(tmp_path / "uploads"), ImagePreprocessor(), TesseractOcrEngine())

    result = pipeline.run(pdf)

    assert len(result.pages) == 2
    assert "1,240.50" in result.pages[0].text
    assert "Closing balance 5,557.43" in result.pages[1].text
    assert all(w.confidence > 50 for p in result.pages for w in p.words)


@needs_tesseract
def test_end_to_end_reads_text_from_jpeg(tmp_path, make_jpeg):
    # 150 DPI is below MIN_IMAGE_WIDTH, so this also runs through upscaling.
    jpeg = make_jpeg(["Payroll J. Smith 1,600.00", "Closing balance 5,557.43"])
    pipeline = OcrPipeline(ImageStore(tmp_path / "uploads"), ImagePreprocessor(), TesseractOcrEngine())

    result = pipeline.run(jpeg)

    assert result.stored.path.suffix == ".jpg"
    assert len(result.pages) == 1
    assert "Payroll J. Smith 1,600.00" in result.pages[0].text
    assert "Closing balance 5,557.43" in result.pages[0].text


@needs_tesseract
def test_end_to_end_reads_text_from_png(tmp_path, make_png):
    png = make_png(["Hydro electricity 212.37", "Monthly account fee 15.95"])
    pipeline = OcrPipeline(ImageStore(tmp_path / "uploads"), ImagePreprocessor(), TesseractOcrEngine())

    result = pipeline.run(png)

    assert result.stored.path.suffix == ".png"
    assert len(result.pages) == 1
    assert "Hydro electricity 212.37" in result.pages[0].text
    assert "Monthly account fee 15.95" in result.pages[0].text


@needs_tesseract
def test_end_to_end_reads_text_from_heic(tmp_path, make_heic):
    heic = make_heic(["Rogers wireless 801.15", "Due date Nov 14 2026"])
    pipeline = OcrPipeline(ImageStore(tmp_path / "uploads"), ImagePreprocessor(), TesseractOcrEngine())

    result = pipeline.run(heic)

    assert result.stored.path.suffix == ".heic"
    assert len(result.pages) == 1
    assert "Rogers wireless 801.15" in result.pages[0].text
    assert "Due date Nov 14 2026" in result.pages[0].text
