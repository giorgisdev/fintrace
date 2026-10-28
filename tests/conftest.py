"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path: Path):
    """Factory that writes a small PDF with the given lines of text, one page per list."""

    def _make(pages: list[list[str]], name: str = "doc.pdf") -> Path:
        doc = pymupdf.open()
        for lines in pages:
            page = doc.new_page(width=612, height=792)
            for i, line in enumerate(lines):
                page.insert_text((72, 100 + i * 24), line, fontsize=14)
        path = tmp_path / name
        doc.save(path)
        return path

    return _make


@pytest.fixture
def make_jpeg(tmp_path: Path, make_pdf):
    """Factory that writes a one-page JPEG of the given lines of text."""

    def _make(lines: list[str], name: str = "doc.jpg", dpi: int = 150) -> Path:
        pdf = make_pdf([lines], name=f"{name}.source.pdf")
        path = tmp_path / name
        with pymupdf.open(pdf) as doc:
            doc[0].get_pixmap(dpi=dpi).save(path, jpg_quality=90)
        return path

    return _make


@pytest.fixture
def make_png(tmp_path: Path, make_pdf):
    """Factory that writes a one-page PNG of the given lines of text."""

    def _make(lines: list[str], name: str = "doc.png", dpi: int = 150) -> Path:
        pdf = make_pdf([lines], name=f"{name}.source.pdf")
        path = tmp_path / name
        with pymupdf.open(pdf) as doc:
            doc[0].get_pixmap(dpi=dpi).save(path)
        return path

    return _make
