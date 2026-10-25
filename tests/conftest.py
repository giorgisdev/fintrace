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
