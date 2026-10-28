"""Runs the first three pipeline stages end to end.

Upload -> ImageStore -> ImagePreprocessor -> OcrEngine

The later stages (StatementParser, TransactionValidator, Repositories, API)
will plug in after OCR. A DocumentService will eventually own this flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.extraction.image_preprocessor import ImagePreprocessor
from app.extraction.ocr_engine import OcrEngine, OcrPage
from app.storage.image_store import ImageStore, StoredFile


@dataclass(frozen=True)
class OcrResult:
    """Everything the OCR stages produced for one uploaded PDF, JPEG, PNG or HEIC."""

    stored: StoredFile
    pages: list[OcrPage]

    @property
    def text(self) -> str:
        """All pages' text, separated by page markers."""
        return "\n\n".join(
            f"--- Page {page.page_number} ---\n{page.text}" for page in self.pages
        )


class OcrPipeline:
    """Wires the stages together. Each stage is injected so it can be swapped."""

    def __init__(
        self,
        store: ImageStore,
        preprocessor: ImagePreprocessor,
        engine: OcrEngine,
    ) -> None:
        self.store = store
        self.preprocessor = preprocessor
        self.engine = engine

    def run(self, path: Path | str) -> OcrResult:
        """Store the PDF, JPEG, PNG or HEIC, preprocess each page, and OCR it."""
        stored = self.store.save(path)
        # Work from the stored copy, so what we OCR is exactly what we kept.
        # Its extension reflects the real format, so process_file can rely on it.
        page_images = self.preprocessor.process_file(stored.path)
        pages = [
            self.engine.recognize(page.image, page_number=page.page_number)
            for page in page_images
        ]
        return OcrResult(stored=stored, pages=pages)
