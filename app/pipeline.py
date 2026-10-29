"""Runs the first three pipeline stages end to end.

Upload -> ImageStore -> ImagePreprocessor -> OcrEngine

An upload is one file or a batch of files (e.g. everything in a folder).

The later stages (StatementParser, TransactionValidator, Repositories, API)
will plug in after OCR. A DocumentService will eventually own this flow.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
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


@dataclass(frozen=True)
class BatchItem:
    """The outcome for one file in a batch: either a result or an error."""

    source: Path  # The file as the user gave it.
    result: OcrResult | None = None
    error: str | None = None  # Why the file failed, if it did.

    @property
    def ok(self) -> bool:
        return self.error is None


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

    def run_many(self, paths: Iterable[Path | str]) -> Iterator[BatchItem]:
        """Run each file through the pipeline, yielding results one at a time.

        Yielding (rather than returning a list at the end) lets the caller
        show progress and print each file's text as soon as it's ready.
        """
        for path in paths:
            try:
                yield BatchItem(source=Path(path), result=self.run(path))
            except Exception as error:  # noqa: BLE001
                # A corrupt or unreadable file shouldn't stop the whole batch,
                # so any failure is recorded against that file and we move on.
                yield BatchItem(source=Path(path), error=f"{type(error).__name__}: {error}")
