"""OcrEngine: pipeline step 3 (the interface).

An OCR engine takes a page image and returns every word it sees, each with a
bounding box and a confidence score. The rest of the pipeline depends only on
this interface, so Tesseract can later be swapped for PaddleOCR or a vision
model without touching anything downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class OcrWord:
    """A single recognized word and where it sits on the page (in pixels)."""

    text: str
    confidence: float  # 0-100. Kept all the way through for low-confidence review.
    left: int
    top: int
    width: int
    height: int
    # Which line the engine grouped this word into. Words with the same
    # line_id are on the same line of text. The StatementParser will do its own
    # row grouping by position, so this is only used to print readable text.
    line_id: tuple[int, ...] = ()


@dataclass(frozen=True)
class OcrPage:
    """All the words found on one page."""

    page_number: int
    words: list[OcrWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The page's words joined back into lines of plain text."""
        lines: list[str] = []
        current_line: tuple[int, ...] | None = None
        for word in self.words:
            if word.line_id != current_line:
                lines.append(word.text)
                current_line = word.line_id
            else:
                lines[-1] += " " + word.text
        return "\n".join(lines)

    @property
    def mean_confidence(self) -> float:
        """Average word confidence, or 0.0 for a page with no words."""
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)


class OcrEngine(ABC):
    """Interface every OCR backend implements."""

    @abstractmethod
    def recognize(self, image: np.ndarray, page_number: int = 1) -> OcrPage:
        """Read all the words in a preprocessed page image."""
