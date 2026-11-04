"""OcrCache: saves OCR output to disk, one JSON file per document.

OCR is slow, so each document is read once and its result reused. The saved
file records the document's hash and the OCR settings that produced it, so a
changed file or a new engine version gets read again. Word boxes and confidence
are kept for the grounding check and low-confidence review; build_dataset.py
will read only the lines.

These files hold text from real financial documents. Never commit them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path

from app.extraction.ocr_engine import OcrPage, OcrWord
from app.pipeline import OcrResult
from app.storage.image_store import ImageStore


@dataclass(frozen=True)
class OcrSettings:
    """What produced a saved result. If any of these change, it's out of date."""

    engine: str
    version: str
    dpi: int


def find_name_clashes(files: list[Path]) -> list[list[Path]]:
    """Groups of files that would save to the same JSON, e.g. a.pdf and a.png.

    Case is ignored because Windows and macOS treat A.json and a.json as one file.
    """
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_stem[path.stem.casefold()].append(path)
    return [group for group in by_stem.values() if len(group) > 1]


class OcrCache:
    """A folder of saved OCR results, named after their input files."""

    def __init__(self, directory: Path | str, settings: OcrSettings) -> None:
        self.directory = Path(directory)
        self.settings = settings

    def path_for(self, source: Path | str) -> Path:
        """Where the result for this input file is saved."""
        return self.directory / f"{Path(source).stem}.json"

    def is_fresh(self, source: Path | str) -> bool:
        """True if a saved result exists for this exact file and these settings."""
        try:
            saved = json.loads(self.path_for(source).read_text(encoding="utf-8"))
            return saved["sha256"] == ImageStore.hash_file(Path(source)) and saved["ocr"] == asdict(
                self.settings
            )
        except (OSError, ValueError, KeyError, TypeError):
            # Missing, unreadable or corrupt: treat it as not saved and read it again.
            return False

    def save(self, source: Path | str, result: OcrResult) -> Path:
        """Write the result for one input file and return where it went."""
        path = self.path_for(source)
        payload = {
            "file": Path(source).name,
            "sha256": result.stored.sha256,
            "ocr": asdict(self.settings),
            "pages": [_page_to_dict(page) for page in result.pages],
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name first, then rename, so a crash mid-write
        # never leaves a half-written file under the final name.
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
        return path


def _page_to_dict(page: OcrPage) -> dict:
    """One page, with its words grouped into lines of text."""
    return {
        "page_number": page.page_number,
        "mean_confidence": round(page.mean_confidence, 1),
        "lines": [
            _line_to_dict(list(words)) for _, words in groupby(page.words, key=lambda w: w.line_id)
        ],
    }


def _line_to_dict(words: list[OcrWord]) -> dict:
    """One line of text, its average confidence, and the words that make it up."""
    return {
        "text": " ".join(word.text for word in words),
        "confidence": round(sum(word.confidence for word in words) / len(words), 1),
        "words": [
            {
                "text": word.text,
                "confidence": word.confidence,
                "box": [word.left, word.top, word.width, word.height],
            }
            for word in words
        ],
    }
