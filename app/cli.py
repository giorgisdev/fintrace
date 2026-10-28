"""Command-line entry point for running OCR on a PDF, JPEG or PNG from your laptop.

Usage:
    python -m app.cli path/to/statement.pdf           # print the text
    python -m app.cli path/to/statement.pdf --json    # print words with boxes and confidence
    python -m app.cli path/to/statement.jpg           # JPEGs and PNGs work the same way
    python -m app.cli path/to/statement.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.extraction.image_preprocessor import DEFAULT_DPI, ImagePreprocessor
from app.extraction.tesseract_ocr_engine import (
    TesseractNotInstalledError,
    TesseractOcrEngine,
)
from app.pipeline import OcrPipeline, OcrResult
from app.storage.image_store import ImageStore, UnsupportedFileError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and read the command-line options."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Extract all text from a PDF, JPEG or PNG using local OCR (Tesseract).",
    )
    parser.add_argument("file", type=Path, help="Path to the PDF, JPEG or PNG file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output every word with its bounding box and confidence, as JSON",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI, help=f"PDF render resolution (default {DEFAULT_DPI})"
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path("data/uploads"),
        help="Where uploaded originals are kept (default data/uploads)",
    )
    return parser.parse_args(argv)


def to_json(result: OcrResult) -> str:
    """Serialize the full OCR result, including per-word boxes and confidence."""
    payload = {
        "file": {
            "sha256": result.stored.sha256,
            "original_name": result.stored.original_name,
            "stored_path": str(result.stored.path),
            "size_bytes": result.stored.size_bytes,
        },
        "pages": [
            {
                "page_number": page.page_number,
                "mean_confidence": round(page.mean_confidence, 1),
                "text": page.text,
                "words": [asdict(word) for word in page.words],
            }
            for page in result.pages
        ],
    }
    return json.dumps(payload, indent=2)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and print the result. Returns a process exit code."""
    args = parse_args(argv)

    if not args.file.is_file():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        return 1

    try:
        pipeline = OcrPipeline(
            store=ImageStore(args.store_dir),
            preprocessor=ImagePreprocessor(dpi=args.dpi),
            engine=TesseractOcrEngine(),
        )
        result = pipeline.run(args.file)
    except (TesseractNotInstalledError, UnsupportedFileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(to_json(result))
    else:
        print(result.text)
        # Summary goes to stderr so `> output.txt` captures only the text.
        status = "new" if result.stored.is_new else "already stored"
        print(
            f"\n[{len(result.pages)} page(s), file {status} as {result.stored.path}]",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
