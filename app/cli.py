"""Command-line entry point for running OCR on a PDF, JPEG, PNG or HEIC from your laptop.

Usage:
    python -m app.cli path/to/statement.pdf           # print the text
    python -m app.cli path/to/statement.pdf --json    # print words with boxes and confidence
    python -m app.cli path/to/statement.jpg           # JPEGs, PNGs and HEICs work the same way
    python -m app.cli path/to/statement.png
    python -m app.cli path/to/receipt.heic
    python -m app.cli path/to/folder                  # bulk: every supported file in the folder
    python -m app.cli a.pdf b.png path/to/folder      # any mix of files and folders
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
from app.pipeline import BatchItem, OcrPipeline, OcrResult
from app.storage.folder_scanner import scan_folder
from app.storage.image_store import ImageStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and read the command-line options."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description=(
            "Extract all text from PDF, JPEG, PNG or HEIC files using local OCR "
            "(Tesseract). Pass files, folders, or both."
        ),
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        metavar="path",
        help="A PDF, JPEG, PNG or HEIC file, or a folder of them",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output every word with its bounding box and confidence, as JSON",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="For folders, only take files directly inside, not in subfolders",
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


def collect_inputs(paths: list[Path], recursive: bool) -> tuple[list[Path], list[Path]]:
    """Expand folders into their supported files.

    Returns (files to OCR, skipped files). Files named directly are always
    attempted, so an unsupported one fails with a clear error rather than
    being silently skipped. Only files found inside folders can be skipped.
    """
    files: list[Path] = []
    skipped: list[Path] = []
    for path in paths:
        if path.is_dir():
            scan = scan_folder(path, recursive=recursive)
            files.extend(scan.supported)
            skipped.extend(scan.skipped)
        else:
            files.append(path)
    return files, skipped


def result_to_dict(result: OcrResult) -> dict:
    """The full OCR result for one file, including per-word boxes and confidence."""
    return {
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


def batch_to_json(items: list[BatchItem], skipped: list[Path]) -> str:
    """Serialize a batch: one entry per file (result or error), plus skipped files."""
    payload = {
        "files": [
            {"source": str(item.source), "error": item.error, "result": None}
            if not item.ok
            else {"source": str(item.source), "error": None, "result": result_to_dict(item.result)}
            for item in items
        ],
        "skipped": [str(path) for path in skipped],
    }
    return json.dumps(payload, indent=2)


def print_summary(items: list[BatchItem], skipped: list[Path]) -> None:
    """Print totals for a batch to stderr, listing anything that failed or was skipped."""
    succeeded = [item for item in items if item.ok]
    failed = [item for item in items if not item.ok]
    duplicates = sum(1 for item in succeeded if not item.result.stored.is_new)
    pages = sum(len(item.result.pages) for item in succeeded)

    print(
        f"\n[{len(succeeded)} file(s) read, {pages} page(s); "
        f"{duplicates} already stored; {len(failed)} failed; "
        f"{len(skipped)} skipped as unsupported]",
        file=sys.stderr,
    )
    for item in failed:
        print(f"  failed:  {item.source} ({item.error})", file=sys.stderr)
    for path in skipped:
        print(f"  skipped: {path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and print the results. Returns a process exit code."""
    args = parse_args(argv)

    missing = [path for path in args.paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"error: not found: {path}", file=sys.stderr)
        return 1

    files, skipped = collect_inputs(args.paths, recursive=not args.no_recursive)
    if not files:
        print("error: no PDF, JPEG, PNG or HEIC files found", file=sys.stderr)
        return 1

    try:
        pipeline = OcrPipeline(
            store=ImageStore(args.store_dir),
            preprocessor=ImagePreprocessor(dpi=args.dpi),
            engine=TesseractOcrEngine(),
        )
    except TesseractNotInstalledError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # A single file named directly keeps the simple output: just its text (or
    # its JSON object). Anything more is a batch, with a header per file.
    single_file = len(args.paths) == 1 and not args.paths[0].is_dir()

    items: list[BatchItem] = []
    for index, item in enumerate(pipeline.run_many(files), start=1):
        items.append(item)
        if single_file or args.json:
            continue
        # Progress and errors go to stderr, text to stdout, so `> output.txt`
        # captures only the extracted text.
        print(f"[{index}/{len(files)}] {item.source}", file=sys.stderr)
        if item.ok:
            print(f"===== {item.source} =====\n{item.result.text}\n")
        else:
            print(f"  error: {item.error}", file=sys.stderr)

    if single_file:
        item = items[0]
        if not item.ok:
            print(f"error: {item.error}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result_to_dict(item.result), indent=2))
        else:
            print(item.result.text)
            status = "new" if item.result.stored.is_new else "already stored"
            print(
                f"\n[{len(item.result.pages)} page(s), file {status} as {item.result.stored.path}]",
                file=sys.stderr,
            )
        return 0

    if args.json:
        print(batch_to_json(items, skipped))
    print_summary(items, skipped)
    # Exit code 1 if any file failed, so scripts can tell something went wrong.
    # Skipped files don't count: ignoring unsupported files is expected.
    return 0 if all(item.ok for item in items) else 1


if __name__ == "__main__":
    sys.exit(main())
