"""Finds the files to OCR when the input is a folder (bulk upload).

Files are picked by their contents, not their extension, using the same check
the ImageStore applies. So a folder can be dumped in as-is: anything that
isn't a PDF, JPEG, PNG or HEIC is skipped and reported, not treated as an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.storage.image_store import ImageStore


@dataclass(frozen=True)
class FolderScan:
    """What a folder scan found."""

    supported: list[Path] = field(default_factory=list)  # Files to OCR, in sorted order.
    skipped: list[Path] = field(default_factory=list)  # Files of an unsupported type.


def scan_folder(folder: Path | str, recursive: bool = True) -> FolderScan:
    """List the supported files in `folder`, including subfolders if `recursive`.

    Hidden files and folders (names starting with ".") are ignored entirely.
    That covers macOS clutter like .DS_Store and the "._" files it leaves on
    USB drives, which would otherwise show up as skipped every time.
    """
    folder = Path(folder)
    candidates = folder.rglob("*") if recursive else folder.iterdir()

    supported: list[Path] = []
    skipped: list[Path] = []
    # Sorted so files are processed in a predictable order (alphabetical by path).
    for path in sorted(candidates):
        relative = path.relative_to(folder)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if ImageStore.is_supported(path):
            supported.append(path)
        else:
            skipped.append(path)
    return FolderScan(supported=supported, skipped=skipped)
