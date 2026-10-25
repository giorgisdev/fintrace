"""ImageStore: pipeline step 1.

Saves the original uploaded file and dedupes it by content hash, so the same
statement uploaded twice is stored once. The original is kept so every
transaction we extract later can link back to it.

Only PDFs are accepted for now. Other formats (PNG, JPEG, HEIC, DOCX) come later.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

# Every PDF file starts with these bytes. Checking them is more reliable than
# trusting the file extension, which a user can rename freely.
PDF_MAGIC = b"%PDF-"


class UnsupportedFileError(ValueError):
    """Raised when an upload is not a file type we can process yet."""


@dataclass(frozen=True)
class StoredFile:
    """Where an uploaded file ended up, and whether it was new."""

    sha256: str  # Content hash; doubles as the file's stable ID.
    path: Path  # Location of the stored copy.
    original_name: str  # Filename as the user uploaded it.
    size_bytes: int
    is_new: bool  # False if an identical file was already stored.


class ImageStore:
    """Content-addressed storage for uploaded documents on the local disk."""

    def __init__(self, root: Path | str = "data/uploads") -> None:
        self.root = Path(root)

    def save(self, source: Path | str) -> StoredFile:
        """Copy the file at `source` into the store and return where it is.

        The stored copy is named after its SHA-256 hash, so two uploads with
        identical bytes map to the same file and the second one is not copied.
        Raises UnsupportedFileError if the file is not a PDF.
        """
        source = Path(source)
        self._check_is_pdf(source)

        digest = self._hash_file(source)
        destination = self.root / f"{digest}.pdf"

        is_new = not destination.exists()
        if is_new:
            self.root.mkdir(parents=True, exist_ok=True)
            # Copy to a temporary name first, then rename. A rename is atomic,
            # so a crash mid-copy never leaves a half-written file under the
            # final name (which would later be mistaken for a valid duplicate).
            temp = destination.with_suffix(".pdf.tmp")
            shutil.copyfile(source, temp)
            temp.replace(destination)

        return StoredFile(
            sha256=digest,
            path=destination,
            original_name=source.name,
            size_bytes=destination.stat().st_size,
            is_new=is_new,
        )

    def path_for(self, sha256: str) -> Path:
        """Return the stored path for a hash, or raise if it isn't stored."""
        path = self.root / f"{sha256}.pdf"
        if not path.exists():
            raise FileNotFoundError(f"No stored file with hash {sha256}")
        return path

    @staticmethod
    def _check_is_pdf(path: Path) -> None:
        """Reject anything that doesn't start with the PDF signature."""
        with path.open("rb") as f:
            header = f.read(len(PDF_MAGIC))
        if header != PDF_MAGIC:
            raise UnsupportedFileError(
                f"{path.name} is not a PDF. Only PDF files are supported for now."
            )

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA-256 of the file, read in chunks so large files don't fill memory."""
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
