"""ImageStore: pipeline step 1.

Saves the original uploaded file and dedupes it by content hash, so the same
statement uploaded twice is stored once. The original is kept so every
transaction we extract later can link back to it.

PDFs and JPEGs are accepted for now. Other formats (PNG, HEIC, DOCX) come later.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

# Each supported format starts with a fixed byte signature. Checking it is more
# reliable than trusting the file extension, which a user can rename freely.
# Maps signature -> extension the stored copy is saved with.
FILE_SIGNATURES = {
    b"%PDF-": ".pdf",
    b"\xff\xd8\xff": ".jpg",  # JPEG; .jpeg and .jpg uploads are both stored as .jpg
}
SUPPORTED_EXTENSIONS = tuple(FILE_SIGNATURES.values())


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
        Raises UnsupportedFileError if the file is not a PDF or JPEG.
        """
        source = Path(source)
        extension = self._detect_extension(source)

        digest = self._hash_file(source)
        destination = self.root / f"{digest}{extension}"

        is_new = not destination.exists()
        if is_new:
            self.root.mkdir(parents=True, exist_ok=True)
            # Copy to a temporary name first, then rename. A rename is atomic,
            # so a crash mid-copy never leaves a half-written file under the
            # final name (which would later be mistaken for a valid duplicate).
            temp = destination.with_suffix(extension + ".tmp")
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
        # The hash alone doesn't say which format it was, so try each one.
        for extension in SUPPORTED_EXTENSIONS:
            path = self.root / f"{sha256}{extension}"
            if path.exists():
                return path
        raise FileNotFoundError(f"No stored file with hash {sha256}")

    @staticmethod
    def _detect_extension(path: Path) -> str:
        """Identify the format from the file's first bytes, or reject it."""
        with path.open("rb") as f:
            header = f.read(max(len(sig) for sig in FILE_SIGNATURES))
        for signature, extension in FILE_SIGNATURES.items():
            if header.startswith(signature):
                return extension
        raise UnsupportedFileError(
            f"{path.name} is not a PDF or JPEG. Only those are supported for now."
        )

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA-256 of the file, read in chunks so large files don't fill memory."""
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
