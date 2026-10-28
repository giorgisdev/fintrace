"""ImageStore: pipeline step 1.

Saves the original uploaded file and dedupes it by content hash, so the same
statement uploaded twice is stored once. The original is kept so every
transaction we extract later can link back to it.

PDFs, JPEGs, PNGs and HEICs are accepted for now. Other formats (DOCX, DOC) come later.
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
    b"\x89PNG\r\n\x1a\n": ".png",
}

# HEIC (iPhone photos) has no fixed signature at the start. It's an ISO media
# file: bytes 4-8 read "ftyp", followed by 4-byte "brands" naming the formats
# the file conforms to. MP4 videos and AVIF images use the same layout with
# other brands, so the brands are what identify HEIC. .heif uploads are HEIC too.
HEIC_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"hevm", b"hevs"}
HEIC_EXTENSION = ".heic"

SUPPORTED_EXTENSIONS = (*FILE_SIGNATURES.values(), HEIC_EXTENSION)

# Enough of the file's start to hold every signature and a typical ftyp box.
HEADER_BYTES = 64


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
        Raises UnsupportedFileError if the file is not a PDF, JPEG, PNG or HEIC.
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
            header = f.read(HEADER_BYTES)
        for signature, extension in FILE_SIGNATURES.items():
            if header.startswith(signature):
                return extension
        if ImageStore._is_heic(header):
            return HEIC_EXTENSION
        raise UnsupportedFileError(
            f"{path.name} is not a PDF, JPEG, PNG or HEIC. Only those are supported for now."
        )

    @staticmethod
    def _is_heic(header: bytes) -> bool:
        """True if the header is an ftyp box with a HEIC brand.

        Layout: box size (bytes 0-4), "ftyp" (4-8), major brand (8-12), minor
        version (12-16), then compatible brands to the end of the box. iPhones
        set the major brand to "heic"; other cameras may put "mif1" first and
        list "heic" among the compatible brands, so all of them are checked.
        """
        if header[4:8] != b"ftyp":
            return False
        box_end = min(int.from_bytes(header[0:4], "big"), len(header))
        brands = [header[8:12]] + [header[i : i + 4] for i in range(16, box_end, 4)]
        return any(brand in HEIC_BRANDS for brand in brands)

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA-256 of the file, read in chunks so large files don't fill memory."""
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
