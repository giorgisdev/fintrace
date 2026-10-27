from __future__ import annotations

import pytest

from app.storage.image_store import ImageStore, UnsupportedFileError


def test_save_copies_pdf_named_by_hash(tmp_path, make_pdf):
    pdf = make_pdf([["hello"]])
    store = ImageStore(tmp_path / "uploads")

    stored = store.save(pdf)

    assert stored.is_new
    assert stored.path == tmp_path / "uploads" / f"{stored.sha256}.pdf"
    assert stored.path.read_bytes() == pdf.read_bytes()
    assert stored.original_name == "doc.pdf"


def test_same_content_is_stored_once(tmp_path, make_pdf):
    pdf = make_pdf([["hello"]])
    copy = tmp_path / "renamed.pdf"
    copy.write_bytes(pdf.read_bytes())
    store = ImageStore(tmp_path / "uploads")

    first = store.save(pdf)
    second = store.save(copy)

    assert not second.is_new
    assert second.path == first.path
    assert len(list((tmp_path / "uploads").iterdir())) == 1


def test_different_content_gets_different_hash(tmp_path, make_pdf):
    store = ImageStore(tmp_path / "uploads")
    a = store.save(make_pdf([["one"]], name="a.pdf"))
    b = store.save(make_pdf([["two"]], name="b.pdf"))
    assert a.sha256 != b.sha256


def test_rejects_non_pdf_even_with_pdf_extension(tmp_path):
    fake = tmp_path / "statement.pdf"
    fake.write_bytes(b"\x89PNG\r\n\x1a\n not really a pdf")
    with pytest.raises(UnsupportedFileError):
        ImageStore(tmp_path / "uploads").save(fake)


def test_jpeg_is_stored_with_jpg_extension(tmp_path, make_jpeg):
    jpeg = make_jpeg(["hello"], name="photo.jpeg")
    store = ImageStore(tmp_path / "uploads")

    stored = store.save(jpeg)

    assert stored.path.suffix == ".jpg"
    assert stored.path.read_bytes() == jpeg.read_bytes()
    assert store.path_for(stored.sha256) == stored.path


def test_jpeg_is_detected_by_content_not_extension(tmp_path, make_jpeg):
    # A JPEG mislabelled as .pdf is still stored as what it really is.
    misnamed = tmp_path / "statement.pdf"
    misnamed.write_bytes(make_jpeg(["hello"]).read_bytes())
    assert ImageStore(tmp_path / "uploads").save(misnamed).path.suffix == ".jpg"


def test_path_for_unknown_hash_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageStore(tmp_path).path_for("0" * 64)
