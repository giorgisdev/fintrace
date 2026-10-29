from __future__ import annotations

import numpy as np

from app.cli import collect_inputs, main
from app.extraction.image_preprocessor import ImagePreprocessor
from app.extraction.ocr_engine import OcrEngine, OcrPage, OcrWord
from app.pipeline import OcrPipeline
from app.storage.folder_scanner import scan_folder
from app.storage.image_store import ImageStore


class FakeOcrEngine(OcrEngine):
    """Stands in for Tesseract so batch behaviour can be tested without it."""

    def recognize(self, image: np.ndarray, page_number: int = 1) -> OcrPage:
        return OcrPage(page_number, [OcrWord("text", 99.0, 0, 0, 1, 1, line_id=(1,))])


def _mixed_folder(tmp_path, make_pdf, make_jpeg, make_png, make_heic):
    """A folder like a user might drop in: all formats, a subfolder and clutter."""
    folder = tmp_path / "upload"
    (folder / "sub").mkdir(parents=True)
    make_pdf([["a"]], name="upload/statement.pdf")
    make_jpeg(["b"], name="upload/receipt.jpg")
    make_png(["c"], name="upload/sub/invoice.png")
    make_heic(["d"], name="upload/sub/photo.heic")
    (folder / "notes.txt").write_text("not a document")
    (folder / ".DS_Store").write_bytes(b"\0\0\0\1Bud1")
    (folder / ".hidden").mkdir()
    make_png(["e"], name="upload/.hidden/secret.png")
    return folder


def test_scan_finds_supported_files_in_subfolders(tmp_path, make_pdf, make_jpeg, make_png, make_heic):
    folder = _mixed_folder(tmp_path, make_pdf, make_jpeg, make_png, make_heic)

    scan = scan_folder(folder)

    names = [p.relative_to(folder).as_posix() for p in scan.supported]
    assert names == ["receipt.jpg", "statement.pdf", "sub/invoice.png", "sub/photo.heic"]
    assert [p.name for p in scan.skipped] == ["notes.txt"]


def test_scan_ignores_hidden_files_and_folders(tmp_path, make_pdf, make_jpeg, make_png, make_heic):
    folder = _mixed_folder(tmp_path, make_pdf, make_jpeg, make_png, make_heic)
    scan = scan_folder(folder)
    every_path = scan.supported + scan.skipped
    assert not any(part.startswith(".") for p in every_path for part in p.relative_to(folder).parts)


def test_scan_without_recursion_stays_in_top_folder(tmp_path, make_pdf, make_jpeg, make_png, make_heic):
    folder = _mixed_folder(tmp_path, make_pdf, make_jpeg, make_png, make_heic)
    scan = scan_folder(folder, recursive=False)
    assert all(p.parent == folder for p in scan.supported)


def test_scan_detects_type_by_content_not_extension(tmp_path, make_png):
    folder = tmp_path / "upload"
    folder.mkdir()
    disguised = folder / "scan.dat"
    disguised.write_bytes(make_png(["x"]).read_bytes())
    (folder / "fake.pdf").write_text("just text")

    scan = scan_folder(folder)

    assert scan.supported == [disguised]
    assert scan.skipped == [folder / "fake.pdf"]


def test_collect_inputs_mixes_files_and_folders(tmp_path, make_pdf, make_png):
    single = make_pdf([["a"]], name="single.pdf")
    folder = tmp_path / "folder"
    folder.mkdir()
    in_folder = make_png(["b"], name="folder/page.png")
    (folder / "readme.txt").write_text("x")

    files, skipped = collect_inputs([single, folder], recursive=True)

    assert files == [single, in_folder]
    assert skipped == [folder / "readme.txt"]


def test_run_many_continues_past_a_broken_file(tmp_path, make_pdf, make_png):
    good_pdf = make_pdf([["a"], ["b"]], name="good.pdf")
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7 not actually a pdf")
    good_png = make_png(["c"])
    pipeline = OcrPipeline(ImageStore(tmp_path / "store"), ImagePreprocessor(dpi=72), FakeOcrEngine())

    items = list(pipeline.run_many([good_pdf, broken, good_png]))

    assert [item.ok for item in items] == [True, False, True]
    assert [item.source for item in items] == [good_pdf, broken, good_png]
    assert len(items[0].result.pages) == 2
    assert items[1].result is None and items[1].error


def test_run_many_marks_duplicates_as_already_stored(tmp_path, make_png):
    original = make_png(["a"], name="one.png")
    copy = tmp_path / "two.png"
    copy.write_bytes(original.read_bytes())
    pipeline = OcrPipeline(ImageStore(tmp_path / "store"), ImagePreprocessor(dpi=72), FakeOcrEngine())

    first, second = pipeline.run_many([original, copy])

    assert first.result.stored.is_new
    assert not second.result.stored.is_new
    assert first.result.stored.path == second.result.stored.path


def test_cli_reports_missing_path(tmp_path, capsys):
    assert main([str(tmp_path / "nope")]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_reports_folder_with_nothing_supported(tmp_path, capsys):
    (tmp_path / "notes.txt").write_text("x")
    assert main([str(tmp_path)]) == 1
    assert "no PDF, JPEG, PNG or HEIC files found" in capsys.readouterr().err
