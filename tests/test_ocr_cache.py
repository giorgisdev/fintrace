from __future__ import annotations

import json

import numpy as np
import pytest

from app import cli
from app.cli import main
from app.extraction.ocr_engine import OcrEngine, OcrPage, OcrWord
from app.pipeline import OcrResult
from app.storage.image_store import StoredFile
from app.storage.ocr_cache import OcrCache, OcrSettings, find_name_clashes

SETTINGS = OcrSettings(engine="fake", version="1.0", dpi=72)


class FakeOcrEngine(OcrEngine):
    """Stands in for Tesseract and counts how many pages it was asked to read."""

    name = "fake"
    version = "1.0"
    calls = 0

    def recognize(self, image: np.ndarray, page_number: int = 1) -> OcrPage:
        FakeOcrEngine.calls += 1
        return OcrPage(
            page_number,
            [
                OcrWord("Total", 90.0, 1, 2, 3, 4, line_id=(1, 1, 1)),
                OcrWord("$25.75", 80.0, 5, 6, 7, 8, line_id=(1, 1, 1)),
                OcrWord("Thanks", 70.0, 9, 10, 11, 12, line_id=(1, 1, 2)),
            ],
        )


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    """Runs the CLI on the fake engine, saving to tmp_path/ocr. Returns its exit code."""
    FakeOcrEngine.calls = 0
    monkeypatch.setattr(cli, "TesseractOcrEngine", FakeOcrEngine)

    def _run(*paths, dpi: int = 72) -> int:
        return main(
            [
                *map(str, paths),
                "--out", str(tmp_path / "ocr"),
                "--store-dir", str(tmp_path / "store"),
                "--dpi", str(dpi),
            ]
        )

    return _run


def _result() -> OcrResult:
    stored = StoredFile("abc", "stored/abc.pdf", "doc_001.pdf", 10, True)
    return OcrResult(stored, [FakeOcrEngine().recognize(np.zeros((1, 1)))])


def test_saves_lines_words_and_settings(tmp_path):
    path = OcrCache(tmp_path, SETTINGS).save("in/doc_001.pdf", _result())

    assert path == tmp_path / "doc_001.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["file"] == "doc_001.pdf"
    assert saved["sha256"] == "abc"
    assert saved["ocr"] == {"engine": "fake", "version": "1.0", "dpi": 72}
    page = saved["pages"][0]
    assert page["page_number"] == 1
    assert page["mean_confidence"] == 80.0
    assert [line["text"] for line in page["lines"]] == ["Total $25.75", "Thanks"]
    assert page["lines"][0]["confidence"] == 85.0
    assert page["lines"][0]["words"][1] == {"text": "$25.75", "confidence": 80.0, "box": [5, 6, 7, 8]}


def test_save_keeps_non_ascii_text_readable(tmp_path):
    result = _result()
    result.pages[0].words[0] = OcrWord("Électronique", 90.0, 1, 2, 3, 4, line_id=(1,))

    path = OcrCache(tmp_path, SETTINGS).save("doc_001.pdf", result)

    assert "Électronique" in path.read_text(encoding="utf-8")


def test_save_leaves_no_temp_file(tmp_path):
    OcrCache(tmp_path, SETTINGS).save("doc_001.pdf", _result())
    assert [p.name for p in tmp_path.iterdir()] == ["doc_001.json"]


def test_is_fresh_is_false_for_a_corrupt_saved_file(tmp_path, make_png):
    source = make_png(["a"])
    cache = OcrCache(tmp_path / "ocr", SETTINGS)
    cache.directory.mkdir()
    cache.path_for(source).write_text("{not json", encoding="utf-8")
    assert not cache.is_fresh(source)


def test_finds_name_clashes_ignoring_folder_extension_and_case(tmp_path):
    a, b, c = tmp_path / "x/doc_001.pdf", tmp_path / "y/DOC_001.png", tmp_path / "doc_002.pdf"
    assert find_name_clashes([a, b, c]) == [[a, b]]


def test_cli_writes_one_json_per_document(tmp_path, run_cli, make_pdf, make_png):
    folder = tmp_path / "upload"
    make_pdf([["a"], ["b"]], name="upload/doc_001.pdf")
    make_png(["c"], name="upload/doc_002.png")

    assert run_cli(folder) == 0

    saved = sorted(p.name for p in (tmp_path / "ocr").iterdir())
    assert saved == ["doc_001.json", "doc_002.json"]
    first = json.loads((tmp_path / "ocr" / "doc_001.json").read_text(encoding="utf-8"))
    assert [page["page_number"] for page in first["pages"]] == [1, 2]


def test_cli_prints_no_document_text_when_saving(run_cli, make_png, capsys):
    assert run_cli(make_png(["a"])) == 0
    assert capsys.readouterr().out == ""


def test_cli_skips_unchanged_files_on_a_second_run(tmp_path, run_cli, make_png, capsys):
    source = make_png(["a"])
    assert run_cli(source) == 0
    assert FakeOcrEngine.calls == 1
    capsys.readouterr()

    assert run_cli(source) == 0

    assert FakeOcrEngine.calls == 1
    assert "1 already saved" in capsys.readouterr().err


def test_cli_reads_again_when_the_file_changes(tmp_path, run_cli, make_png):
    source = make_png(["a"])
    assert run_cli(source) == 0
    make_png(["something else"])

    assert run_cli(source) == 0

    assert FakeOcrEngine.calls == 2


def test_cli_reads_again_when_settings_change(run_cli, make_png):
    source = make_png(["a"])
    assert run_cli(source, dpi=72) == 0

    assert run_cli(source, dpi=100) == 0

    assert FakeOcrEngine.calls == 2


def test_cli_stops_on_a_name_clash_before_reading_anything(tmp_path, run_cli, make_pdf, make_png, capsys):
    make_pdf([["a"]], name="upload/doc_001.pdf")
    make_png(["b"], name="upload/doc_001.png")

    assert run_cli(tmp_path / "upload") == 1

    assert FakeOcrEngine.calls == 0
    assert not (tmp_path / "ocr").exists()
    assert "same output file" in capsys.readouterr().err


def test_cli_treats_a_file_named_twice_as_one_file(tmp_path, run_cli, make_png):
    (tmp_path / "upload").mkdir()
    source = make_png(["a"], name="upload/doc_001.png")

    assert run_cli(source, tmp_path / "upload") == 0

    assert FakeOcrEngine.calls == 1


def test_cli_saves_nothing_for_a_failed_file_and_retries_it(tmp_path, run_cli, make_png):
    (tmp_path / "upload").mkdir()
    good = make_png(["a"], name="upload/good.png")
    broken = tmp_path / "upload" / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7 not actually a pdf")

    assert run_cli(good, broken) == 1
    assert [p.name for p in (tmp_path / "ocr").iterdir()] == ["good.json"]

    make_png(["a"], name="broken.png")
    broken.write_bytes((tmp_path / "broken.png").read_bytes())
    assert run_cli(good, broken) == 0

    assert FakeOcrEngine.calls == 2
    assert (tmp_path / "ocr" / "broken.json").exists()


def test_cli_rejects_json_with_out(tmp_path, make_png, capsys):
    with pytest.raises(SystemExit):
        main([str(make_png(["a"])), "--json", "--out", str(tmp_path / "ocr")])
    assert "not allowed with" in capsys.readouterr().err
