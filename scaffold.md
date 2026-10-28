# Scaffold

The plan for the Fintrace MVP: reading numbers off bank statements and turning them into structured, searchable transactions. We start with PDF, JPEG, PNG and HEIC statements (see [Current milestone](#current-milestone-pdf-jpeg-png-and-heic-ocr)). This file covers the OCR tech stack, the pipeline and the build order. Receipts, invoices, payroll, login, mobile and the frontend come later.

## Goal of the MVP

Upload a file(docx, pdf, doc, png, jpeg, heic) of a bank statement and get back a list of transactions (date, description, amount, balance), each linked to the source image. Every number should be checkable: if the extracted rows don't add up, we say so rather than silently store wrong data.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | The OCR and image-processing ecosystem is Python. Both of us can work in it and there is no second language to learn. |
| OCR engine | Tesseract via `pytesseract` | Free, runs locally, mature, and good on clean digital documents like statement screenshots. Documents never leave our machines, which matters for financial data. |
| Alternative OCR | PaddleOCR | Stronger on phone photos, skewed or blurry images, which we will need for receipts. Not needed for statement PNGs, so we keep it as a swap-in. |
| Image handling | OpenCV and Pillow | Grayscale, deskew and thresholding before OCR. These steps often raise accuracy more than changing the engine. |
| API | FastAPI and Pydantic | Small, fast to write, validates request and response shapes, and generates API docs for free. |
| Database | SQLite first, PostgreSQL later | SQLite needs no setup, so we can start in minutes. SQLAlchemy and Alembic keep the move to Postgres to a config change. |
| ORM and migrations | SQLAlchemy and Alembic | One model definition for both databases, with versioned schema changes. |
| Tests | pytest | Standard, and our parser can be tested without running OCR at all. |
| PDFs | PyMuPDF (`pymupdf`) | Renders each PDF page to an image for OCR. Installs with pip, with no system dependency like poppler. Text-based PDFs could later skip OCR and be read directly. |

### Why Tesseract and not an LLM vision API

A vision model (Claude, GPT) could read a statement in one call and is the strongest option for messy photos. We are not starting there because:

- **Privacy:** statements hold real account numbers and balances. Local OCR means nothing is sent to a third party while we are still building trust.
- **Cost:** per-document API fees add up for a product aimed at small businesses on thin margins. Tesseract costs nothing per page.
- **Predictability:** OCR gives words with bounding boxes and confidence scores, which we need for row reconstruction and for the low-confidence review feature. A model's answer is harder to verify.
- **It's our core tech:** this is the main feature, so we want to understand and own the pipeline instead of wrapping someone else's.

Because OCR sits behind an `OcrEngine` interface, we can add a vision-model engine later as a fallback for documents Tesseract scores poorly on, without touching the rest of the code.

## Pipeline

```
Upload -> ImageStore -> ImagePreprocessor -> OcrEngine -> StatementParser -> TransactionValidator -> Repositories -> API
```

1. **ImageStore:** saves the original file and dedupes by content hash, so the same statement uploaded twice is stored once. The original is kept so every transaction can link back to it.
2. **ImagePreprocessor:** grayscale, deskew, threshold. Statement screenshots are already clean, so this stays minimal for now.
3. **OcrEngine:** returns words, each with a bounding box and a confidence score. Using Tesseract's word-level output (`image_to_data`) instead of plain text is what makes table reconstruction possible.
4. **StatementParser:** groups words into rows by vertical position, assigns them to columns using the header's horizontal position, and converts each row into a transaction. This is the hardest part of the MVP and is plain code, so it is testable without OCR.
5. **TransactionValidator:** checks that opening balance plus the transactions equals the closing balance, and that each row's running balance follows from the previous one. A failed check flags the document for review. This is our main guard against OCR misreads such as `8` read as `3`.
6. **Repositories:** the only layer that talks to the database, so storage can change without touching the logic above it.
7. **API:** thin routes that call a `DocumentService`.

## Current milestone: PDF, JPEG, PNG and HEIC OCR

The first working slice of the pipeline: give it a PDF, JPEG, PNG or HEIC, get back all the text it sees.

**Scope**

- **Input:** PDF, JPEG, PNG and HEIC files only (built in that order). Other formats (DOCX, DOC) come later. Files are checked by their contents (the `%PDF-`, JPEG `FF D8 FF` or PNG `89 50 4E 47` signature, or for HEIC an `ftyp` box with a HEIC brand such as `heic`), not their extension.
- **Stages built:** the first three pipeline steps only, `ImageStore -> ImagePreprocessor -> OcrEngine`.
  - `ImageStore` saves the original to `data/uploads/<sha256>.pdf`, `.jpg`, `.png` or `.heic` and dedupes by content hash.
  - `ImagePreprocessor` turns the file into page images, then converts to grayscale, deskews (projection-profile search within ±10°) and thresholds (Otsu).
    - PDF: renders every page at 300 DPI with PyMuPDF.
    - JPEG, PNG and HEIC: one page. Applies the EXIF orientation so phone photos are upright, and upscales images narrower than 2000 px so small text is big enough for Tesseract.
    - PNG only: transparent areas (common in screenshots) are flattened onto white first, since a plain grayscale conversion would turn them black and hide black text.
    - HEIC only: decoded with pillow-heif (libheif bundled in the wheel, no system install). HEIC keeps its rotation in the file rather than EXIF, and pillow-heif applies it while decoding.
  - `OcrEngine` / `TesseractOcrEngine` returns every word with its bounding box, confidence and line, per page.
- **Output:** all recognized text, page by page. With `--json`, every word with its box and confidence, ready for the `StatementParser`.
- **Not in this milestone:** `StatementParser` and everything after it, any database or API, and mobile support (added later). The pipeline stops at the OCR output: parsing it into transactions will be done by a local AI model fine-tuned on invoices and receipts, which we will build separately.
- **Run locally** on our laptops through a command-line tool, `python -m app.cli`. See [Running it](#running-it).

**Code layout for this milestone**

```
app/
  storage/image_store.py               ImageStore (step 1)
  extraction/image_preprocessor.py     ImagePreprocessor (step 2)
  extraction/ocr_engine.py             OcrEngine interface, OcrWord, OcrPage (step 3)
  extraction/tesseract_ocr_engine.py   TesseractOcrEngine (step 3)
  pipeline.py                          OcrPipeline: runs steps 1-3 in order
  cli.py                               Command-line entry point
tests/                                 pytest tests for each stage
```

### Running it

One-time setup (macOS):

```sh
# 1. Install Homebrew if you don't have it (https://brew.sh), then Tesseract
brew install tesseract

# 2. From the repo folder, create a virtual environment and install packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

One-time setup (Windows, PowerShell):

```powershell
# 1. Install Tesseract with the UB Mannheim installer:
#    https://github.com/UB-Mannheim/tesseract/wiki
#    and tick "Add to PATH" (or add C:\Program Files\Tesseract-OCR to PATH)

# 2. From the repo folder
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Every time (after `source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` on Windows):

```sh
python -m app.cli path/to/statement.pdf          # print all text
python -m app.cli path/to/statement.jpg          # same for a JPEG
python -m app.cli path/to/statement.png          # or a PNG
python -m app.cli path/to/receipt.heic           # or an iPhone HEIC
python -m app.cli path/to/statement.pdf --json   # words with boxes and confidence
python -m pytest                                      # run the tests
```

## Decisions

- **Money is stored as integer cents,** never floats. `0.1 + 0.2` is not `0.3` in floating point, and bookkeeping can't tolerate that.
- **Each stage is its own class behind a small interface.** Stages can be built, tested and replaced one at a time, and the two of us can work on different stages without conflicts.
- **Keep per-word confidence** all the way through. It feeds the later "review low-confidence data" feature.
- **Original images are kept and linked** to every transaction they produced.

## Folders

```
app/
  models/         Document, Transaction
  storage/        ImageStore
  extraction/     ImagePreprocessor, OcrEngine, TesseractOcrEngine, StatementParser, TransactionValidator
  repositories/
  services/       DocumentService
  api/            routes
data/uploads/     git-ignored: holds real financial documents
samples/          fake or redacted statements only
tests/
```

## Build order

One class at a time, each with tests before moving on.

1. `Document` model
2. `Transaction` model
3. `ImageStore`
4. `ImagePreprocessor`
5. `OcrEngine` and `TesseractOcrEngine`
6. `StatementParser`
7. `TransactionValidator`
8. Repositories
9. `DocumentService`
10. API routes: `POST /documents`, `GET /documents/{id}`, `GET /transactions` (search)

## Before we start

- Install Tesseract and the Python packages: see [Running it](#running-it).
- Pick one bank and collect sample statement PDFs, JPEGs, PNGs and HEICs.
- **This repo is public.** Sample statements must be fake or redacted, and `data/uploads` must be in `.gitignore` from the first commit. Never commit real account numbers.

## Out of scope for the MVP

Receipts, invoice tracking, rent tracking, payroll, login and roles, mobile support, and the frontend. Each builds on this pipeline, so getting statement PDFs, JPEGs, PNGs and HEICs right first means the rest reuses it.
