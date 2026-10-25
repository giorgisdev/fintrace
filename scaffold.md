# Scaffold

The plan for the Fintrace MVP: reading numbers off bank statement PNGs and turning them into structured, searchable transactions. This file covers the OCR tech stack, the pipeline and the build order. Receipts, invoices, PDFs, payroll, login and the frontend come later.

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
| PDFs (later) | PyMuPDF or pdfplumber | Text-based PDFs can skip OCR entirely and be read directly. Decide when we reach that milestone. |

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

- Install Tesseract (`brew install tesseract` on macOS, the UB Mannheim installer on Windows) and `pip install pytesseract`.
- Pick one bank and collect sample statement PNGs.
- **This repo is public.** Sample statements must be fake or redacted, and `data/uploads` must be in `.gitignore` from the first commit. Never commit real account numbers.

## Out of scope for the MVP

Receipts, invoice tracking, PDF support, rent tracking, payroll, login and roles, and the frontend. Each builds on this pipeline, so getting statement PNGs right first means the rest reuses it.
