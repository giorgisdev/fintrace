<img width="1024" height="254" alt="image" src="https://github.com/user-attachments/assets/908cf838-08cb-4a20-a991-7b3321b8a299" />

# Fintrace

> early development

**Vision:** Bookkeeping software for local businesses. Upload receipts, invoices, bank statements and other transactional documents. Fintrace extracts data from image files and PDFs to automate the process of calculating expenses and payroll.

## team

- **Adam Giorgis** ([@giorgisdev](https://github.com/giorgisdev)): University of British Columbia · CS '29
- **Yeon-Woo Doo** ([@YWD007](https://github.com/YWD007)): University of Toronto · Physics '29

## core focus

Helping businesses track numbers across scattered documents. Fintrace turns PDFs and images into searchable, structured transactions, linked to the original file.

## now building

**Fine-tuning Qwen3-0.6B with LoRA for transaction parsing**, trained on our own hand-labelled receipts, invoices and bills. OCR returns every word on the page but not which ones matter; the model picks out the ones that do.

- Reads the OCR text and returns structured fields: vendor, date, amount, order number
- Small enough to run on-device, so documents stay local
- Checks every value against the OCR output and sends low-confidence reads to the user for review; corrections become new training data

## early-stage features

- Document analysis: extract transactions from bank statements, receipts and invoices. Reads PDF, JPEG, PNG and HEIC. DOCX and DOC next.
- Invoice tracking: create, track and mark invoices paid

## status

Working today: a command-line tool that reads a PDF, JPEG, PNG or HEIC (iPhone photo) and prints all the text it recognizes, page by page, with each word's position and confidence. Files are identified by their contents, not their extension. It runs locally, so documents never leave your machine.

Next: transaction parsing, balance validation, a database and API, DOCX and DOC support, and a frontend. See [`scaffold.md`](scaffold.md) for the plan.

## tech stack

| | Choice | Why |
|---|---|---|
| Language | Python | The OCR and image-processing ecosystem is Python. |
| OCR | Tesseract (`pytesseract`) | Free, local and accurate on clean documents, so financial data stays on our machines. |
| Image processing | OpenCV, Pillow | Deskewing and thresholding often improve OCR more than changing the engine. |
| PDF rendering | PyMuPDF | Installs with pip, with no system dependencies. |
| HEIC decoding | pillow-heif | Opens iPhone photos with pip alone, no system install. |
| Tests | pytest | |
| Field extraction | Qwen3-0.6B, LoRA (PyTorch, Hugging Face Transformers) | Small enough for phones. Apache 2.0 licensed. |
| Planned | FastAPI, SQLite then PostgreSQL | Lightweight API; a database with no setup that can scale later. |

## getting started

Requires Python 3.11+ and [Tesseract](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` on macOS, the [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) on Windows).

```sh
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## usage

```sh
python -m app.cli path/to/statement.pdf          # print all recognized text
python -m app.cli path/to/receipt.jpg            # JPEG, PNG and HEIC work the same way
python -m app.cli path/to/statement.pdf --json   # every word with its box and confidence
python -m pytest                                 # run the tests
```

Uploaded files are copied to `data/uploads/`, which is git-ignored.
