"""Generates samples/sample_statement.pdf: a FAKE bank statement for testing.

All names, account numbers and amounts are made up. Run from the repo root:
    python samples/make_sample_statement.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

OUTPUT = Path(__file__).parent / "sample_statement.pdf"

HEADER = [
    "MAPLE TRUST BANK",
    "Account Statement - Business Chequing",
    "Account holder: Example Bakery Ltd.    Account: 0000-000000-00",
    "Statement period: Sep 01, 2026 - Sep 30, 2026",
    "",
]

# (date, description, withdrawal, deposit, balance). Balances add up correctly
# so this file stays useful once the TransactionValidator exists.
ROWS = [
    ("Sep 01", "Opening balance", "", "", "5,000.00"),
    ("Sep 03", "Deposit - Square payout", "", "1,240.50", "6,240.50"),
    ("Sep 05", "Flour Supply Co. invoice 1182", "830.00", "", "5,410.50"),
    ("Sep 12", "Hydro electricity", "212.37", "", "5,198.13"),
    ("Sep 15", "Payroll - J. Smith", "1,600.00", "", "3,598.13"),
    ("Sep 20", "Deposit - Square payout", "", "1,975.25", "5,573.38"),
    ("Sep 28", "Monthly account fee", "15.95", "", "5,557.43"),
    ("Sep 30", "Closing balance", "", "", "5,557.43"),
]

# Left edge of each column, in points. Amount columns are right-aligned to
# the next column's edge, like a real statement.
COLUMNS = [("Date", 50), ("Description", 110), ("Withdrawals", 330), ("Deposits", 420), ("Balance", 500)]
RIGHT_EDGES = {2: 405, 3: 490, 4: 565}


def main() -> None:
    """Draw the header and transaction table onto a single Letter-size page."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    y = 60
    for line in HEADER:
        page.insert_text((50, y), line, fontsize=11, fontname="helv")
        y += 18

    # Column headings.
    y += 10
    for name, x in COLUMNS:
        page.insert_text((x, y), name, fontsize=10, fontname="hebo")
    y += 20

    for row in ROWS:
        for col, value in enumerate(row):
            if not value:
                continue
            if col in RIGHT_EDGES:
                # Right-align numbers so the decimal points line up.
                width = pymupdf.get_text_length(value, fontname="helv", fontsize=10)
                x = RIGHT_EDGES[col] - width
            else:
                x = COLUMNS[col][1]
            page.insert_text((x, y), value, fontsize=10, fontname="helv")
        y += 18

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
