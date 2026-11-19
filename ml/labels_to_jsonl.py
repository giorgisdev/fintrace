"""Convert the labelling sheet's CSV export into data/labels.jsonl."""
import csv, json, sys
from decimal import Decimal, InvalidOperation

FIELDS = ["doc_type", "vendor", "gst_number", "invoice_number", "date", "due_date",
          "subtotal", "fees", "discounts", "currency", "gst_hst", "pst", "tax_total", "total"]
AMOUNTS = {"subtotal", "fees", "discounts", "gst_hst", "pst", "tax_total", "total"}

def clean(field, raw):
    value = (raw or "").strip()
    if not value:
        return None
    if field in AMOUNTS:
        try:
            return float(Decimal(value.replace("$", "").replace(",", "")))
        except InvalidOperation:
            sys.exit(f"bad amount in {field}: {raw!r}")
    if field == "gst_number":
        return value.replace(" ", "")
    return value

src = sys.argv[1] if len(sys.argv) > 1 else "data/labels.csv"
with open(src, newline="", encoding="utf-8-sig") as f, \
     open("data/labels.jsonl", "w", encoding="utf-8") as out:
    for row in csv.DictReader(f):
        if not row.get("file", "").strip():
            continue
        label = {k: clean(k, row.get(k)) for k in FIELDS}
        a = {k: Decimal(str(label[k])) if label[k] is not None else None for k in AMOUNTS}
        taxes = a["tax_total"] if a["tax_total"] is not None else (a["gst_hst"] or 0) + (a["pst"] or 0)
        if a["subtotal"] is not None and a["total"] is not None:
            calc = a["subtotal"] + (a["fees"] or 0) - (a["discounts"] or 0) + taxes
            if calc != a["total"]:
                print(f"check {row['file']}: computed {calc} != total {a['total']}")
        out.write(json.dumps({"file": row["file"].strip(), "output": label}, ensure_ascii=False) + "\n")
print("wrote data/labels.jsonl")
