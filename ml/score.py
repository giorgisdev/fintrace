"""python ml/score.py data/test.jsonl predictions.jsonl -> accuracy per field."""
import json, sys
from decimal import Decimal, InvalidOperation

FIELDS = ["doc_type", "vendor", "gst_number", "invoice_number", "date", "due_date",
          "subtotal", "fees", "discounts", "currency", "gst_hst", "pst", "tax_total", "total"]
AMOUNTS = {"subtotal", "fees", "discounts", "gst_hst", "pst", "tax_total", "total"}

def norm(field, v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if field in AMOUNTS:
        try:
            return Decimal(str(v).replace("$", "").replace(",", "")).quantize(Decimal("0.01"))
        except InvalidOperation:
            return str(v)
    s = str(v).strip().casefold()
    return s.replace(" ", "") if field == "gst_number" else s

gold = {r["file"]: r["output"] for r in map(json.loads, open(sys.argv[1], encoding="utf-8"))}
pred = {r["file"]: r.get("prediction") or {} for r in map(json.loads, open(sys.argv[2], encoding="utf-8"))}
right = {f: 0 for f in FIELDS}
has, has_right = {f: 0 for f in FIELDS}, {f: 0 for f in FIELDS}
perfect = 0
for file, g in gold.items():
    p = pred.get(file, {})
    ok = []
    for f in FIELDS:
        good = norm(f, g.get(f)) == norm(f, p.get(f))
        ok.append(good); right[f] += good
        if norm(f, g.get(f)) is not None:
            has[f] += 1; has_right[f] += good
        if not good:
            print(f"  {file} {f}: want {g.get(f)!r} got {p.get(f)!r}")
    perfect += all(ok)
n = len(gold)
print(f"\n{n} test docs        all     when printed")
for f in FIELDS:
    hp = f"{has_right[f]}/{has[f]}" if has[f] else "-"
    print(f"{f:15} {right[f]}/{n}     {hp}")
print(f"{'all fields':15} {perfect}/{n}")
