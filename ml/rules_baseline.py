import json, re
AMT = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})*\.\d{2})")
GST = re.compile(r"\b(\d{9})\s?RT\s?(\d{4})\b", re.I)
DATE = re.compile(r"\b(20\d{2})[-/](\d{2})[-/](\d{2})\b")

with open("data/pred_rules.jsonl", "w", encoding="utf-8") as out:
    for row in map(json.loads, open("data/test.jsonl", encoding="utf-8")):
        text, p = row["input"], {}
        totals = [float(m.replace(",", "")) for line in text.splitlines()
                  if "total" in line.lower() and "sub" not in line.lower()
                  for m in AMT.findall(line)]
        p["total"] = max(totals) if totals else None
        if m := GST.search(text): p["gst_number"] = f"{m[1]}RT{m[2]}"
        if m := DATE.search(text): p["date"] = f"{m[1]}-{m[2]}-{m[3]}"
        out.write(json.dumps({"file": row["file"], "prediction": p}) + "\n")
