"""Join data/ocr/*.json with data/labels.jsonl into data/train.jsonl and data/test.jsonl."""
import json
from pathlib import Path

MAX_PAGES = 2
MAX_CHARS = 6000

def ocr_text(ocr):
    pages = ["\n".join(line["text"] for line in p["lines"]) for p in ocr["pages"][:MAX_PAGES]]
    return "\n\n".join(pages)[:MAX_CHARS]

train, test = [], []
for line in open("data/labels.jsonl", encoding="utf-8"):
    label = json.loads(line)
    path = Path("data/ocr") / f"{Path(label['file']).stem}.json"
    if not path.exists():
        print("no OCR for", label["file"]); continue
    ocr = json.loads(path.read_text(encoding="utf-8"))
    row = {"file": label["file"], "input": ocr_text(ocr), "output": label["output"]}
    (test if int(ocr["sha256"][:8], 16) % 5 == 0 else train).append(row)

for name, rows in (("train", train), ("test", test)):
    with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"train {len(train)}, test {len(test)}")
print("test files:", [r["file"] for r in test])
