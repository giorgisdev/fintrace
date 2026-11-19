"""python ml/ollama_eval.py qwen3:0.6b data/pred_qwen_base.jsonl"""
import json, sys, urllib.request
sys.path.insert(0, "ml")
from prompt import SYSTEM

model, out_path = sys.argv[1], sys.argv[2]
with open(out_path, "w", encoding="utf-8") as out:
    for row in map(json.loads, open("data/test.jsonl", encoding="utf-8")):
        body = {"model": model, "stream": False, "think": False, "format": "json",
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": row["input"]}]}
        req = urllib.request.Request("http://localhost:11434/api/chat",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        text = json.load(urllib.request.urlopen(req))["message"]["content"]
        try: pred = json.loads(text)
        except json.JSONDecodeError: pred = {}
        out.write(json.dumps({"file": row["file"], "prediction": pred, "raw": text}, ensure_ascii=False) + "\n")
        print(row["file"], "done")
