#!/usr/bin/env python3
import json
import argparse
import os

def extract_question(prompt: str) -> str:
    # Your prompt is like "<image>\n{question}" (or sometimes already text)
    if prompt is None:
        return ""
    s = str(prompt)
    s = s.replace("<image>\n", "").replace("<image>", "").strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, default="data/train_mix.jsonl")
    ap.add_argument("--out_jsonl", type=str, default="data/train_swift.jsonl")
    ap.add_argument("--max_rows", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    n_in = 0
    n_out = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            n_in += 1

            img_path = obj.get("image", "")
            q = extract_question(obj.get("prompt", ""))
            a = str(obj.get("response", "")).strip()

            if not img_path or not q or not a:
                continue

            # ms-swift Query-Response format + multimodal:
            # - query includes <img></img> placeholder
            # - images list holds the actual path
            out = {
                "query": "<img></img>\n" + q,
                "response": a,
                "images": [img_path],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

            if args.max_rows > 0 and n_out >= args.max_rows:
                break

    print(f"Converted: {n_out}/{n_in} -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
