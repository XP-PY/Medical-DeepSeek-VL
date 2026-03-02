import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import env, utils
from src.rag_retriever import PMCOARetriever

import json
from tqdm import tqdm
import torch
import argparse
import os

def strip_image_token(prompt: str) -> str:
    # Your prompt format is typically "<image>\n{question...}"
    if prompt.startswith("<image>"):
        return prompt.split("\n", 1)[-1].strip()
    return prompt.strip()

def build_rag_prompt(original_prompt: str, evidence: str, task_prefix: str) -> str:
    """
    Keep <image> at the beginning (DeepSeek-VL2 expects it if image is provided),
    then add evidence + instructions.
    """
    question = strip_image_token(original_prompt)

    if task_prefix == "pmcvqa":
        # MCQ style: force short output
        # Your prompt already contains choices if you built it that way; if not, keep original as-is.
        return (
            f"<image>\n"
            f"{question}\n\n"
            f"Medical evidence (may help):\n{evidence}\n\n"
            "Answer with only one letter: A, B, C, or D."
        )
    else:
        # general doc QA style
        return (
            f"<image>\n"
            f"Evidence:\n{evidence}\n\n"
            f"Question: {question}\n\n"
            "Answer briefly. If the answer is a number, output only the number."
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_ID", type=str, default="deepseek-ai/deepseek-vl2-tiny")
    ap.add_argument("--lora_path", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results_rag")

    # RAG args
    ap.add_argument("--faiss_index", type=str, required=True, help="e.g. corpus/pmc_oa_1k/faiss.index")
    ap.add_argument("--meta_jsonl", type=str, required=True, help="e.g. corpus/pmc_oa_1k/chunks_meta.jsonl")
    ap.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_evidence_chars", type=int, default=1500)

    args = ap.parse_args()

    # model
    wrapper = utils.load_model_loRAwrapper(args.model_ID, lora_path=args.lora_path, dtype=torch.bfloat16)

    # RAG retriever (load once)
    retriever = PMCOARetriever(
        index_path=args.faiss_index,
        meta_path=args.meta_jsonl,
        embed_model=args.embed_model
    )

    # dataset
    val_ds = utils.JsonlVLDataset(args.val_jsonl)

    # Evaluation containers
    total_result = {
        "pmcvqa": {"acc": 0, "count": 0},
        # If you later want: chartqa/ocrbench/docvqa, add them back.
    }

    rows = []
    for x in tqdm(val_ds, desc="Evaluation (RAG)"):
        _id, answer, prompt, img = x["id"], x["response"], x["prompt"], x["image_pil"]
        prefix = _id.split("_")[0]

        # Only evaluate pmcvqa here (matches your current code)
        if prefix != "pmcvqa":
            continue

        # RAG retrieve using question text (without <image>)
        qtext = strip_image_token(prompt)
        hits = retriever.search(qtext, k=args.topk)
        evidence = retriever.format_hits(hits, max_chars=args.max_evidence_chars)

        rag_prompt = build_rag_prompt(prompt, evidence, prefix)

        # MCQ → keep short decode
        pred = wrapper.infer_one(img, rag_prompt, max_new_tokens=16)

        score_acc = utils.acc(pred, answer)
        total_result[prefix]["acc"] += score_acc
        total_result[prefix]["count"] += 1

        rows.append({
            "id": _id,
            "task": prefix,
            "question": qtext,
            "rag_prompt": rag_prompt,
            "answer": answer,
            "pred": pred,
            "acc": score_acc,
            "hits": hits[:args.topk],
        })

    # Write results
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/rag_eval.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {}
    for prefix in ["pmcvqa"]:
        cnt = total_result[prefix]["count"]
        acc = total_result[prefix]["acc"] / cnt if cnt > 0 else 0.0
        summary[prefix] = {"task": prefix, "Count": cnt, "ACC": acc}

        print("=" * 20)
        print(f"Task: {prefix}")
        print(f"ACC:  {acc:.4f}")
        print("=" * 20, "\n")

    with open(f"{args.out_dir}/rag_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved to {args.out_dir}/rag_eval.jsonl and {args.out_dir}/rag_eval_summary.json")

if __name__ == "__main__":
    main()