import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src import env, utils
import re
import json
from tqdm import tqdm
from rapidfuzz import fuzz
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = env.MODEL_ID

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\.\-/%]", "", s)
    return s

def em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

def f1_fuzzy(pred: str, gold: str) -> float:
    return fuzz.token_set_ratio(normalize(pred), normalize(gold)) / 100.0

def eval_ocrbench(n=200):
    ds = load_dataset("echo840/OCRBench", split="test", cache_dir=env.DATA_DIR)
    # OCRBench schema varies; common fields: image, question, answer
    return ("OCRBench", ds, "question", "answer", "image", n)

def eval_chartqa(n=200):
    ds = load_dataset("HuggingFaceM4/ChartQA", split="val", cache_dir=env.DATA_DIR)
    # ChartQA fields commonly: image, query, label
    return ("ChartQA", ds, "query", "label", "image", n)

def eval_docvqa(n=200):
    ds = load_dataset("lmms-lab/DocVQA", name='DocVQA', split="validation", cache_dir=env.DATA_DIR)  # :contentReference[oaicite:6]{index=6}
    # often: image, question, answers (list) or answer
    # We'll treat gold as first answer if list.
    return ("DocVQA", ds, "question", "answers", "image", n)

def get_gold(ex, ans_key):
    a = ex.get(ans_key)
    if isinstance(a, list) and len(a) > 0:
        return str(a[0])
    return str(a)

def eval_pmc_vqa(n=200):
    ds = load_dataset("hamzamooraj99/PMC-VQA-1", split="train", cache_dir=env.DATA_DIR)
    return ("PMC-VQA", ds, n)

def build_mcq_prompt(ex):
    q = str(ex["Question"])
    a = str(ex.get("Choice A", ""))
    b = str(ex.get("Choice B", ""))
    c = str(ex.get("Choice C", ""))
    d = str(ex.get("Choice D", ""))

    prompt = (
        f"{q}\n\n"
        f"Choices:\n"
        f"A) {a}\n"
        f"B) {b}\n"
        f"C) {c}\n"
        f"D) {d}\n\n"
        "Answer with only one letter: A, B, C, or D."
    )
    return prompt

def parse_letter(pred: str):
    # Pick the first standalone A/B/C/D
    pred = pred.strip().upper()
    for ch in ["A", "B", "C", "D"]:
        if pred == ch:
            return ch
    # fallback: search anywhere
    import re
    m = re.search(r"\b([ABCD])\b", pred)
    return m.group(1) if m else None

def run_one_pmc_mcq(ds, wrapper, n=200, out_path="results/baseline_PMC-VQA.json"):
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    correct = 0
    rows = []

    for ex in tqdm(ds, desc="Eval PMC-VQA (MCQ)"):
        img = ex["image"]
        prompt = build_mcq_prompt(ex)
        gold = str(ex["Answer_label"]).strip().upper()  # A/B/C/D

        pred_text = wrapper.infer_one(img, prompt, max_new_tokens=64)
        pred = parse_letter(pred_text)

        ok = int(pred == gold)
        correct += ok

        rows.append({
            "q": ex["Question"],
            "gold_label": gold,
            "pred_text": pred_text,
            "pred_label": pred,
            "acc": ok
        })

    result = {"task": "PMC-VQA", "n": len(ds), "ACC": correct / len(ds)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": result, "samples": rows[:50]}, f, ensure_ascii=False, indent=2)

    print("\n", result)
    return result

def run_one(task_tuple, wrapper, out_path):
    name, ds, qk, ak, ik, n = task_tuple
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    total_em, total_fz = 0, 0.0
    rows = []

    for ex in tqdm(ds, desc=f"Eval {name}"):
        img = ex[ik]
        q = str(ex[qk])
        gold = get_gold(ex, ak)

        pred = wrapper.infer_one(img, q, max_new_tokens=64)
        score_em = em(pred, gold)
        score_fz = f1_fuzzy(pred, gold)

        total_em += score_em
        total_fz += score_fz
        rows.append({"task": name, "q": q, "gold": gold, "pred": pred, "em": score_em, "fuzzy": score_fz})

    result = {
        "task": name,
        "n": len(ds),
        "EM": total_em / len(ds),
        "Fuzzy": total_fz / len(ds),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": result, "samples": rows[:50]}, f, ensure_ascii=False, indent=2)

    print("\n", result)
    return result

def main():
    wrapper = utils.load_model_wrapper(MODEL_ID, dtype=torch.bfloat16)

    tasks = [
        eval_ocrbench(200),
        eval_chartqa(200),
        eval_docvqa(200),
    ]

    summaries = []
    for t in tasks:
        summaries.append(run_one(t, wrapper, out_path=f"results/baseline_{t[0]}.json"))

    # PMC-VQA MCQ eval
    pmc_name, pmc_ds, pmc_n = eval_pmc_vqa(200)
    summaries.append(run_one_pmc_mcq(pmc_ds, wrapper, n=pmc_n,
                                    out_path="results/baseline_PMC-VQA.json"))

    print("\n=== BASELINE SUMMARY ===")
    for s in summaries:
        print(s)

if __name__ == "__main__":
    main()
