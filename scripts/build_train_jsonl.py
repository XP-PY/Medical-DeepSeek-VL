#!/usr/bin/env python3
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src import env, utils
import os
import re
import json
import argparse
from typing import Any, Dict, Optional, Tuple, List

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ----------------------------
# Utils
# ----------------------------
def ensure_rgb(img: Image.Image) -> Image.Image:
    """Force PIL image to RGB (fixes RGBA/LA/P issues)."""
    if isinstance(img, Image.Image) and img.mode != "RGB":
        return img.convert("RGB")
    return img

def first_nonempty(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return str(x)
    return str(x).strip()

def pick_question(ex: Dict[str, Any]) -> Optional[str]:
    # Common variants
    return first_nonempty(
        ex.get("question"),
        ex.get("Question"),
        ex.get("query")
    )

def pick_answer(ex: Dict[str, Any]) -> Optional[str]:
    # Some datasets store list of answers
    a = first_nonempty(
        ex.get("answer"), 
        ex.get("answers"), 
        ex.get("label"), 
        ex.get("Answer_label")
    )
    if a is not None:
        if isinstance(a, list) and len(a) > 0:
            return safe_str(a[0])
        return safe_str(a)

    return None

def pick_image(ex: Dict[str, Any]) -> Optional[Image.Image]:
    # HF image feature usually under "image"
    img = ex.get("image")
    if isinstance(img, Image.Image):
        return ensure_rgb(img)

def build_mcq_prompt(question: str, A: str, B: str, C: str, D: str) -> str:
    return (
        f"{question}\n\n"
        "Choices:\n"
        f"A) {A}\n"
        f"B) {B}\n"
        f"C) {C}\n"
        f"D) {D}\n\n"
        "Answer with only one letter: A, B, C, or D."
    )

def normalize_answer_label(x: str) -> Optional[str]:
    x = safe_str(x).upper()
    if x in ["A", "B", "C", "D"]:
        return x
    # Sometimes numeric labels appear
    mapping = {"0": "A", "1": "B", "2": "C", "3": "D"}
    return mapping.get(x, None)

def make_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:07d}"

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Dataset converters
# Each returns standardized samples:
# {id, image (PIL), prompt, response, meta}
# NOTE: We keep PIL in-memory while building, and save images to disk for jsonl.
# ----------------------------
def convert_generic(
    ds,
    prefix: str,
    max_samples: int,
    shuffle_seed: int,
    mcq: bool = False
) -> List[Dict[str, Any]]:
    ds2 = ds.shuffle(seed=shuffle_seed)
    take = min(max_samples, len(ds2))
    ds2 = ds2.select(range(take))

    out = []
    for i, ex in enumerate(tqdm(ds2, desc=f"Convert {prefix}")):
        img = pick_image(ex)
        q = pick_question(ex)
        a = pick_answer(ex)
        if img is None or q is None or a is None:
            continue
        if mcq:
            A = safe_str(ex.get("Choice A"))
            B = safe_str(ex.get("Choice B"))
            C = safe_str(ex.get("Choice C"))
            D = safe_str(ex.get("Choice D"))
            q = build_mcq_prompt(safe_str(q), A, B, C, D)
        out.append({
            "id": make_id(prefix, i),
            "image_pil": img,
            "prompt": f"<image>\n{safe_str(q)}",
            "response": safe_str(a),
            "meta": {"source": prefix, "mcq": mcq}
        })
    return out

def convert_pmc_vqa(
    ds,
    prefix: str,
    max_samples: int,
    shuffle_seed: int,
    label_as_response: bool = True,
) -> List[Dict[str, Any]]:
    ds2 = ds.shuffle(seed=shuffle_seed)
    take = min(max_samples, len(ds2))
    ds2 = ds2.select(range(take))

    out = []
    for i, ex in enumerate(tqdm(ds2, desc=f"Convert {prefix} (MCQ)")):
        img = pick_image(ex)
        if img is None:
            continue

        q = first_nonempty(ex.get("Question"), ex.get("question"))
        if q is None:
            continue

        A = safe_str(ex.get("Choice A"))
        B = safe_str(ex.get("Choice B"))
        C = safe_str(ex.get("Choice C"))
        D = safe_str(ex.get("Choice D"))
        ans_label = normalize_answer_label(ex.get("Answer_label", ""))

        # Fallback to free-form answer if label missing
        free_ans = safe_str(ex.get("Answer"))

        if label_as_response:
            if ans_label is None:
                continue
            prompt = build_mcq_prompt(safe_str(q), A, B, C, D)
            response = ans_label
        else:
            # Train model to output the actual answer string
            # (more flexible, but slightly less stable)
            prompt = f"<image>\n{safe_str(q)}"
            response = free_ans if free_ans else (ans_label or "")
            if response == "":
                continue

        out.append({
            "id": make_id(prefix, i),
            "image_pil": img,
            "prompt": f"<image>\n{prompt}" if not prompt.startswith("<image>") else prompt,
            "response": response,
            "meta": {"source": prefix, "mcq": True}
        })
    return out


# ----------------------------
# Save images to disk and finalize jsonl
# ----------------------------
def materialize_images(rows: List[Dict[str, Any]], image_root: str) -> List[Dict[str, Any]]:
    os.makedirs(image_root, exist_ok=True)
    finalized = []

    for r in tqdm(rows, desc="Saving images"):
        img: Image.Image = r.pop("image_pil")
        sid = r["id"]
        # store as jpg to save space
        out_path = os.path.join(image_root, f"{sid}.jpg")
        img.save(out_path, format="JPEG", quality=95)

        r["image"] = out_path  # store path
        finalized.append(r)

    return finalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='data')
    parser.add_argument("--train_out", type=str, default="train_mix.jsonl")
    parser.add_argument("--val_out", type=str, default="val_mix.jsonl")
    parser.add_argument("--image_dir", type=str, default="images_mix")

    # sample counts
    parser.add_argument("--train_chartqa", type=int, default=30000)
    parser.add_argument("--train_ocrbench", type=int, default=10000)
    parser.add_argument("--train_docvqa", type=int, default=20000)
    parser.add_argument("--train_pmc", type=int, default=10000)

    parser.add_argument("--val_each", type=int, default=500)  # per dataset
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # Load datasets
    # ----------------------------
    print("Loading datasets from HuggingFace cache...")
    chart_train = load_dataset("HuggingFaceM4/ChartQA", split="train", cache_dir=env.DATA_DIR)
    chart_val = load_dataset("HuggingFaceM4/ChartQA", split="val", cache_dir=env.DATA_DIR)

    ocr_test = load_dataset("echo840/OCRBench", split="test")  # only test exists; we can sample for train
    # if OCRBench has train split in your environment, switch accordingly.

    docvqa_val = load_dataset("lmms-lab/DocVQA", name="DocVQA", split="validation", cache_dir=env.DATA_DIR)

    pmc_train = load_dataset("hamzamooraj99/PMC-VQA-1", split="train", cache_dir=env.DATA_DIR)

    # ----------------------------
    # Build TRAIN mix
    # ----------------------------
    train_rows = []
    train_rows += convert_generic(chart_train, "chartqa", args.train_chartqa, args.seed)
    train_rows += convert_generic(ocr_test, "ocrbench", args.train_ocrbench, args.seed + 1)
    train_rows += convert_generic(docvqa_val, "docvqa", args.train_docvqa, args.seed + 2)
    train_rows += convert_generic(pmc_train, "pmcvqa", args.train_pmc, args.seed + 3, mcq=True)
    # train_rows += convert_pmc_vqa(pmc_train, "pmcvqa", args.train_pmc, args.seed + 3, label_as_response=args.pmc_label_response)

    # ----------------------------
    # Build VAL mix (smaller, balanced)
    # ----------------------------
    val_rows = []
    val_rows += convert_generic(chart_val, "chartqa_val", args.val_each, args.seed)
    val_rows += convert_generic(ocr_test, "ocrbench_val", args.val_each, args.seed + 1)
    val_rows += convert_generic(docvqa_val, "docvqa_val", args.val_each, args.seed + 2)
    val_rows += convert_generic(pmc_train, "pmcvqa_val", args.val_each, args.seed + 3, mcq=True)
    # val_rows += convert_pmc_vqa(pmc_train, "pmcvqa_val", args.val_each, args.seed + 3, label_as_response=args.pmc_label_response)

    # ----------------------------
    # Materialize images
    # ----------------------------
    train_img_root = os.path.join(out_dir, args.image_dir, "train")
    val_img_root = os.path.join(out_dir, args.image_dir, "val")

    train_final = materialize_images(train_rows, train_img_root)
    val_final = materialize_images(val_rows, val_img_root)

    # ----------------------------
    # Write jsonl
    # ----------------------------
    train_path = os.path.join(out_dir, args.train_out)
    val_path = os.path.join(out_dir, args.val_out)

    write_jsonl(train_path, train_final)
    write_jsonl(val_path, val_final)

    print("\nDone!")
    print(f"Train samples: {len(train_final)} -> {train_path}")
    print(f"Val samples:   {len(val_final)} -> {val_path}")
    print(f"Images saved under: {os.path.join(out_dir, args.image_dir)}")

    # Quick sanity print
    if len(train_final) > 0:
        print("\nExample row:")
        print(json.dumps(train_final[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
