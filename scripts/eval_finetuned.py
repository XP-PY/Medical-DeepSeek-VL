import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src import env, utils
import json
from tqdm import tqdm
import torch
import argparse
import os

MODEL_ID = env.MODEL_ID

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_ID", type=str, default="deepseek-ai/deepseek-vl2-tiny")
    ap.add_argument("--lora_path", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    # model
    wrapper = utils.load_model_loRAwrapper(args.model_ID, lora_path=args.lora_path, dtype=torch.bfloat16)

    # dataset
    val_ds = utils.JsonlVLDataset(args.val_jsonl)

    # Evaluation
    total_result = {
        'chartqa': {'em':0, 'fz': 0.0, 'count': 0},
        'ocrbench': {'em':0, 'fz': 0.0, 'count': 0},
        'docvqa': {'em':0, 'fz': 0.0, 'count': 0},
        'pmcvqa': {'acc': 0, 'count': 0},
    }
    rows = []
    for x in tqdm(val_ds, desc="Evaluation"):
        id, answer, prompt, img = x['id'], x['response'], x['prompt'], x['image_pil']
        pred = wrapper.infer_one(img, prompt, max_new_tokens=4096)

        prefix = id.split("_")[0]

        if prefix == 'pmcvqa':
            score_acc = utils.acc(pred, answer)

            total_result[prefix]['acc'] += score_acc
            total_result[prefix]['count'] += 1
            rows.append({"id": id, "prompt": prompt, "answer": answer, "pred": pred, 'acc': score_acc})
        else:
            score_em = utils.em(pred, answer)
            score_fz = utils.f1_fuzzy(pred, answer)

            total_result[prefix]['em'] += score_em
            total_result[prefix]['fz'] += score_fz
            total_result[prefix]['count'] += 1
            rows.append({"id": id, "prompt": prompt, "answer": answer, "pred": pred, "em": score_em, "fuzzy": score_fz})

    # Write in JSON file
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/finetuned_eval.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {}
    for prefix in ['chartqa', 'ocrbench', 'docvqa', 'pmcvqa']:
        if prefix == 'pmcvqa':
            acc = total_result[prefix]['acc'] / total_result[prefix]['count'] if total_result[prefix]['count'] > 0 else 0

            summary[prefix] = {
                "task": prefix,
                "Count": total_result[prefix]['count'],
                "ACC": acc,
            }

            print('='*20)
            print(f"Task: {prefix}")
            print(f"ACC: {acc:.4f}")
            print('='*20, '\n\n')
        else:
            avg_em = total_result[prefix]['em'] / total_result[prefix]['count'] if total_result[prefix]['count'] > 0 else 0
            avg_fz = total_result[prefix]['fz'] / total_result[prefix]['count'] if total_result[prefix]['count'] > 0 else 0

            summary[prefix] = {
                "task": prefix,
                "Count": total_result[prefix]['count'],
                "EM": avg_em,
                "Fuzzy": avg_fz
            }

            print('='*20)
            print(f"Task: {prefix}")
            print(f"Average EM: {avg_em:.4f}")
            print(f"Average Fuzzy F1: {avg_fz:.4f}")
            print('='*20, '\n\n')

    with open(f"{args.out_dir}/finetuned_eval_summary.jsonl", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"The results have been saved to {args.out_dir}/finetuned_eval.jsonl and {args.out_dir}/finetuned_eval_summary.jsonl.")
    
if __name__ == "__main__":
    main()
