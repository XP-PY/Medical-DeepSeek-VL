import os
import json
import argparse
from PIL import Image
from torch.utils.data import Dataset

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# from src import utils, data_collator_deepseek_vl2
from src.utils import load_model, JsonlVLDataset
from src.data_collator_deepseek_vl2 import DeepSeekVL2DataCollator

def find_lora_targets(model):
    """
    Reasonable default for Llama-like backbones.
    If some names don't exist, PEFT will ignore them; but you can tailor it by printing model modules.
    """
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_ID", type=str, default="deepseek-ai/deepseek-vl2-tiny")
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="checkpoints/lora_docqa_med_hf")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint: pass a path or 'True' for latest.")

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # processor + base model (DeepSeek-VL2 uses trust_remote_code)
    vl_processor, tokenizer, base, _ = load_model(MODEL_ID = args.model_ID)
    base.train()
    # base.gradient_checkpointing_enable()
    base.config.use_cache = False

    # LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=find_lora_targets(base),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    # datasets
    train_ds = JsonlVLDataset(args.train_jsonl)
    val_ds = JsonlVLDataset(args.val_jsonl)

    # data collator
    pad_token_id = tokenizer.eos_token_id
    collator = DeepSeekVL2DataCollator(vl_processor, pad_token_id=pad_token_id)

    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,  # IMPORTANT: we pass custom keys (images, masks, etc.)
        dataloader_num_workers=4,
        max_steps=args.max_steps,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    resume_from_checkpoint = args.resume
    if resume_from_checkpoint == "True":
        resume_from_checkpoint = True   # Automutically find newest checkpoint in output_dir
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save adapter
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Saved LoRA adapter to:", args.out_dir)


if __name__ == "__main__":
    main()