# Set up environment
conda create --name deepseek-vl2 python=3.10
conda activate deepseek-vl2
pip install -r requirements.txt

# Simple Test
python scripts/smoke_infer.py

# Download datasets
python scripts/download_data.py

# Build mix train/val dataset
python scripts/build_train_jsonl.py \
  --train_chartqa 27000 \
  --train_ocrbench 500 \
  --train_docvqa 4500 \
  --train_pmc 60000 \
  --val_each 500

# Run LoRA fine-tuning
# DDP
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision bf16 scripts/train_lora_hf.py \
  --model_ID deepseek-ai/deepseek-vl2-tiny \
  --train_jsonl data/train_mix.jsonl \
  --val_jsonl data/val_mix.jsonl \
  --out_dir output/checkpoints/lora_docqa_med_hf_92000 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 2e-4 \
  --save_steps 200 \
  --eval_steps 200 \
  --bf16
  # --resume True
# Single GPU
CUDA_VISIBLE_DEVICES=3 python scripts/train_lora_hf.py \
  --model_ID deepseek-ai/deepseek-vl2-tiny \
  --train_jsonl data/train_mix.jsonl \
  --val_jsonl data/val_mix.jsonl \
  --out_dir output/checkpoints/lora_docqa_med_hf_92000 \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 1e-4 \
  --save_steps 100 \
  --eval_steps 500 \
  --bf16 \
  # --resume True

# Baseline evaluation
CUDA_VISIBLE_DEVICES=3 python scripts/eval_baseline.py \
  --model_ID deepseek-ai/deepseek-vl2-tiny \
  --val_jsonl data/val_mix.jsonl \
  --out_dir results

# Fine-tuned model evaluation
CUDA_VISIBLE_DEVICES=3 python scripts/eval_finetuned.py \
  --model_ID deepseek-ai/deepseek-vl2-tiny \
  --lora_path output/checkpoints/lora_docqa_med_hf_92000 \
  --val_jsonl data/val_mix.jsonl \
  --out_dir results