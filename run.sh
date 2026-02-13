# Set up environment
pip install -r requirements.txt

# Simple Test
python scripts/smoke_infer.py

# Download datasets
python scripts/download_data.py

# Baseline evaluation
python scripts/eval_baseline.py

# Build mix train/val dataset
python scripts/build_train_jsonl.py \
  --train_chartqa 500 \
  --train_ocrbench 200 \
  --train_docvqa 300 \
  --train_pmc 300 \
  --val_each 50