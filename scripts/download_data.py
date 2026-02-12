import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import env
from datasets import load_dataset

def main():
    # Doc skills
    load_dataset("echo840/OCRBench", split="test", cache_dir=env.DATA_DIR)                 # OCR benchmark  not needed here
    load_dataset("HuggingFaceM4/ChartQA", split="train", cache_dir=env.DATA_DIR)
    load_dataset("HuggingFaceM4/ChartQA", split="val", cache_dir=env.DATA_DIR)

    # DocVQA without RRC login (HF formatted version)
    load_dataset("lmms-lab/DocVQA", name="DocVQA", split="validation", cache_dir=env.DATA_DIR)            # :contentReference[oaicite:3]{index=3}

    # Medical VQA from biomedical papers (HF mirror)
    load_dataset("hamzamooraj99/PMC-VQA-1", split="train", cache_dir=env.DATA_DIR)         # :contentReference[oaicite:4]{index=4}

    print("All datasets downloaded into HF cache.")

if __name__ == "__main__":
    main()
