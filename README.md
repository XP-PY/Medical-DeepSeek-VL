# Dataset Construction
* **Stage A (doc skills):** train on DocVQA/ChartQA/OCRBench → learn layout/table/chart/OCR well
* **Stage B (medical domain):** add **medical VQA from biomedical papers** (best match: **PMC-VQA**) → learn medical terminology + biomedical figure/table style
* **Stage C (medical focus without labeling):** inference-time **PDF RAG** over PubMed/Guidelines for “medical-ness” and evidence

# Training Plan
* Trained doc-understanding on DocVQA/ChartQA/OCRBench
* Added PMC-VQA to inject biomedical terminology + paper-style figures/tables
* Final system answers questions over medical PDFs with evidence via RAG

# Schedule
| Stage | Task content | Script | Remark |
|:---:|:---:|:---:|:---:|
| **Set up environment** | Create a conda/venv env | pip install -r requirements.txt | - |
| **Simple Test** | Test base function (env.py/utils.py) | python scripts/smoke_infer.py | Run DeepSeeK-VL2 successfully |
| **Download datasets** | Download open source datasets ([OCRBench](https://huggingface.co/datasets/echo840/OCRBench)/[ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)/[DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)/[PMC-VQA](https://huggingface.co/datasets/hamzamooraj99/PMC-VQA-1)) | python scripts/download_data.py | Three doc skills datasets and a medical domain dataset |
| **Baseline evaluation** | Baseline evaluation (zero-shot) on small subsets | python scripts/eval_baseline.py | OCRBench/ChartQA/DocVQA metric: EM/Fuzzy; PMC-VQA metric: MCQ Accuracy |