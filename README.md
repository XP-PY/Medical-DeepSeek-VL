# Dataset Construction
* **Stage A (doc skills):** train on DocVQA/ChartQA/OCRBench → learn layout/table/chart/OCR well
* **Stage B (medical domain):** add **medical VQA from biomedical papers** (best match: **PMC-VQA**) → learn medical terminology + biomedical figure/table style
* **Stage C (medical focus without labeling):** inference-time **PDF RAG** over PubMed/Guidelines for “medical-ness” and evidence

# Training Plan
* Trained doc-understanding on DocVQA/ChartQA/OCRBench
* Added PMC-VQA to inject biomedical terminology + paper-style figures/tables
* Final system answers questions over medical PDFs with evidence via RAG