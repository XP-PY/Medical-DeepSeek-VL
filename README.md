# Medical Deepseek-VL
This repo builds a medical multimodal document QA system based on DeepSeek-VL2-Tiny. I fine-tune the model with LoRA on a mixed set of document understanding benchmarks (OCRBench/ChartQA/DocVQA) plus a biomedical VQA dataset (PMC-VQA), then add a RAG module using a PMC Open Access (PMC OA) 1k-document corpus indexed by BGE embeddings + FAISS. The project provides reproducible scripts for data preparation, training, evaluation, and RAG retrieval.

# Schedule

|              Stage              | Task content                                                  | Remark                              |
| :-----------------------------: | :------------------------------------------------------------ | :---------------------------------- |
|      **Set up environment**     | Create conda env + install dependencies                       | ✅                                   |
|    **Git DeepSeek-VL2 repo**    | Clone official DeepSeek-VL2 repo                              | ✅                                   |
|         **Simple Test**         | Verify DeepSeek-VL2 inference                                 | ✅                                   |
|      **Download datasets**      | OCRBench / ChartQA / DocVQA / PMC-VQA                         | ✅                                   |
| **Build mix train/val dataset** | Sample and build JSONL                                        | **Train:** 92,000; **Val:** 2,000 ✅ |
|           **Training**          | LoRA fine-tune DeepSeek-VL2                                   | ✅                                   |
|     **Baseline evaluation**     | Zero-shot on val subsets                                      | ✅                                   |
|    **Fine-tuned evaluation**    | Evaluate LoRA model on same subsets                           | ✅                                   |
|           **Add RAG**           | Build PMC OA corpus (1k docs) + FAISS index + RAG evaluation  | ✅                                   |
|          **Iteration**          | Rebalance mixture + output normalization (reduce regressions) | 🔜                                  |
|             **Demo**            | Gradio PDF QA + evidence (baseline vs LoRA vs LoRA+RAG)       | 🔜                                  |

---

# Project Structure

## `scripts/`

* **`smoke_infer.py`**
  Minimal end-to-end inference sanity check. Renders a PDF page (or loads an image) and runs DeepSeek-VL2-Tiny to verify the runtime environment and model loading.

* **`download_data.py`**
  Downloads/loads public fine-tuning and evaluation datasets (OCRBench / ChartQA / DocVQA / PMC-VQA) via Hugging Face `datasets` cache.

* **`build_train_jsonl.py`**
  Builds a mixed training/validation set by sampling from the public datasets and converting them into a unified JSONL format:

  ```json
  {"id": "...", "image": "...jpg", "prompt": "<image>\n...", "response": "..."}
  ```

  Also materializes images to local disk to simplify training and evaluation.

* **`train_lora_hf.py`**
  LoRA SFT training with Hugging Face `Trainer` + PEFT on DeepSeek-VL2-Tiny. Supports single-GPU and DDP (via `accelerate launch`).

* **`eval_baseline.py`**
  Zero-shot evaluation of the base model (`deepseek-ai/deepseek-vl2-tiny`) on the validation subsets.
  Outputs per-sample predictions and a summary JSON.

* **`eval_finetuned.py`**
  Evaluation of the LoRA fine-tuned model on the same validation subsets as baseline.
  Outputs per-sample predictions and a summary JSON.

* **`download_pmc_oa_1k.py`**
  Builds a RAG corpus by discovering and downloading **1,000** PMC Open Access article packages (`.tar.gz`) using official PMC OA service/links.

* **`extract_pmc_xml_text.py`**
  Extracts text chunks from each downloaded PMC OA package (prefers `.nxml` JATS XML).
  Writes `chunks.jsonl` for embedding and retrieval.

* **`redownload_bad_pmc.py`**
  Re-downloads corrupted PMC OA packages listed by `extract_pmc_xml_text.py` (FTP/network truncation handling), then re-runs extraction.

* **`build_faiss_index.py`**
  Embeds RAG chunks using a sentence embedding model (default: `BAAI/bge-small-en-v1.5`) and builds a FAISS index:

  * `faiss.index`
  * `chunks_meta.jsonl`

* **`eval_finetuned_with_rag.py`**
  Runs evaluation with RAG enabled:

  * retrieves top-k evidence chunks from FAISS
  * augments prompt with evidence
  * evaluates the LoRA model under the same metric setting

## `src/`

* **`env.py`**
  Central configuration (model IDs, paths, defaults).

* **`utils.py`**
  Shared utilities:

  * model/processor loading wrappers (base / LoRA)
  * dataset JSONL reader (`JsonlVLDataset`)
  * metrics helpers (`em`, `fuzzy`, `acc`)
  * prompt and image preprocessing helpers

---

# Results
## Evaluation Setup

* **Base model:** `deepseek-ai/deepseek-vl2-tiny`
* **Fine-tuning:** LoRA SFT on a mixed dataset (DocVQA / ChartQA / OCRBench / PMC-VQA)
* **Fine-tuning with RAG:** Fine-tuned model with RAG (PMC-VQA only)
* **Validation:** 500 samples per dataset
* **Metrics:**

  * **ChartQA / OCRBench / DocVQA:** Exact Match (**EM**) + RapidFuzz (**Fuzzy**)
  * **PMC-VQA:** Multiple-choice **Accuracy** (A/B/C/D)
  > ### Huggingface
  >| Dataset | Download |
  >|:--------:|:--------------------------------------------------------------------------:|
  >| ChartQA  | [🤗 Hugging Face](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)  |
  >| OCRBench | [🤗 Hugging Face](https://huggingface.co/datasets/echo840/OCRBench)       |
  >| DocVQA   | [🤗 Hugging Face](https://huggingface.co/datasets/lmms-lab/DocVQA)        |
  >| PMC-VQA  | [🤗 Hugging Face](https://huggingface.co/datasets/hamzamooraj99/PMC-VQA-1)|

## Metrics Summary (n=500 each)

| Dataset      | Metric | Baseline | Fine-tuned | Fine-tuned + RAG | $\Delta$ (FT−Base) | $\Delta$ (RAG−FT) |
| ------------ | -----: | -------: | ---------: | ---------------: | -----------------: | ----------------: |
| **ChartQA**  |     EM |    0.692 |      0.526 |                — |             -0.166 |                 — |
|              |  Fuzzy |   0.8361 |     0.8306 |                — |            -0.0056 |                 — |
| **OCRBench** |     EM |    0.688 |      0.424 |                — |             -0.264 |                 — |
|              |  Fuzzy |   0.8785 |     0.9040 |                — |            +0.0255 |                 — |
| **DocVQA**   |     EM |    0.758 |      0.122 |                — |             -0.636 |                 — |
|              |  Fuzzy |   0.9336 |     0.8862 |                — |            -0.0474 |                 — |
| **PMC-VQA**  |    ACC |    0.362 |      0.480 |            0.426 |         **+0.118** |            -0.054 |

### Key Takeaway

* LoRA fine-tuning **significantly improved medical-domain VQA** performance (**PMC-VQA +11.8% ACC**).
* However, performance on general document QA benchmarks **dropped**, especially **DocVQA EM**.

---

# Reproducible Commands

## 1) Set up environment

```bash
conda create --name deepseek-vl2 python=3.10 -y
conda activate deepseek-vl2
pip install -r requirements.txt
```

## 2) Clone DeepSeek-VL2 repo (optional if installed another way)

```bash
mkdir -p GitRepo
cd GitRepo
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
cd ..
```

## 3) Simple test (verify inference)

```bash
python scripts/smoke_infer.py
```

## 4) Download fine-tuning datasets

```bash
python scripts/download_data.py
```

## 5) Build mixed train/val dataset (JSONL)

```bash
python scripts/build_train_jsonl.py \
  --train_chartqa 27000 \
  --train_ocrbench 500 \
  --train_docvqa 4500 \
  --train_pmc 60000 \
  --val_each 500
```

## 6) LoRA fine-tuning

### 6.1 DDP (2 GPUs example)

```bash
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
```

### 6.2 Single GPU

```bash
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
  --bf16
  # --resume True
```

## 7) Baseline evaluation

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/eval_baseline.py \
  --model_ID deepseek-ai/deepseek-vl2-tiny \
  --val_jsonl data/val_mix.jsonl \
  --out_dir results
```

## 8) Fine-tuned model evaluation

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/eval_finetuned.py \
  --model_ID deepseek-ai/deepseek-vl2-tiny \
  --lora_path output/checkpoints/lora_docqa_med_hf_92000 \
  --val_jsonl data/val_mix.jsonl \
  --out_dir results
```

# RAG (PMC OA, 1k docs)

## 9) Download PMC OA corpus (1k OA packages)

```bash
python scripts/download_pmc_oa_1k.py
```

## 10) Extract text chunks for RAG

```bash
python scripts/extract_pmc_xml_text.py
```

### If you see `Bad archives`, re-download corrupted files and extract again

```bash
python scripts/redownload_bad_pmc.py
python scripts/extract_pmc_xml_text.py
```

## 11) Build FAISS index

```bash
python scripts/build_faiss_index.py \
  --chunks_jsonl corpus/pmc_oa_1k/chunks.jsonl \
  --out_dir corpus/pmc_oa_1k \
  --embed_model BAAI/bge-small-en-v1.5 \
  --index_type flatip \
  --batch_size 128
```

## 12) Fine-tuned model evaluation with RAG

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/eval_finetuned_with_rag.py \
  --model_ID deepseek-ai/deepseek-vl2-tiny \
  --lora_path output/checkpoints/lora_docqa_med_hf_92000 \
  --val_jsonl data/val_mix.jsonl \
  --out_dir results \
  --faiss_index corpus/pmc_oa_1k/faiss.index \
  --meta_jsonl corpus/pmc_oa_1k/chunks_meta.jsonl \
  --topk 5
```

# Known Issues / Next Iteration

## 1) Regression on general document QA after fine-tuning

**Observed:** PMC-VQA improves (ACC ↑), but ChartQA/DocVQA EM drops sharply.
**Likely causes:**

* **Mixture imbalance / task dominance:** PMC-VQA (MCQ) can bias outputs toward “letter-style” answers or short formats.
* **Output format drift:** Fine-tuned model may add extra tokens/prefixes (e.g., `Answer:`), causing EM to collapse even if semantics are correct.
* **Task mismatch:** ChartQA/DocVQA are free-form extractive/generative; PMC-VQA is discrete MCQ.

**Next iteration actions:**

* **Rebalance sampling** (cap PMC steps to 20–30% of training; upweight DocVQA/ChartQA).
* **Two-stage fine-tuning**:

  1. doc skills (ChartQA/OCRBench/DocVQA)
  2. medical domain (PMC-VQA) with smaller LR/fewer steps
* **Strict output constraints** in training prompts:

  * For DocVQA/ChartQA/OCRBench: “Output only the final answer string.”
  * For PMC-VQA: “Output only one letter A/B/C/D.”
* **Eval-time normalization** (strip `Answer:` / whitespace / punctuation) to better reflect true correctness.

## 2) RAG does not always improve PMC-VQA

**Observed:** LoRA+RAG ACC decreased compared with LoRA-only on PMC-VQA.
**Reason:** PMC-VQA questions are often **figure-specific**; retrieving text from unrelated PMC papers can introduce noise. RAG shines when the answer is present in the retrieved corpus and aligns with the query context.

**Next iteration actions:**

* **Use RAG for “medical PDF QA” tasks** (guidelines/papers) rather than figure-only MCQ benchmarks.
* **Build “in-article RAG”**: index the *same* article text/captions associated with each figure when possible (requires mapping figure → PMCID/XML).
* Add a simple **retrieval confidence gate**:

  * if top score < threshold → disable RAG and answer from image only.

## 3) Data quality / corrupted archives in PMC OA download

**Observed:** occasional corrupted `.tar.gz` causing extraction errors (`zlib.error`).
**Next iteration actions:**

* keep retry + re-download logic (already implemented)
* add checksum verification if available
* log fail rate and automatically replace missing docs to keep corpus size stable

## 4) More meaningful evaluation for medical RAG

Current benchmark set is mainly “doc skills + PMC-VQA.”
**Next iteration actions:**

* Add a small **Medical PDF QA evaluation set** (50–200 questions) built from open-access PMC OA PDFs/guidelines:

  * answerable by retrieved evidence
  * requires citations (evidence page/section)
* Report:

  * Answer accuracy
  * Evidence overlap / citation correctness
  * Hallucination rate (no-evidence answers)

<!-- ---

# Analysis (Why did general DocVQA/ChartQA drop?)

This behavior usually indicates **domain/task overfitting** or **format bias** during fine-tuning:

1. **PMC-VQA is MCQ-style**, and if it dominates training, the model can become biased toward short “option-like” answers, hurting free-form extraction tasks (DocVQA).
2. **Prompt/answer format mismatch** between training and evaluation can cause EM to collapse:

   * If the fine-tuned model starts producing extra prefixes (`Answer: ...`, `A) ...`) or explanations, EM drops sharply.
3. **Imbalanced mixture / sampling**: even if the dataset counts look okay, training steps may still skew heavily toward one source (especially if shuffled without per-source weighting).
4. **Label masking / SFT boundary errors** can silently damage supervised learning (less likely since PMC improved, but still worth checking).

# why RAG not help PMC-VQA much

PMC-VQA questions are often visual (figure content). Your RAG corpus is text from other papers, not necessarily the paper the figure comes from, so retrieval might be weak.

RAG shines when your query is about medical PDFs/guidelines and the answer is present in text.

If you want RAG to help PMC-VQA, you’d need “in-article retrieval”: index the same article’s caption/text that corresponds to the figure (harder, but doable if the dataset provides Figure_path → PMCID mapping).

--- -->