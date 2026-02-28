<!-- # Dataset Construction
* **Stage A (doc skills):** train on DocVQA/ChartQA/OCRBench → learn layout/table/chart/OCR well
* **Stage B (medical domain):** add **medical VQA from biomedical papers** (best match: **PMC-VQA**) → learn medical terminology + biomedical figure/table style
* **Stage C (medical focus without labeling):** inference-time **PDF RAG** over PubMed/Guidelines for “medical-ness” and evidence

# Training Plan
* Trained doc-understanding on DocVQA/ChartQA/OCRBench
* Added PMC-VQA to inject biomedical terminology + paper-style figures/tables
* Final system answers questions over medical PDFs with evidence via RAG

# Schedule
| Stage | Task content | Remark |
|:---:|:---:|:---:|
| **Set up environment** | Create a conda/venv env | - |
| **Simple Test** | Test base function (env.py/utils.py) | Run DeepSeeK-VL2 successfully |
| **Download datasets** | Download open source datasets ([OCRBench](https://huggingface.co/datasets/echo840/OCRBench)/[ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)/[DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)/[PMC-VQA](https://huggingface.co/datasets/hamzamooraj99/PMC-VQA-1)) | Three **doc skills datasets** and one **medical domain dataset** |
| **Build mix train/val dataset** | Sample data from OCRBench/ChartQA/DocVQA/PMC-VQA to build train/val dataset | **Train data nummber:** 92000; **Val data nummber:** 2000 |
| **Training** | Fine tune Deepseek-VL2 using LoRA | - |
| **Baseline evaluation** | Baseline evaluation (zero-shot) on Val datasets | OCRBench/ChartQA/DocVQA metric: EM/Fuzzy; PMC-VQA metric: MCQ Accuracy |
| **Fine-tuned model evaluation** | Fine-tuned model evaluation (zero-shot) on Val datasets | Metrics same as baseline | -->


# Schedule

|              Stage              | Task content                             | Remark                            |
| :-----------------------------: | :--------------------------------------- | :-------------------------------- |
|      **Set up environment**     | Create conda env                         | ✅                                 |
|         **Simple Test**         | Verify DeepSeek-VL2 inference            | ✅                                 |
|      **Download datasets**      | OCRBench / ChartQA / DocVQA / PMC-VQA    | ✅                                 |
| **Build mix train/val dataset** | Sample and build JSONL                   | **Train:** 92,000; **Val:** 2,000 |
|           **Training**          | LoRA fine-tune DeepSeek-VL2              | ✅                                 |
|     **Baseline evaluation**     | Zero-shot on val subsets                 | ✅                                 |
|    **Fine-tuned evaluation**    | Evaluate LoRA model on same subsets      | ✅                                 |
|          **Iteration**          | rebalance mixture + output normalization | 🔜                                |
|             **Demo**            | Gradio PDF QA + evidence                 | 🔜                                |

# Results

## Evaluation Setup

* **Base model:** `deepseek-ai/deepseek-vl2-tiny`
* **Fine-tuning:** LoRA SFT on a mixed dataset (DocVQA / ChartQA / OCRBench / PMC-VQA)
* **Validation:** 500 samples per dataset
* **Metrics:**

  * **ChartQA / OCRBench / DocVQA:** Exact Match (**EM**) + RapidFuzz (**Fuzzy**)
  * **PMC-VQA:** Multiple-choice **Accuracy** (A/B/C/D)

## Metrics Summary (n=500 each)

| Dataset      | Metric | Baseline | Fine-tuned |          $\Delta$ |
| ------------ | -----: | -------: | ---------: | ---------: |
| **ChartQA**  |     EM |    0.692 |      0.526 |     -0.166 |
|              |  Fuzzy |   0.8361 |     0.8306 |    -0.0056 |
| **OCRBench** |     EM |    0.688 |      0.424 |     -0.264 |
|              |  Fuzzy |   0.8785 |     0.9040 |    +0.0255 |
| **DocVQA**   |     EM |    0.758 |      0.122 |     -0.636 |
|              |  Fuzzy |   0.9336 |     0.8862 |    -0.0474 |
| **PMC-VQA**  |    ACC |    0.362 |      0.480 | **+0.118** |

### Key Takeaway

* LoRA fine-tuning **significantly improved medical-domain VQA** performance (**PMC-VQA +11.8% ACC**).
* However, performance on general document QA benchmarks **dropped**, especially **DocVQA EM**.

<!-- ---

# Analysis (Why did general DocVQA/ChartQA drop?)

This behavior usually indicates **domain/task overfitting** or **format bias** during fine-tuning:

1. **PMC-VQA is MCQ-style**, and if it dominates training, the model can become biased toward short “option-like” answers, hurting free-form extraction tasks (DocVQA).
2. **Prompt/answer format mismatch** between training and evaluation can cause EM to collapse:

   * If the fine-tuned model starts producing extra prefixes (`Answer: ...`, `A) ...`) or explanations, EM drops sharply.
3. **Imbalanced mixture / sampling**: even if the dataset counts look okay, training steps may still skew heavily toward one source (especially if shuffled without per-source weighting).
4. **Label masking / SFT boundary errors** can silently damage supervised learning (less likely since PMC improved, but still worth checking).

---

# Next Iteration (Planned Improvements)

To keep the medical gain **without sacrificing doc skills**, the next iteration will apply:

## 1) Mixture Rebalancing

* **Cap PMC-VQA ratio** (e.g., 20–30% steps)
* Increase **DocVQA/ChartQA** share
* Use a **weighted sampler** rather than naive concatenation + shuffle

## 2) Output Format Normalization (Critical for EM)

Train and evaluate with a strict format:

* For free-form datasets: output **only the final answer string**
* For MCQ datasets: output **only A/B/C/D**
* Add post-processing in eval to strip prefixes like `Answer:` / whitespace / punctuation.

## 3) Two-Stage Fine-tuning (Safer)

* Stage 1: doc skills (ChartQA/OCRBench/DocVQA)
* Stage 2: add medical domain (PMC-VQA) with smaller LR and fewer steps

## 4) Add “Medical PDF QA” Demo (System Contribution)

Deploy a Gradio demo:

* PDF upload → page rendering → OCR/text-block extraction → retrieval → VL answer
* Show **Answer + Evidence** (page index + retrieved snippet)

--- -->

# Project Pipeline

## Dataset Construction

* **Stage A (doc skills):** train on DocVQA/ChartQA/OCRBench → learn layout/table/chart/OCR
* **Stage B (medical domain):** add **PMC-VQA** → learn medical terminology + biomedical paper figure/table style
* **Stage C (medical focus without labeling):** inference-time **PDF RAG** over PubMed/Guidelines for medical PDF QA with evidence

## Training Plan

* Trained doc-understanding on DocVQA/ChartQA/OCRBench
* Added PMC-VQA to inject biomedical terminology + paper-style figures/tables
* Final system answers questions over medical PDFs with evidence via RAG (in progress)