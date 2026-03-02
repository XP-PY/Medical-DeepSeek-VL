#!/usr/bin/env python3
"""
TODO

Step D.5 Best practices (small tweaks that improve quality a lot)
1) Better chunking (optional but recommended)

Your current extraction emits each <p> as one chunk, which is OK but sometimes too long/too short.

Upgrade later:

merge short paragraphs

split long ones by sentence

keep chunks ~200-400 words (or 800-1200 chars)

2) Query/document “instruction” for BGE (optional)

BGE is often used with different “instructions” for query and passage. A simple improvement:

query embedding text:
Represent this sentence for searching relevant passages: {question}

passage embedding text:
{chunk_text}

If you want this, I'll adjust both build_faiss_index.py and retrieval code.

3) Choose FAISS index type

For 1k docs:

flatip is fastest to implement and already instant.
For 100k+ chunks later:

switch to hnsw or IVF.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import env
import os
import json
import argparse
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", type=str, default="corpus/pmc_oa_1k/chunks.jsonl")
    ap.add_argument("--out_dir", type=str, default="corpus/pmc_oa_1k")
    ap.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_chars", type=int, default=1200, help="truncate very long chunks")
    ap.add_argument("--index_type", type=str, default="flatip", choices=["flatip", "hnsw"])
    ap.add_argument("--hnsw_m", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "chunks_meta.jsonl")
    index_path = os.path.join(args.out_dir, "faiss.index")
    stats_path = os.path.join(args.out_dir, "index_stats.json")

    # 1) Load chunks + prepare texts
    metas = []
    texts = []
    for obj in iter_jsonl(args.chunks_jsonl):
        text = (obj.get("text") or "").strip()
        if not text:
            continue
        if len(text) > args.max_chars:
            text = text[:args.max_chars]
        pmcid = obj.get("pmcid", "")
        kind = obj.get("kind", "")
        idx = obj.get("idx", -1)

        # retrieval text: include light metadata (helps in practice)
        # NOTE: keep it short; too much metadata hurts embedding
        embed_text = f"{text}"

        metas.append({
            "pmcid": pmcid,
            "kind": kind,
            "idx": idx,
            "text": text
        })
        texts.append(embed_text)

    if len(texts) == 0:
        raise RuntimeError("No valid chunks found. Check chunks.jsonl.")

    # 2) Embed
    model = SentenceTransformer(args.embed_model, cache_folder=env.MODEL_DIR)
    # Normalize embeddings => use inner product as cosine similarity
    # For bge models, you can optionally use instruction:
    # query: "Represent this sentence for searching relevant passages: ..."
    # But keep it simple first.
    all_emb = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Embedding"):
        batch = texts[i:i + args.batch_size]
        emb = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
        all_emb.append(emb.astype(np.float32))
    emb = np.vstack(all_emb)
    dim = emb.shape[1]

    # 3) Build FAISS index
    if args.index_type == "flatip":
        index = faiss.IndexFlatIP(dim)
    else:
        # HNSW for faster search (overkill for 1k docs but nice)
        index = faiss.IndexHNSWFlat(dim, args.hnsw_m)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64

    index.add(emb)

    # 4) Save
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    stats = {
        "chunks": len(metas),
        "dim": dim,
        "embed_model": args.embed_model,
        "index_type": args.index_type,
        "index_path": index_path,
        "meta_path": meta_path
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nDone!")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()