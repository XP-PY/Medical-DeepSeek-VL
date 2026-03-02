import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

class PMCOARetriever:
    """
    FAISS + SentenceTransformer retriever over PMC OA chunks
    """
    def __init__(self, index_path: str, meta_path: str, embed_model: str = "BAAI/bge-small-en-v1.5"):
        self.index = faiss.read_index(index_path)
        self.metas = list(_iter_jsonl(meta_path))
        self.model = SentenceTransformer(embed_model)

    def search(self, query: str, k: int = 5):
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, ids = self.index.search(q, k)
        hits = []
        for rank, idx in enumerate(ids[0].tolist()):
            if idx < 0:
                continue
            m = self.metas[idx]
            hits.append({
                "score": float(scores[0][rank]),
                "pmcid": m.get("pmcid", ""),
                "kind": m.get("kind", ""),
                "idx": m.get("idx", -1),
                "text": m.get("text", "")
            })
        return hits

    @staticmethod
    def format_hits(hits, max_chars: int = 1500):
        # compact evidence block
        lines, total = [], 0
        for h in hits:
            s = f"[{h['pmcid']} {h['kind']}#{h['idx']}] {h['text']}"
            if total + len(s) > max_chars:
                break
            lines.append(s)
            total += len(s)
        return "\n".join(lines)