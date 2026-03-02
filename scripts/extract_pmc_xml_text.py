#!/usr/bin/env python3
"""
This:
- finds the main *.nxml file inside each tar.gz
- extracts title/abstract/paragraphs
- writes a chunk JSONL you can embed
"""
import os, tarfile, json, gzip
from bs4 import BeautifulSoup
from tqdm import tqdm

PKG_DIR = "corpus/pmc_oa_1k/packages"
OUT_JSONL = "corpus/pmc_oa_1k/chunks.jsonl"
BAD_LIST = "corpus/pmc_oa_1k/bad_archives.txt"

def extract_text_from_nxml(nxml_bytes: bytes):
    soup = BeautifulSoup(nxml_bytes, "lxml-xml")
    title = soup.find("article-title")
    title = title.get_text(" ", strip=True) if title else ""

    abstract = soup.find("abstract")
    abstract_text = abstract.get_text(" ", strip=True) if abstract else ""

    ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    ps = [x for x in ps if x]
    return title, abstract_text, ps

def main():
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    bad = []

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for fn in tqdm(sorted(os.listdir(PKG_DIR)), desc="Extracting"):
            if not fn.endswith(".tar.gz"):
                continue
            pmcid = fn.replace(".tar.gz", "")
            path = os.path.join(PKG_DIR, fn)

            try:
                with tarfile.open(path, "r:gz") as tar:
                    # streaming iteration is often more robust than getmembers()
                    nxml_member = None
                    for m in tar:
                        if m.name.endswith(".nxml"):
                            nxml_member = m
                            break
                    if nxml_member is None:
                        continue
                    f = tar.extractfile(nxml_member)
                    if f is None:
                        continue
                    nxml = f.read()

                title, abstract, ps = extract_text_from_nxml(nxml)

                def emit(text, kind, idx):
                    if not text:
                        return
                    out.write(json.dumps({
                        "pmcid": pmcid,
                        "kind": kind,
                        "idx": idx,
                        "text": text
                    }, ensure_ascii=False) + "\n")

                emit(title, "title", 0)
                emit(abstract, "abstract", 0)
                for i, p in enumerate(ps):
                    emit(p, "p", i)

            except (tarfile.ReadError, EOFError, gzip.BadGzipFile, OSError, Exception) as e:
                bad.append((fn, str(e)))
                continue

    with open(BAD_LIST, "w", encoding="utf-8") as f:
        for fn, err in bad:
            f.write(f"{fn}\t{err}\n")

    print(f"Wrote chunks: {OUT_JSONL}")
    print(f"Bad archives: {len(bad)} -> {BAD_LIST}")

if __name__ == "__main__":
    main()