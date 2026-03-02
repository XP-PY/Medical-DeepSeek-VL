#!/usr/bin/env python3
import os, csv, subprocess

MANIFEST = "corpus/pmc_oa_1k/manifest.tsv"
BAD_LIST = "corpus/pmc_oa_1k/bad_archives.txt"
PKG_DIR = "corpus/pmc_oa_1k/packages"

def main():
    # Load bad filenames
    bad_files = []
    with open(BAD_LIST, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            bad_files.append(line.split("\t")[0].strip())
    bad_set = set(bad_files)
    if not bad_set:
        print("No bad archives found.")
        return

    # Map pmcid -> href from manifest
    pmcid2href = {}
    with open(MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pmcid2href[row["pmcid"]] = row["href"]

    os.makedirs(PKG_DIR, exist_ok=True)

    for fn in bad_files:
        pmcid = fn.replace(".tar.gz", "")
        url = pmcid2href.get(pmcid)
        if not url:
            print("Missing href for", pmcid)
            continue

        out_path = os.path.join(PKG_DIR, fn)
        # remove corrupted file first
        if os.path.exists(out_path):
            os.remove(out_path)

        print("Re-downloading", pmcid)
        # wget with retries
        subprocess.run(["wget", "-q", "--tries=10", "--timeout=30", "-O", out_path, url], check=True)

    print("Done re-download.")

if __name__ == "__main__":
    main()