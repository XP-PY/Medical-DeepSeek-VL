#!/usr/bin/env python3
"""
Download ~1k PMC Open Access (OA) article packages (tar.gz) for building a medical RAG corpus.

Design goals:
- Discover OA packages via the official OA Web Service API (oa.fcgi).
- Prefer HTTPS over FTP for reliability.
- Retry/resume downloads.
- Do NOT crash on missing/bad links; log failures.
- Over-collect links to ensure we end up with ~target successful packages.

Outputs:
  corpus/pmc_oa_1k/
    manifest.tsv
    failed_downloads.tsv
    packages/
      PMCxxxxxxx.tar.gz

Run:
  python scripts/download_pmc_oa_1k.py

Optional args:
  --out_dir corpus/pmc_oa_1k
  --target 1000
  --collect_target 1300
  --from_date 2018-01-01
  --sleep 0.2
  --license_mode oa_comm   (only CC0/CC-BY* without NC)  [default]
  --license_mode any       (no license filtering)
"""

import os
import time
import glob
import argparse
import subprocess
from typing import Dict, List, Tuple, Optional

import requests
from lxml import etree
from tqdm import tqdm

OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


# ----------------------------
# OA API helpers
# ----------------------------
def fetch_xml(params: Dict[str, str]) -> bytes:
    r = requests.get(OA_API, params=params, timeout=60)
    r.raise_for_status()
    return r.content


def parse_records(xml_bytes: bytes, want_format: str = "tgz") -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Returns:
      records: list of {"pmcid", "license", "href"} for records that have desired format link
      token: resumptionToken (string) if more pages exist
    """
    root = etree.fromstring(xml_bytes)
    records = root.xpath(".//record")
    out = []
    for rec in records:
        pmcid = rec.get("id")
        license_ = rec.get("license") or ""
        # pick first link matching format (tgz)
        links = rec.xpath(".//link")
        href = None
        for lk in links:
            if lk.get("format") == want_format:
                href = lk.get("href")
                break
        if pmcid and href:
            out.append({"pmcid": pmcid, "license": license_, "href": href})

    # resumption token
    token = None
    res = root.xpath(".//resumption/link")
    if len(res) > 0:
        token = res[0].get("token")
    return out, token


# ----------------------------
# License filtering
# ----------------------------
def is_oa_comm(license_str: str) -> bool:
    """
    Approximation for "commercial reuse allowed" style licenses:
      - CC0
      - CC BY (no NC)
      - CC BY-SA (no NC)
      - CC BY-ND (no NC)
    """
    if not license_str:
        return False
    lic = license_str.upper().strip()
    if lic.startswith("CC0"):
        return True
    if lic.startswith("CC BY") and "NC" not in lic:
        return True
    return False


# ----------------------------
# Download helpers
# ----------------------------
def ftp_to_https(url: str) -> str:
    # Convert: ftp://ftp.ncbi.nlm.nih.gov/... -> https://ftp.ncbi.nlm.nih.gov/...
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return url.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
    return url


def wget_download(url: str, out_path: str, tries: int = 5, timeout: int = 60, passive_ftp: bool = False) -> bool:
    cmd = [
        "wget",
        "-q",
        "-c",  # resume
        f"--tries={tries}",
        f"--timeout={timeout}",
        "-O",
        out_path,
        url,
    ]
    if passive_ftp:
        cmd.insert(3, "--passive-ftp")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def download_package(url: str, out_path: str) -> bool:
    """
    Robust:
      1) try HTTPS
      2) fallback to FTP passive mode
      3) return True/False without raising
    """
    https_url = ftp_to_https(url)

    # 1) try HTTPS
    if wget_download(https_url, out_path, tries=5, timeout=60, passive_ftp=False):
        return True

    # 2) fallback to FTP passive
    if wget_download(url, out_path, tries=3, timeout=60, passive_ftp=True):
        return True

    return False


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="corpus/pmc_oa_1k")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--collect_target", type=int, default=1300, help="collect extra links to offset dead packages")
    ap.add_argument("--from_date", type=str, default="2018-01-01", help="OA API 'from' date (YYYY-MM-DD)")
    ap.add_argument("--sleep", type=float, default=0.2, help="sleep between OA API pages")
    ap.add_argument("--license_mode", type=str, default="oa_comm", choices=["oa_comm", "any"])
    args = ap.parse_args()

    out_dir = args.out_dir
    pkg_dir = os.path.join(out_dir, "packages")
    os.makedirs(pkg_dir, exist_ok=True)

    manifest_path = os.path.join(out_dir, "manifest.tsv")
    failed_path = os.path.join(out_dir, "failed_downloads.tsv")

    target = args.target
    collect_target = max(args.collect_target, target)

    got: List[Dict[str, str]] = []
    seen = set()

    # ----------------------------
    # Discover tgz links via OA API
    # ----------------------------
    params = {"from": args.from_date}
    token = None

    pbar = tqdm(total=collect_target, desc=f"Collecting tgz links ({args.license_mode})")
    while len(got) < collect_target:
        if token:
            xml = fetch_xml({"resumptionToken": token})
        else:
            xml = fetch_xml(params)

        recs, token = parse_records(xml, want_format="tgz")

        for r in recs:
            if len(got) >= collect_target:
                break
            pmcid = r["pmcid"]
            if pmcid in seen:
                continue

            if args.license_mode == "oa_comm":
                if not is_oa_comm(r.get("license", "")):
                    continue

            seen.add(pmcid)
            got.append(r)
            pbar.update(1)

        if not token and len(got) < collect_target:
            raise RuntimeError(
                f"Ran out of OA API results before reaching collect_target={collect_target}. "
                f"Try an earlier --from_date (e.g., 2015-01-01)."
            )

        time.sleep(args.sleep)
    pbar.close()

    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("pmcid\tlicense\thref\n")
        for r in got:
            f.write(f"{r['pmcid']}\t{r.get('license','')}\t{r['href']}\n")
    print("Saved manifest:", manifest_path)

    # ----------------------------
    # Download packages until we have `target` successful files
    # ----------------------------
    failed: List[Tuple[str, str]] = []

    def count_downloaded() -> int:
        return len(glob.glob(os.path.join(pkg_dir, "PMC*.tar.gz")))

    # Resume-friendly: skip if file exists and size > 0
    downloaded_before = count_downloaded()
    if downloaded_before:
        print(f"Already downloaded: {downloaded_before} packages (resume mode)")

    # Iterate through collected list, download, and stop once we hit target
    for r in tqdm(got, desc="Downloading packages"):
        if count_downloaded() >= target:
            break

        pmcid = r["pmcid"]
        url = r["href"]
        out_path = os.path.join(pkg_dir, f"{pmcid}.tar.gz")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        ok = download_package(url, out_path)

        if not ok:
            # remove zero-byte file if any
            if os.path.exists(out_path) and os.path.getsize(out_path) == 0:
                os.remove(out_path)
            failed.append((pmcid, url))

    downloaded = count_downloaded()
    print(f"Downloaded packages: {downloaded} (target={target})")

    # Save failures
    if failed:
        with open(failed_path, "w", encoding="utf-8") as f:
            f.write("pmcid\thref\n")
            for pmcid, url in failed:
                f.write(f"{pmcid}\t{url}\n")
        print(f"Failed downloads: {len(failed)} (saved to {failed_path})")

    if downloaded < target:
        print(
            "\nWARNING: downloaded < target. "
            "Increase --collect_target (e.g., 1600) and rerun; "
            "it will resume and skip already downloaded files."
        )
    else:
        print("Done. Packages in:", pkg_dir)


if __name__ == "__main__":
    main()