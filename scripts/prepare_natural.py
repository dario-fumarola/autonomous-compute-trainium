"""Fetch a bounded natural-document qualification sample outside the repository."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import requests
from tokenizers import Tokenizer

from compute_trainium.data import content_hash, source_group, split_for_group


SOURCES = {
    "web": ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
            "data/CC-MAIN-2013-20/train-00000-of-00014.parquet"),
    "math": ("HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594",
             "finemath-4plus/train-00000-of-00064.parquet"),
    "code": ("codeparrot/codeparrot-clean", "35a59fb025bc0a102f7d96eac09d145b896d487b",
             "file-000000000001.json.gz"),
}
TOKENIZER_REV = "27d67f1b5f57dc0953326b2601d68371d40ea8da"


def download(url, path):
    if not path.exists():
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with path.with_suffix(path.suffix + ".partial").open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        path.with_suffix(path.suffix + ".partial").replace(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    if path.suffix == ".parquet":
        for b in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from b.to_pylist()
    else:
        with gzip.open(path, "rt") as f:
            for line in f:
                yield json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--documents-per-domain", type=int, default=600)
    p.add_argument("--max-tokens", type=int, default=8192)
    args = p.parse_args()
    root = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        raise ValueError("Store source data outside the repository.")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "corpus.json").exists():
        raise FileExistsError("Choose a new output for a new corpus.")
    tok_file = root / "tokenizer.json"
    tok_sha = download(f"https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/{TOKENIZER_REV}/tokenizer.json", tok_file)
    tokenizer = Tokenizer.from_file(str(tok_file))
    assert tokenizer.get_vocab_size() == 32000
    receipts = {}
    def fetch(item):
        domain, (repo_id, rev, file) = item
        path = root / (domain + (".parquet" if file.endswith("parquet") else ".json.gz"))
        sha = download(f"https://huggingface.co/datasets/{repo_id}/resolve/{rev}/{file}", path)
        return domain, path, {"repository": repo_id, "revision": rev, "file": file, "sha256": sha}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        downloaded = list(pool.map(fetch, SOURCES.items()))
    documents, seen = [], set()
    for domain, path, receipt in downloaded:
        accepted, rejected = 0, {}
        for row in rows(path):
            text = row.get("text", row.get("content", ""))
            source = row.get("repo_name", "") if domain == "code" else row.get("url", "")
            license_name = str(row.get("license", "")).lower()
            reason = None
            if domain == "code" and license_name not in {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"}:
                reason = "license_filter"
            try:
                group = source_group(domain, source)
            except ValueError:
                reason = "missing_source"
            key = content_hash(text)
            if not reason and key in seen:
                reason = "exact_duplicate"
            ids = tokenizer.encode(text, add_special_tokens=False).ids if not reason else []
            if not reason and not 64 <= len(ids) <= args.max_tokens:
                reason = "length_filter"
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            seen.add(key)
            documents.append({"domain": domain, "id": key,
                              "group": hashlib.sha256(group.encode()).hexdigest(),
                              "split": split_for_group(group), "tokens": ids,
                              "license": license_name, "source": source})
            accepted += 1
            if accepted >= args.documents_per_domain:
                break
        if accepted < args.documents_per_domain:
            raise ValueError(f"Not enough accepted {domain} documents: {accepted}.")
        receipts[domain] = {**receipt, "accepted": accepted, "rejected": rejected}
    corpus_path = root / "corpus.json"
    corpus_path.write_text(json.dumps(documents))
    counts = {}
    for d in documents:
        c = counts.setdefault(d["domain"] + "/" + d["split"], {"documents": 0, "real_tokens": 0})
        c["documents"] += 1
        c["real_tokens"] += len(d["tokens"])
    manifest = {"purpose": "qualification convenience sample; not a scientific benchmark",
                "synthetic_generation": False, "sources": receipts,
                "tokenizer": {"repository": "mistralai/Mistral-7B-v0.1", "revision": TOKENIZER_REV,
                              "sha256": tok_sha, "vocab_size": 32000, "pause_id": 0},
                "split_rule": "global host/repository hash 80/10/10 before windows",
                "content_dedup": "global normalized exact; near-duplicate audit outstanding",
                "counts": counts, "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
