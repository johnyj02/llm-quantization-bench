#!/usr/bin/env python3
"""Build the anonymized supplementary archive: corpus, probe data, annotation, probe.

    .venv/bin/python scripts/build_release.py     # -> artifacts/release/<NAME>.tar.xz

Reads `artifacts/generations/` locally and the determinism probe passes from S3 (read-only).
🚨 Refuses to write an archive if an identifying string appears anywhere in what it would
ship, including the tar headers, which carry the builder's user name unless overwritten.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantartifact.determinism import probe_fingerprint  # noqa: E402

NAME = "quantized-eval-ordering-supplement"
OUT = REPO / "artifacts" / "release"
STAGE = OUT / "stage"
GENERATIONS = REPO / "artifacts" / "generations"
# Smoke runs are rehearsals and are left out; the superseded v1 runs are kept (README).
RUNS = ("main-v1", "sweep-v1", "h6-v1", "template-v2", "pilot-v2", "anchor-v1",
        "pilot-v1", "template-v1")
EXPECTED = {"main-v1": 105_950}   # the paper states this number
S3 = "s3://sagemaker-us-east-1-654654594225/johnyj02/QA/v0.1/checkpoints"
PROBE_DATA = {"l40s": "determinism", "a10g": "determinism-spotcheck"}
REPORTS = {"determinism.json": "l40s", "determinism_ord10.json": "l40s",
           "determinism_spotcheck_a10g.json": "a10g"}
ANNOTATION = ("sample_key.json", "sample_manifest.json", "to_annotate_A1.json",
              "to_annotate_A2.json", "labels_A1.jsonl", "labels_A2.jsonl", "adjudicated.json")
ANNOTATORS = {"Joseph Sankoorikal Johny (A1)": "A1", "Syed Ahmed Danish (A2)": "A2",
              "Joseph Sankoorikal Johny": "A1", "Syed Ahmed Danish": "A2"}
# Full names and account identifiers only: single first names occur in benchmark items.
DENY = re.compile(rb"johnyj02|sankoorikal|syed ahmed|ahmed danish|654654594225|midway|"
                  rb"/Users/|@amazon\.com", re.I)


def stage_corpus() -> dict:
    counts = {}
    for run in RUNS:
        n = 0
        for src in sorted((GENERATIONS / run).rglob("*.jsonl.gz")):
            dst = STAGE / "corpus" / run / src.relative_to(GENERATIONS / run).with_suffix("")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            n += sum(1 for line in dst.open("rb") if line.strip())
        if run in EXPECTED and n != EXPECTED[run]:
            raise SystemExit(f"{run}: {n:,} records, expected {EXPECTED[run]:,}")
        counts[run] = n
        print(f"  corpus/{run:12} {n:>8,} records")
    return counts


def stage_determinism() -> dict:
    env = {**os.environ, "AWS_PROFILE": "qa-gpu"}
    for gpu, prefix in PROBE_DATA.items():
        subprocess.run(["aws", "s3", "sync", "--quiet", "--region", "us-east-1",
                        f"{S3}/{prefix}/", str(STAGE / "determinism" / gpu),
                        "--exclude", "payload_*"], check=True, env=env)
    reports = STAGE / "determinism" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for name in REPORTS:
        shutil.copy2(REPO / "artifacts" / name, reports / name)

    # Map each report cell to the raw file behind it. IDENTITY comes from the cell's own
    # analysis file, whose name is the cell tag and whose per-pass timings make it unique;
    # the pass hashes then VERIFY the raw file. Hashes alone cannot identify a cell: bf16's
    # text is identical at every batch size, so its cells share hashes.
    analyses: dict[tuple, list[str]] = {}
    for gpu in PROBE_DATA:
        for f in (STAGE / "determinism" / gpu).glob("*.json"):
            if not f.name.startswith("raw_"):
                a = json.loads(f.read_text())
                key = (gpu, tuple(a.get("seconds_per_pass") or ()), tuple(a.get("fingerprints") or ()))
                analyses.setdefault(key, []).append(f.stem)
    index: dict[str, list[str]] = {
        f"{gpu}/{raw.name}": [] for gpu in PROBE_DATA
        for raw in sorted((STAGE / "determinism" / gpu).glob("raw_*.json"))}
    n_cells = 0
    for name, gpu in REPORTS.items():
        report = json.loads((reports / name).read_text())
        for section in ("sweep", "marlin_attribution", "autotune_contrast", "model_generality"):
            for c in report.get(section) or []:
                label = f"{name}:{section}:{c.get('model_key', '')}/{c['arm']}@{c['max_num_seqs']}"
                tags = analyses.get((gpu, tuple(c["seconds_per_pass"]), tuple(c["fingerprints"])), [])
                if len(tags) != 1:
                    raise SystemExit(f"{label}: {len(tags)} matching cell files, need exactly 1")
                raw = STAGE / "determinism" / gpu / f"raw_{tags[0]}.json"
                hashes = [probe_fingerprint(p["outputs"]) for p in json.loads(raw.read_text())["passes"]]
                if hashes != c["fingerprints"]:
                    raise SystemExit(f"{label}: {raw.name} does not hold the passes it reports")
                index[f"{gpu}/{raw.name}"].append(label)
                n_cells += 1
    index = {k: v or ["superseded"] for k, v in index.items()}
    (STAGE / "determinism" / "index.json").write_text(json.dumps(index, indent=2))
    n_used = sum(v != ["superseded"] for v in index.values())
    print(f"  determinism: {len(index)} raw files; all {n_cells} report cells matched by name "
          f"and verified by hash, using {n_used} raw files")
    return {"raw_files": len(index), "used_by_reports": n_used}


def stage_annotation() -> None:
    dst = STAGE / "annotation"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ANNOTATION:
        shutil.copy2(REPO / "annotation" / name, dst / name)
    rubric = (REPO / "annotation" / "rubric.md").read_text()
    for real, anon in ANNOTATORS.items():
        rubric = rubric.replace(real, anon)
    (dst / "rubric.md").write_text(rubric)


UNFINISHED = re.compile(rb"PROPOSED|awaiting|CONFIRM or correct|executor|planner|TODO")
PROSE = {"README.md", "rubric.md", "adjudicated.json"}


def deny_scan() -> None:
    hits = []
    for f in sorted(STAGE.rglob("*")):
        if f.is_file():
            with f.open("rb") as handle:
                for n, line in enumerate(handle, 1):
                    # Our own prose only: benchmark text and model output can say anything.
                    if f.name in PROSE and (m := UNFINISHED.search(line)):
                        hits.append(f"{f.relative_to(STAGE)}:{n}: unfinished: {m.group().decode()}")
                        break
                    if m := DENY.search(line):
                        hits.append(f"{f.relative_to(STAGE)}:{n}: {m.group().decode()}")
                        break
    if hits:
        raise SystemExit("IDENTIFYING OR UNFINISHED TEXT FOUND, no archive written:\n  " + "\n  ".join(hits))
    print("  identifying-string scan: clean")


def write_tar(counts: dict, determinism: dict) -> Path:
    files = sorted(f for f in STAGE.rglob("*") if f.is_file())
    manifest = {"records": counts, "determinism": determinism, "files": {
        str(f.relative_to(STAGE)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (STAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    files.append(STAGE / "MANIFEST.json")

    # Corpus files grouped by task file name across cells, so the same prompts sit close
    # together and the compressor's window can see the repeats.
    def order(f: Path):
        rel = f.relative_to(STAGE)
        return (0, rel.parts[1], rel.name, str(rel)) if rel.parts[0] == "corpus" else (1, str(rel))

    tar_path = OUT / f"{NAME}.tar"
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tar:
        for f in sorted(files, key=order):
            info = tar.gettarinfo(str(f), arcname=f"{NAME}/{f.relative_to(STAGE)}")
            # The header would otherwise carry the builder's user name and uid.
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 1792108800   # 2026-10-16, fixed so rebuilds are byte-identical
            info.pax_headers = {}
            with f.open("rb") as handle:
                tar.addfile(info, handle)
    with tarfile.open(tar_path) as tar:
        bad = [m.name for m in tar.getmembers() if m.uname or m.gname or m.uid or m.gid]
    if bad:
        raise SystemExit(f"tar headers still carry an owner: {bad[:3]}")
    return tar_path


def main() -> int:
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    counts = stage_corpus()
    determinism = stage_determinism()
    stage_annotation()
    for name in ("README.md", "order_invariance_probe.py"):
        shutil.copy2(REPO / "release" / name, STAGE / name)
    deny_scan()
    tar_path = write_tar(counts, determinism)
    print(f"  tar: {tar_path.stat().st_size / 1e6:,.0f} MB uncompressed")

    # xz at its strongest preset, single-threaded (threads split the input and cost ratio),
    # with a 256 MiB window so prompts repeated across cells and runs are matched.
    xz = OUT / f"{NAME}.tar.xz"
    xz.unlink(missing_ok=True)
    subprocess.run(["xz", "-T1", "-k", "--format=xz", "--lzma2=preset=9e,dict=256MiB",
                    str(tar_path)], check=True)
    subprocess.run(["xz", "-t", str(xz)], check=True)
    tar_path.unlink()
    digest = hashlib.sha256(xz.read_bytes()).hexdigest()
    (OUT / f"{NAME}.tar.xz.sha256").write_text(f"{digest}  {xz.name}\n")
    print(f"  wrote {xz.relative_to(REPO)}  {xz.stat().st_size / 1e6:.1f} MB  sha256 {digest[:16]}")
    shutil.rmtree(STAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
