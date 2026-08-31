#!/usr/bin/env python3
"""Run scripts/run_inference.py over a directory in fixed-size chunks.

One long run accumulates memory across images -- the MPS allocator holds cached
buffers -- and on a machine whose swap is already full that eventually stalls
the process rather than slowing it. Each chunk therefore runs in its own
subprocess, so the operating system reclaims everything between chunks. The
cost is one model load per chunk.

The per-chunk detailed reports are concatenated into a single file with the
same shape run_inference.py would have produced for the whole directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv/bin/python"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--detailed-output", required=True)
    parser.add_argument("--output", default=None, help="Optional merged simple JSON")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--no-transformations", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    source = Path(args.input_dir)
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        raise SystemExit(f"No images found in {source}")

    chunks = [images[i : i + args.chunk_size] for i in range(0, len(images), args.chunk_size)]
    detailed: List[dict] = []
    simple: List[dict] = []
    started = time.time()

    for index, chunk in enumerate(chunks, start=1):
        with tempfile.TemporaryDirectory(prefix="rl_chunk_") as tmp:
            staging = Path(tmp) / "images"
            staging.mkdir()
            for path in chunk:
                shutil.copy2(path, staging / path.name)
            out_simple = Path(tmp) / "simple.json"
            out_detail = Path(tmp) / "detail.json"
            command = [
                str(PYTHON), str(PROJECT_ROOT / "scripts/run_inference.py"),
                "--input-dir", str(staging),
                "--device", args.device,
                "--relative-paths", "--no-calibration", "--quiet",
                "--output", str(out_simple),
                "--detailed-output", str(out_detail),
            ]
            if args.adapter_dir:
                command += ["--adapter-dir", args.adapter_dir]
            if args.no_transformations:
                command.append("--no-transformations")
            command += [a for a in args.extra if a != "--"]

            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
                raise SystemExit(f"chunk {index} failed with code {result.returncode}")
            if out_detail.is_file():
                detailed.extend(json.loads(out_detail.read_text(encoding="utf-8")))
            if out_simple.is_file():
                simple.extend(json.loads(out_simple.read_text(encoding="utf-8")))

        done = sum(len(c) for c in chunks[:index])
        elapsed = time.time() - started
        rate = elapsed / done
        print(f"  chunk {index}/{len(chunks)}  {done}/{len(images)} images"
              f"  {elapsed/60:.1f} min elapsed  eta {(len(images)-done)*rate/60:.1f} min", flush=True)

    target = PROJECT_ROOT / args.detailed_output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(detailed, indent=1), encoding="utf-8")
    if args.output:
        merged = PROJECT_ROOT / args.output
        merged.parent.mkdir(parents=True, exist_ok=True)
        merged.write_text(json.dumps(simple, indent=1), encoding="utf-8")
    print(f"wrote {args.detailed_output}  ({len(detailed)} records, {(time.time()-started)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
