#!/usr/bin/env python3
"""One-command setup, and a doctor that says exactly what is missing.

    python3 scripts/setup.py --check     # what do I have? what is missing?
    python3 scripts/setup.py --yes       # deps + checkpoint, no prompts
    python3 scripts/setup.py             # the same, confirming the download first
    python3 scripts/setup.py --all --yes # also the evaluation dataset
    python3 scripts/setup.py --sample    # only the smoke-test images
    python3 scripts/setup.py --adapter   # only the pixel-space adapter

Runs on the system Python: it creates the virtualenv, so it cannot assume one
exists. Every step is idempotent and safe to re-run -- a half-finished
checkpoint download resumes rather than restarting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

SAMPLE_DIR = PROJECT_ROOT / "data/smoke_sample"
CHECKPOINT = PROJECT_ROOT / "models/pretrained/pytorch_model.pt"
# Pinned to a commit, not to `main`. `resolve/main` follows the branch, so if
# the publisher ever replaces the file every number this project reports would
# silently come from different weights and stop being reproducible.
CHECKPOINT_REVISION = "6455cf791436ee914c9556ab71578cce9761fef7"
CHECKPOINT_URL = (
    "https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2/"
    f"resolve/{CHECKPOINT_REVISION}/pytorch_model.pt"
)
CHECKPOINT_BYTES = 2_105_483_083
# Upstream's LFS sha256 for that revision, checked after a fresh download.
CHECKPOINT_SHA256 = "caae0c005d8e37e7aa086aa241d1c9445d296ef77649004655c14f5c81130d4b"

# The demo's adapter toggle needs these files; without them the app can only
# report "Adapter directory not found".
ADAPTER_NAME = "robustness_head"
ADAPTER_DIR = PROJECT_ROOT / "models/adapters" / ADAPTER_NAME

DATASET_DIR = PROJECT_ROOT / "data/sid_set"
CALIBRATION = PROJECT_ROOT / "outputs/calibration.json"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def colour(text: str, code: str) -> str:
    return text if not sys.stdout.isatty() else f"{code}{text}{RESET}"


def ssl_context() -> ssl.SSLContext:
    """Return an SSL context with a CA bundle that actually resolves.

    This script runs on the SYSTEM Python, which on macOS ships without a CA
    bundle unless "Install Certificates.command" was ever run -- so the default
    context fails every HTTPS request with CERTIFICATE_VERIFY_FAILED. The
    virtualenv this script just built does have certifi, so borrow its bundle.
    Falls back to the default context when the venv is not usable yet, which is
    the correct behaviour on platforms where the default already works.
    """

    context = ssl.create_default_context()
    if not VENV_PYTHON.exists():
        return context
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", "import certifi; print(certifi.where())"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return context
    cafile = result.stdout.strip()
    if result.returncode == 0 and cafile and Path(cafile).is_file():
        try:
            return ssl.create_default_context(cafile=cafile)
        except (OSError, ssl.SSLError):  # pragma: no cover - unreadable bundle
            return context
    return context


def confirm(question: str, default: bool = True) -> bool:
    """Ask a yes/no question, and survive a stdin that is not a terminal.

    The checkpoint prompt defaults to YES: downloading it is the whole point of
    running this script, so pressing Enter must not be the answer that leaves
    the project unable to start. A piped or non-interactive stdin (CI, `| tee`,
    a container build) takes the default rather than raising EOFError, which
    used to abort setup outright.
    """

    hint = "[Y/n]" if default else "[y/N]"
    if not sys.stdin or not sys.stdin.isatty():
        print(f"{question} {hint} {'y' if default else 'n'}  (stdin is not a terminal)")
        return default
    try:
        answer = input(f"{question} {hint} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def venv_status() -> tuple[bool, str]:
    if not VENV_PYTHON.exists():
        return False, "not created"
    try:
        version = subprocess.run(
            [str(VENV_PYTHON), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
        return True, f"Python {version}"
    except (subprocess.SubprocessError, OSError):
        return False, "present but unusable"


def dependency_status() -> tuple[bool, str]:
    if not VENV_PYTHON.exists():
        return False, "no virtualenv yet"
    # importlib.util must be imported explicitly; `import importlib` alone does
    # not bind the submodule on every Python version.
    probe = (
        "import importlib.util\n"
        "mods=['torch','torchvision','PIL','numpy','yaml','transformers','peft','timm',"
        "'streamlit','pandas','matplotlib','scipy','huggingface_hub','pyarrow','pytest']\n"
        "print(','.join(m for m in mods if importlib.util.find_spec(m) is None))\n"
    )
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", probe], capture_output=True, text=True, timeout=300
    )
    missing = [m for m in result.stdout.strip().split(",") if m]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, "all present"


def checkpoint_status() -> tuple[bool, str]:
    if not CHECKPOINT.exists():
        return False, "not downloaded (2.11 GB)"
    size = CHECKPOINT.stat().st_size
    if size != CHECKPOINT_BYTES:
        return False, f"incomplete: {human(size)} of {human(CHECKPOINT_BYTES)}"
    return True, f"{human(size)}, size verified"


def verify_checkpoint() -> bool:
    """True when the file on disk is the exact revision this project pins."""

    digest = hashlib.sha256()
    try:
        with CHECKPOINT.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return False
    return digest.hexdigest() == CHECKPOINT_SHA256


def adapter_status() -> tuple[bool, str]:
    weights = ADAPTER_DIR / "adapter_model.safetensors"
    head = ADAPTER_DIR / "classifier_head.pt"
    if not (weights.exists() and head.exists()):
        return False, "not downloaded (33 MB; the demo's adapter toggle needs it)"
    size = sum(item.stat().st_size for item in ADAPTER_DIR.iterdir() if item.is_file())
    return True, f"{human(size)} in models/adapters/{ADAPTER_NAME}/"


def download_adapter() -> bool:
    """Fetch the pixel-space adapter so the demo's toggle works out of the box.

    Optional on purpose: it is 33 MB of opt-in explainability, the base
    checkpoint is what actually runs by default, and a network failure here
    must not report the whole project as broken.

    Shelled out to the venv interpreter because it needs huggingface_hub, and
    this file must import nothing outside the standard library -- it runs on
    the system Python before any dependency exists.
    """

    ok, detail = adapter_status()
    if ok:
        print(f"  adapter already present ({detail})")
        return True
    if not VENV_PYTHON.exists():
        print(colour("  create the virtualenv first: python3 scripts/setup.py", RED))
        return False
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            str(PROJECT_ROOT / "scripts/download_adapters.py"),
            "--adapter",
            ADAPTER_NAME,
        ]
    )
    if result.returncode != 0:
        print(colour("  adapter download failed; the base checkpoint is unaffected", YELLOW))
        return False
    return True


def sample_status() -> tuple[bool, str]:
    images = sorted(SAMPLE_DIR.glob("*.png")) if SAMPLE_DIR.exists() else []
    if not images:
        return False, "not generated (optional; only proves the pipeline runs)"
    return True, f"{len(images)} generated image(s) in {SAMPLE_DIR.name}/"


def create_sample_images() -> bool:
    """Generate a couple of images so the CLI has something to run on.

    These are GENERATED PATTERNS, not photographs and not AI-generated
    pictures: they exist to prove the plumbing works end to end -- validation,
    transformation, batching, JSON output -- on a machine with no dataset
    downloaded. Any score they receive is meaningless as detection evidence,
    which the written README beside them says too.
    """

    if not VENV_PYTHON.exists():
        print(colour("  create the virtualenv first: python3 scripts/setup.py", RED))
        return False
    # Delegated to a separate script run by the VENV interpreter: generating
    # the images needs numpy and Pillow, and this file must keep importing
    # nothing outside the standard library because it runs before they exist.
    result = subprocess.run(
        [str(VENV_PYTHON), str(PROJECT_ROOT / "scripts/make_smoke_images.py"), str(SAMPLE_DIR)]
    )
    if result.returncode != 0:
        print(colour("  could not generate the smoke-test images", RED))
        return False
    return True


def dataset_status() -> tuple[bool, str]:
    shards = sorted(DATASET_DIR.rglob("*.parquet")) if DATASET_DIR.exists() else []
    if not shards:
        return False, "not downloaded (optional; needed only for evaluation)"
    total = sum(s.stat().st_size for s in shards)
    return True, f"{len(shards)} shard(s), {human(total)}"


def calibration_status() -> tuple[bool, str]:
    if not CALIBRATION.exists():
        return False, "not fitted (optional; scores stay uncalibrated)"
    try:
        payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        return True, f"{payload.get('method', '?')} fitted on {payload.get('fitted_on', '?')}"
    except (OSError, json.JSONDecodeError):
        return False, "present but unreadable"


CHECKS = [
    ("Virtualenv", venv_status, True, "python3 scripts/setup.py"),
    ("Dependencies", dependency_status, True, "python3 scripts/setup.py"),
    ("Model checkpoint", checkpoint_status, True, "python3 scripts/setup.py --checkpoint"),
    # Generated locally, never cloned: data/ is gitignored in full, so a
    # bundled sample could not reach a fresh clone however the repository was
    # obtained. It used to be listed as REQUIRED with "re-clone the repository"
    # as the fix, which reported every correct clone as incomplete and gave
    # advice that could not work. It is a smoke test, so it is optional.
    ("Smoke-test images", sample_status, False, "python3 scripts/setup.py --sample"),
    ("Pixel-space adapter", adapter_status, False, "python3 scripts/setup.py --adapter"),
    ("Evaluation dataset", dataset_status, False, "python3 scripts/setup.py --dataset"),
    (
        "Calibration",
        calibration_status,
        False,
        "./.venv/bin/python scripts/evaluate_confidence.py --save-calibration outputs/calibration.json",
    ),
]


def run_check() -> int:
    print("\nProject status\n" + "-" * 66)
    blocking = 0
    for name, probe, required, fix in CHECKS:
        ok, detail = probe()
        if ok:
            mark = colour("OK      ", GREEN)
        elif required:
            mark = colour("MISSING ", RED)
            blocking += 1
        else:
            mark = colour("optional", YELLOW)
        print(f"  {mark} {name:<20} {detail}")
        if not ok:
            print(f"           {colour('fix: ' + fix, DIM)}")
    print("-" * 66)
    if blocking:
        print(
            f"{colour(str(blocking) + ' required item(s) missing.', RED)} "
            f"Run: python3 scripts/setup.py --all"
        )
    else:
        print(colour("Ready. Start the demo with:", GREEN))
        print("  ./.venv/bin/streamlit run app.py")
        if SAMPLE_DIR.exists():
            print("\nOr check the command line against the generated smoke-test images:")
            relative = SAMPLE_DIR.relative_to(PROJECT_ROOT)
            print(f"  ./.venv/bin/python scripts/run_inference.py --input-dir {relative} \\")
            print("    --output outputs/predictions.json")
            print(colour("  (those images only prove the pipeline runs; see their README)", DIM))
    print()
    return 0 if blocking == 0 else 1


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def create_venv() -> bool:
    ok, detail = venv_status()
    if ok:
        print(f"  virtualenv already present ({detail})")
        return True
    print(f"  creating virtualenv at {VENV} ...")
    result = subprocess.run([sys.executable, "-m", "venv", str(VENV)])
    if result.returncode != 0:
        print(colour("  failed to create the virtualenv", RED))
        return False
    subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    return True


def install_dependencies() -> bool:
    print("  installing requirements.txt (a few minutes on first run) ...")
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")]
    )
    return result.returncode == 0


def download_checkpoint(assume_yes: bool) -> bool:
    ok, detail = checkpoint_status()
    if ok:
        print(f"  checkpoint already present ({detail})")
        return True
    if not assume_yes and not confirm(
        f"  Download the model checkpoint ({human(CHECKPOINT_BYTES)})?"
    ):
        print("  skipped. The demo cannot run without it; fetch it later with:")
        print("    python3 scripts/setup.py --checkpoint")
        return False

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    existing = CHECKPOINT.stat().st_size if CHECKPOINT.exists() else 0
    request = urllib.request.Request(CHECKPOINT_URL)
    if existing:
        # Resume rather than restart a partial 2 GB download.
        request.add_header("Range", f"bytes={existing}-")
        print(f"  resuming from {human(existing)} ...")
    else:
        print(f"  downloading {human(CHECKPOINT_BYTES)} ...")

    try:
        with urllib.request.urlopen(request, context=ssl_context()) as response:
            mode = "ab" if existing and response.status == 206 else "wb"
            if mode == "wb":
                existing = 0
            downloaded = existing
            with CHECKPOINT.open(mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    percent = 100 * downloaded / CHECKPOINT_BYTES
                    print(
                        f"\r    {human(downloaded)} / {human(CHECKPOINT_BYTES)} ({percent:5.1f}%)",
                        end="",
                        flush=True,
                    )
        print()
    except Exception as exc:  # network errors of every shape
        print(f"\n  {colour('download failed', RED)}: {type(exc).__name__}: {exc}")
        print("  Re-run this command; the download resumes where it stopped.")
        return False

    ok, detail = checkpoint_status()
    if not ok:
        print(f"  {colour('checkpoint ' + detail, RED)}")
        return False
    # Only after a fresh download: hashing 2 GB costs seconds, and the size
    # check in checkpoint_status covers the common truncated-download case.
    # This catches the rest -- a corrupted resume, or a file that is the right
    # length but not the weights this project's numbers were measured on.
    print("  verifying checksum ...")
    if not verify_checkpoint():
        print(colour("  checksum MISMATCH: the file is not the pinned revision.", RED))
        print("  Delete it and re-run to download again:")
        print(f"    rm {CHECKPOINT}")
        return False
    print(f"  {colour('checkpoint ready', GREEN)} ({detail}, checksum verified)")
    return True


def download_dataset(assume_yes: bool, shards: int) -> bool:
    ok, detail = dataset_status()
    if ok:
        print(f"  dataset already present ({detail})")
        return True
    # Defaults to no: the dataset is optional and only needed to re-run the
    # evaluation, so it should never be downloaded by an absent-minded Enter.
    if not assume_yes and not confirm(
        f"  Download {shards} SID_Set shard(s) (~{shards * 0.5:.1f} GB)?", default=False
    ):
        print("  skipped.")
        return False
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            str(PROJECT_ROOT / "scripts/download_dataset.py"),
            "--split",
            "validation",
            "--shards",
            str(shards),
            "--yes",
        ]
    )
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Set up the project, or report exactly what is missing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Report status and exit")
    parser.add_argument("--all", action="store_true", help="Also download the evaluation dataset")
    parser.add_argument("--checkpoint", action="store_true", help="Only download the checkpoint")
    parser.add_argument("--dataset", action="store_true", help="Only download the dataset")
    parser.add_argument(
        "--sample", action="store_true", help="Only generate the smoke-test images"
    )
    parser.add_argument(
        "--adapter", action="store_true", help="Only download the pixel-space adapter"
    )
    parser.add_argument("--shards", type=int, default=4, help="Dataset shards to fetch")
    parser.add_argument("--yes", "-y", action="store_true", help="Do not prompt")
    parser.add_argument("--skip-checkpoint", action="store_true", help="Set up code only")
    args = parser.parse_args(argv)

    if args.check:
        return run_check()

    print("\nSetting up: Robust Detection of AI-Generated Images\n" + "=" * 66)

    if args.checkpoint or args.dataset or args.sample or args.adapter:
        if args.checkpoint and not download_checkpoint(args.yes):
            return 1
        if args.sample and not create_sample_images():
            return 1
        if args.adapter and not download_adapter():
            return 1
        if args.dataset:
            if not VENV_PYTHON.exists():
                print(colour("  create the virtualenv first: python3 scripts/setup.py", RED))
                return 1
            if not download_dataset(args.yes, args.shards):
                return 1
        return run_check()

    print("\n[1/6] Virtualenv")
    if not create_venv():
        return 1

    print("\n[2/6] Dependencies")
    if not install_dependencies():
        print(colour("  dependency installation failed", RED))
        return 1

    print("\n[3/6] Model checkpoint")
    if args.skip_checkpoint:
        print("  skipped (--skip-checkpoint)")
    else:
        download_checkpoint(args.yes)

    print("\n[4/6] Smoke-test images")
    create_sample_images()

    # Optional, and never fatal: the demo runs on the base checkpoint alone.
    print("\n[5/6] Pixel-space adapter")
    download_adapter()

    print("\n[6/6] Evaluation dataset")
    if args.all:
        download_dataset(args.yes, args.shards)
    else:
        print("  skipped (optional; add --all, or run --dataset later)")

    return run_check()


if __name__ == "__main__":
    raise SystemExit(main())
