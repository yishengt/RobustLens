"""Reference utility for creating viewable transformed test images.

This is a convenience implementation for the transformation owner. The model
and evaluation side consumes its output through the manifest contract and does
not depend on how the transformations are generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def materialize_transforms(
    input_dir: str | Path,
    output_dir: str | Path,
    config: Dict[str, Any],
    cases: Iterable[str] | None = None,
) -> List[Dict[str, str]]:
    """Write transformed copies while preserving each input's relative path."""

    from PIL import Image

    from src.data.augmentations import build_robustness_image_transform, robustness_cases
    from src.data.dataset import DEFAULT_EXTENSIONS, list_image_files, read_image, validate_image

    source_root = Path(input_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser()
    extensions = {
        str(extension).lower()
        for extension in config.get("data", {}).get("extensions", DEFAULT_EXTENSIONS)
    }
    image_paths = list_image_files(source_root, extensions)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in input directory: {source_root}")
    for path in image_paths:
        validate_image(path)
    selected_cases = list(cases) if cases is not None else robustness_cases(config)[1:]
    if not selected_cases:
        raise ValueError("At least one transformation case is required")

    manifest: List[Dict[str, str]] = []
    for case in selected_cases:
        transform = build_robustness_image_transform(config, case)
        for source in image_paths:
            relative = source.relative_to(source_root)
            target = destination_root / case / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            transformed = transform(image=read_image(source))["image"]
            Image.fromarray(transformed).save(target)
            manifest.append(
                {
                    "source_path": str(source),
                    "transformed_path": str(target),
                    "case": case,
                }
            )
    destination_root.mkdir(parents=True, exist_ok=True)
    with (destination_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize robustness test images.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional case names; default is every non-clean configured case",
    )
    args = parser.parse_args()
    from src.utils.config import load_config, resolve_config_path

    config = load_config(args.config)
    output_arg = args.output_dir or config["paths"].get(
        "robustness_dir", "data/processed/robustness"
    )
    manifest = materialize_transforms(
        resolve_config_path(config, args.input_dir),
        resolve_config_path(config, output_arg),
        config,
        args.cases,
    )
    print(f"Wrote {len(manifest)} transformed images")


if __name__ == "__main__":
    main()
