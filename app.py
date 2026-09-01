"""RobustLens — AI-generated image detection under real-world transformations.

Launch with::

    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.model_loader import ModelSetupError  # noqa: E402
from src.pipeline.patches import overlay_patch_heatmap  # noqa: E402
from src.pipeline.pipeline import DetectionPipeline, PipelineResult  # noqa: E402
from src.pipeline.prediction import LABEL_AI, LABEL_AUTHENTIC  # noqa: E402
from src.pipeline.validation import ImageValidationError  # noqa: E402
from src.utils.config import load_config, resolve_config_path  # noqa: E402

DEFAULT_CONFIG = "configs/config.yaml"
# Defaults to the path scripts/setup.py downloads to, so the app works with no
# manual step. checkpoints/best.pt is still accepted if you keep one there.
DEFAULT_CHECKPOINT = "models/pretrained/pytorch_model.pt"
# Optional head-only adapter. Not adopted as the shipped default: it improves
# pixel-space diffusion (ADM recall 0.305 -> 0.863) and does not improve the
# DALL-E benchmarks, so the base checkpoint stays the default and this is opt-in.
DEFAULT_ADAPTER = "models/adapters/robustness_head"

STYLE = """
<style>
  #MainMenu, footer, header {visibility: hidden;}
  .block-container {padding-top: 3rem; padding-bottom: 4rem; max-width: 1100px;}
  h1, h2, h3 {letter-spacing: -0.02em; font-weight: 600;}
  h1 {font-size: 1.9rem !important; margin-bottom: .1rem;}
  h3 {font-size: 1.05rem !important; margin-top: 1.4rem;}
  [data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 600;}
  [data-testid="stMetricLabel"] {opacity: .7; font-size: .8rem;}
  .rl-sub {opacity: .6; font-size: .9rem; margin-bottom: 1.6rem;}
  .rl-verdict {font-size: 1.55rem; font-weight: 600; letter-spacing: -.02em;}
  .rl-note {opacity: .55; font-size: .8rem; line-height: 1.5;}
  hr {margin: 1.8rem 0; opacity: .12;}
</style>
"""


@st.cache_resource(show_spinner="Loading model…")
def load_pipeline(
    config_path: str, checkpoint_path: str, device: str, adapter_dir: str = ""
) -> DetectionPipeline:
    """Load the config and checkpoint once and reuse them across reruns.

    ``adapter_dir`` is part of the cache key, so switching the adapter on or off
    rebuilds the pipeline rather than silently reusing the previous weights.
    """

    config = load_config(config_path)
    pipeline = DetectionPipeline.from_checkpoint(
        checkpoint_path, config, device=None if device == "auto" else device
    )
    if adapter_dir:
        from src.finetune.model import load_saved_adapter_into_model

        load_saved_adapter_into_model(pipeline.bundle.model, adapter_dir)
    return pipeline


def _verdict_colour(label: str) -> str:
    return {LABEL_AI: "#d1493f", LABEL_AUTHENTIC: "#2e7d54"}.get(label, "#b8860b")


def pretty_version(name: str) -> str:
    """Turn an internal transform id into something a reader understands.

    Chart axes and tables are read at a glance, so ``jpeg_q70`` becomes
    "JPEG quality 70". Parsed from the naming convention rather than a fixed
    lookup, so a new transform added to the config still gets a sensible label
    instead of falling back to its raw id.
    """

    if name in ("original", "clean"):
        return "Original"
    try:
        if name.startswith("jpeg_q"):
            return f"JPEG quality {int(name[6:])}"
        if name.startswith("blur_s"):
            return f"Blur {float(name[6:]):g}px"
        if name.startswith("resize_"):
            return f"Resized to {float(name[7:].rstrip('x')) * 100:g}%"
        if name.startswith("noise_s"):
            return f"Noise {float(name[7:]) * 100:g}%"
        if name.startswith("center_crop_"):
            return f"Cropped to {int(name.rsplit('_', 1)[1])}%"
    except (ValueError, IndexError):
        pass
    if name.startswith("color_jitter"):
        suffix = name[len("color_jitter"):].strip("_")
        return f"Colour shift {suffix}" if suffix else "Colour shift"
    return name.replace("_", " ").capitalize()


def render_headline(result: PipelineResult) -> None:
    """Verdict, probability, confidence and stability."""

    colour = _verdict_colour(result.label)
    st.markdown(
        f"<div class='rl-verdict' style='color:{colour}'>{result.label}</div>",
        unsafe_allow_html=True,
    )

    columns = st.columns(3)
    columns[0].metric("AI-generated", f"{result.ai_probability:.1%}")
    columns[1].metric("Confidence", result.confidence.level)
    columns[2].metric("Stability", f"{result.consistency.consistency_score:.0%}")
    st.progress(min(1.0, max(0.0, float(result.ai_probability))))

    decision = result.abstention
    if decision is not None and decision.abstain:
        st.info(decision.statement)


def render_calibration_status(pipeline: DetectionPipeline) -> None:
    """One compact line saying what the number is and where the threshold came from.

    Compact, but never omitted. Four states must stay distinguishable: a
    calibrated probability versus a raw model score, and a threshold fitted on
    data versus an interface default. Presenting a raw score beside a default
    threshold as though both were derived from data would overstate what the
    system knows, so the line is always shown and the detail is one click away.
    """

    status = pipeline.calibration_status()
    calibrated = status["calibrated"]
    derived = status["threshold_source"].startswith("data-derived")

    probability_text = "Calibrated probability" if calibrated else "Uncalibrated score"
    threshold_text = (
        f"threshold {status['threshold']:.2f} from data"
        if derived
        else f"threshold {status['threshold']:.2f} (default)"
    )
    marker = "✓" if calibrated and derived else "!"
    st.markdown(
        f"<div class='rl-note'>{marker} {probability_text} · {threshold_text}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Scoring details"):
        st.write(status["probability_note"])
        st.write(status["threshold_note"])
        bands = status["label_bands"]
        if bands.get("authentic_max") is not None:
            st.caption(
                f"Authentic below {bands['authentic_max']:.2f} · "
                f"AI-generated at or above {bands['ai_min']:.2f} "
                f"({status['label_bands_source']})."
            )
        if not calibrated:
            st.caption(
                "Fit calibration on labelled validation data to replace both with "
                "data-derived values: `scripts/calibrate_threshold.py`"
            )


def robustness_frame(result: PipelineResult) -> pd.DataFrame:
    """Per-version scores as percentages, in the order they were generated."""

    rows: List[Dict[str, Any]] = [
        {
            "Version": pretty_version(item.name),
            "Likelihood": round(item.ai_probability * 100, 1),
            "Type": "Original" if item.is_original else "Transformed",
        }
        for item in result.predictions
    ]
    return pd.DataFrame(rows)


def robustness_chart(frame: pd.DataFrame, threshold_percent: float) -> "alt.LayerChart":
    """Horizontal bars, one row per version, with the decision threshold marked.

    Horizontal because the labels are words: fifteen rotated captions along an
    x-axis are unreadable, while fifteen left-aligned rows are scannable. The
    order is the order the versions were generated -- sorting by score would
    scatter the JPEG, blur and resize families and hide the pattern the chart
    exists to show.
    """

    order = frame["Version"].tolist()
    base = alt.Chart(frame).encode(
        y=alt.Y("Version:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X(
            "Likelihood:Q",
            title="Likelihood AI-generated (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
    )
    bars = base.mark_bar(height=14, cornerRadiusEnd=2).encode(
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(
                domain=["Original", "Transformed"], range=["#3d6ea8", "#9bb4d0"]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("Version:N", title="Version"),
            alt.Tooltip("Likelihood:Q", title="Likelihood (%)", format=".1f"),
        ],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=11, color="#666").encode(
        text=alt.Text("Likelihood:Q", format=".1f")
    )
    threshold = (
        alt.Chart(pd.DataFrame({"Threshold": [threshold_percent]}))
        .mark_rule(strokeDash=[4, 4], color="#d1493f", size=1)
        .encode(x=alt.X("Threshold:Q", scale=alt.Scale(domain=[0, 100])))
    )
    return (bars + labels + threshold).properties(height=max(220, 26 * len(frame)))


def render_robustness(result: PipelineResult) -> None:
    """How the score moves across transformed versions of the same image."""

    st.markdown("### Robustness")
    frame = robustness_frame(result)
    st.altair_chart(
        robustness_chart(frame, float(result.threshold_used) * 100.0),
        use_container_width=True,
    )
    st.caption(
        f"Dashed line marks the {result.threshold_used:.0%} decision threshold. "
        f"Bars to its right were called AI-generated."
    )

    consistency = result.consistency
    columns = st.columns(4)
    columns[0].metric("Versions tested", consistency.num_versions)
    columns[1].metric("Versions agreeing", f"{consistency.agreement:.0%}")
    columns[2].metric("Spread", f"{consistency.score_range * 100:.1f}%")
    columns[3].metric("Typical variation", f"{consistency.std * 100:.1f}%")
    st.caption(
        f"Lowest {consistency.minimum:.1%} · highest {consistency.maximum:.1%} across "
        f"{consistency.num_versions} versions of the same image."
    )


def _weight_of(container: Any, key: str) -> float:
    """Read one fusion/confidence weight, treating an absent term as zero."""

    weights = getattr(container, "weights", None) or {}
    try:
        return float(weights.get(key, 0.0))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


def _patch_weight_caption(result: PipelineResult) -> str:
    """Say what region evidence actually contributed to this result."""

    probability_weight = _weight_of(getattr(result, "fusion", None), "patch")
    confidence_weight = _weight_of(getattr(result, "confidence", None), "patch_agreement")

    if probability_weight <= 0 and confidence_weight <= 0:
        return (
            "Region evidence carried zero weight in both the reported probability "
            "and the confidence score — it is shown to indicate where to look, and "
            "did not move the result."
        )
    parts = []
    if probability_weight > 0:
        parts.append(f"{probability_weight:.0%} of the reported probability")
    if confidence_weight > 0:
        parts.append(f"{confidence_weight:.0%} of the confidence score")
    return (
        f"Region evidence contributed {' and '.join(parts)} for this image, so it "
        f"did move the result. It still is not proof that any region was edited."
    )


def render_patches(result: PipelineResult) -> None:
    """Region heat map, with the caveats that keep it from being over-read."""

    report = result.patches
    if report is None or not report.available:
        return

    st.markdown("### Regions")
    if result.original_image is not None:
        overlay = overlay_patch_heatmap(result.original_image, report)
        if overlay is not None:
            left, right = st.columns(2)
            left.image(result.original_image, use_container_width=True)
            right.image(overlay, use_container_width=True)

    columns = st.columns(3)
    columns[0].metric("Regions scored", len(report.patches))
    columns[1].metric("Highest", f"{report.max_probability:.1%}")
    if report.coverage is not None:
        columns[2].metric("Coverage", f"{float(report.coverage.mean()):.0%}")

    if report.coverage is not None and float(report.coverage.mean()) < 0.999:
        st.caption(
            f"Regions cover {report.coverage.mean():.0%} of the image; uncovered areas are "
            f"left untinted because nothing was measured there, not because they look authentic."
        )

    st.caption(
        "A highlighted region is a suspicious region that influenced the model's "
        "score. It is not proof of AI editing, a segmentation mask, or a "
        "reconstruction of editing history."
    )
    # Read the weights this analysis actually used rather than asserting they
    # are zero. They are zero under the shipped config, but both are
    # configurable -- fusion.mode: whole_patch_transform gives region evidence
    # 20% of the probability -- and a hardcoded claim would then be a false
    # statement about the very number displayed above it.
    st.caption(_patch_weight_caption(result))

    # Every scored region, not just the outlined ones. `patches.top_k` controls
    # how many boxes get drawn on the overlay -- outlining all of them would be
    # visual noise -- but the table must account for the count reported above,
    # otherwise "12 regions scored" sits next to a three-row table.
    outlined = {(patch.x, patch.y, patch.width, patch.height) for patch in report.top_patches}
    ranked = sorted(report.patches, key=lambda patch: patch.ai_probability, reverse=True)
    with st.expander(f"All {len(ranked)} region scores"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rank": index,
                        "Position": f"{patch.x}, {patch.y}",
                        "Size": f"{patch.width} x {patch.height}",
                        "Likelihood AI-generated": f"{patch.ai_probability:.1%}",
                        "Outlined": "yes"
                        if (patch.x, patch.y, patch.width, patch.height) in outlined
                        else "",
                    }
                    for index, patch in enumerate(ranked, start=1)
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Sorted by score. The {len(outlined)} highest are outlined on the image above."
        )


def render_errors(result: PipelineResult) -> None:
    if not result.errors:
        return
    with st.expander(f"{len(result.errors)} non-fatal issue(s)"):
        for error in result.errors:
            st.write(f"**{error.get('stage', 'pipeline')}**: {error.get('error')}")


def main() -> None:
    st.set_page_config(page_title="RobustLens", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)

    st.markdown("# RobustLens")
    st.markdown(
        "<div class='rl-sub'>AI-generated image detection under real-world "
        "transformations</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Settings")
        device_choice = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
        adapter_on = st.toggle(
            "Pixel-space adapter",
            value=False,
            help=(
                "Applies the head-only adapter trained on six generators. Raises ADM "
                "recall from 0.305 to 0.863 and GLIDE from 0.739 to 0.967, at the cost "
                "of more false positives (0.013 to 0.072). It does not improve DALL-E "
                "detection, so the base checkpoint is the default."
            ),
        )
        with st.expander("Advanced"):
            config_path = st.text_input("Config", DEFAULT_CONFIG)
            checkpoint_input = st.text_input("Checkpoint", DEFAULT_CHECKPOINT)
            adapter_input = st.text_input("Adapter directory", DEFAULT_ADAPTER)

    try:
        config = load_config(config_path)
        checkpoint_path = resolve_config_path(config, checkpoint_input)
        adapter_path = ""
        if adapter_on:
            resolved = Path(resolve_config_path(config, adapter_input))
            if not resolved.is_dir():
                st.error(f"Adapter directory not found: {resolved}")
                st.stop()
            adapter_path = str(resolved)
        pipeline = load_pipeline(
            str(Path(config_path).expanduser().resolve()),
            str(checkpoint_path),
            device_choice,
            adapter_path,
        )
        if adapter_path:
            st.sidebar.caption("Adapter active — pixel-space diffusion")
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    except ModelSetupError as exc:
        st.error("Model not found")
        st.code(str(exc))
        st.stop()

    summary = pipeline.bundle.summary()
    with st.sidebar:
        st.divider()
        st.markdown(
            f"<div class='rl-note'>{summary['parameters_millions']:.0f}M parameters<br>"
            f"{summary['device']}<br>"
            f"{len(pipeline.transform_specs)} transformations</div>",
            unsafe_allow_html=True,
        )

    uploaded = st.file_uploader(
        "Image", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False
    )
    if uploaded is None:
        return

    try:
        with st.spinner("Analysing…"):
            result = pipeline.analyse_bytes(uploaded.getvalue(), uploaded.name)
    except ImageValidationError as exc:
        st.error(f"Invalid image: {exc}")
        return
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        return

    image_column, result_column = st.columns([1, 1.3])
    with image_column:
        if result.original_image is not None:
            st.image(result.original_image, use_container_width=True)
    with result_column:
        render_headline(result)
        render_calibration_status(pipeline)

    st.divider()
    render_robustness(result)

    if result.patches is not None and result.patches.available:
        st.divider()
        render_patches(result)

    render_errors(result)

    st.divider()
    detailed = result.as_detailed_dict()
    st.download_button(
        "Download JSON",
        data=json.dumps(detailed, indent=2),
        file_name=f"{Path(result.image_path).stem or 'result'}.json",
        mime="application/json",
    )
    st.markdown(
        f"<div class='rl-note'>{result.confidence.statement}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
