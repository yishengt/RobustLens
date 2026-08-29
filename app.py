"""Streamlit demo for the robust AI-generated image detector.

Launch with::

    streamlit run app.py

Upload one image to see the final classification, the AI and real
probabilities, the confidence level, the transformation-consistency score, the
per-transformation predictions, and a Grad-CAM heatmap where available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.confidence import (  # noqa: E402
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
)
from src.pipeline.model_loader import ModelSetupError  # noqa: E402
from src.pipeline.patches import overlay_patch_heatmap  # noqa: E402
from src.pipeline.pipeline import DetectionPipeline, PipelineResult  # noqa: E402
from src.pipeline.prediction import LABEL_AI, LABEL_AUTHENTIC  # noqa: E402
from src.pipeline.validation import ImageValidationError  # noqa: E402
from src.utils.config import load_config, resolve_config_path  # noqa: E402

DEFAULT_CONFIG = "configs/config.yaml"
DEFAULT_CHECKPOINT = "checkpoints/best.pt"


@st.cache_resource(show_spinner="Loading detector checkpoint...")
def load_pipeline(config_path: str, checkpoint_path: str, device: str) -> DetectionPipeline:
    """Load the config and checkpoint once and reuse them across reruns."""

    config = load_config(config_path)
    return DetectionPipeline.from_checkpoint(
        checkpoint_path, config, device=None if device == "auto" else device
    )


def label_badge(label: str) -> str:
    """Return a coloured markdown badge for the verdict."""

    colour = {LABEL_AI: "red", LABEL_AUTHENTIC: "green"}.get(label, "orange")
    return f":{colour}[**{label}**]"


def confidence_badge(level: str) -> str:
    colour = {CONFIDENCE_HIGH: "green", CONFIDENCE_MEDIUM: "orange", CONFIDENCE_LOW: "red"}
    return f":{colour.get(level, 'gray')}[**{level}**]"


def is_calibrated(pipeline: DetectionPipeline) -> bool:
    """True when persisted calibration parameters were loaded."""

    return getattr(pipeline, "calibrator", None) is not None


def render_metadata(result: PipelineResult) -> None:
    metadata = result.metadata
    if metadata is None:
        return
    st.caption(
        f"{metadata.filename} · {metadata.file_type} · {metadata.file_size_human} · "
        f"{metadata.width}x{metadata.height} · original colour mode {metadata.color_mode}"
    )


def render_headline(result: PipelineResult) -> None:
    """Show the verdict, probabilities, confidence and consistency."""

    st.markdown(f"### Result: {label_badge(result.label)}")
    st.write(result.confidence.statement)

    columns = st.columns(4)
    columns[0].metric("AI-generated probability", f"{result.ai_probability:.1%}")
    columns[1].metric("Real-image probability", f"{result.real_probability:.1%}")
    columns[2].metric("Confidence", result.confidence.level)
    columns[3].metric("Transformation consistency", f"{result.consistency.consistency_score:.1%}")
    st.progress(
        min(1.0, max(0.0, float(result.ai_probability))),
        text=f"AI-generated likelihood: {result.ai_probability:.1%}",
    )
    st.markdown(
        f"Confidence level: {confidence_badge(result.confidence.level)} · "
        f"estimated transformation sensitivity: **{result.manipulation_severity}**"
    )


def render_per_transformation(result: PipelineResult) -> None:
    """Table and chart of the per-image-version predictions."""

    rows: List[Dict[str, Any]] = [
        {
            "Version": item.name,
            "AI probability": round(item.ai_probability, 4),
            "Real probability": round(item.real_probability, 4),
            "Label": item.label,
        }
        for item in result.predictions
    ]
    frame = pd.DataFrame(rows).set_index("Version")

    st.subheader("Predictions for each image version")
    st.bar_chart(frame[["AI probability"]], height=280)
    st.dataframe(frame, use_container_width=True)

    consistency = result.consistency
    stats = st.columns(5)
    stats[0].metric("Average", f"{consistency.mean:.3f}")
    stats[1].metric("Minimum", f"{consistency.minimum:.3f}")
    stats[2].metric("Maximum", f"{consistency.maximum:.3f}")
    stats[3].metric("Std. deviation", f"{consistency.std:.3f}")
    stats[4].metric("Range", f"{consistency.score_range:.3f}")
    st.caption(
        f"{consistency.agreement:.0%} of the {consistency.num_versions} image versions "
        f"agreed with the original image's verdict."
    )


def render_charts(result: PipelineResult) -> None:
    """Render the drift and confidence-component charts."""

    charts = (result.explanation.charts if result.explanation else {}) or {}

    drift = charts.get("transformation_scores")
    if drift and drift["labels"]:
        st.subheader(drift["title"])
        st.caption("Positive bars mean the transformation pushed the score toward AI-generated.")
        st.bar_chart(pd.DataFrame({"Drift": drift["values"]}, index=drift["labels"]), height=260)

    components = charts.get("confidence_components")
    if components:
        st.subheader("Confidence breakdown")
        st.bar_chart(
            pd.DataFrame({"Score": components["values"]}, index=components["labels"]),
            height=240,
        )


def render_patches(result: PipelineResult) -> None:
    """Patch-risk heatmap, highest-risk regions, and the honest caveats."""

    st.subheader("Patch-level risk map")
    report = result.patches
    if report is None:
        st.info("Patch-level analysis was not run for this image.")
        return
    if not report.available:
        st.warning(report.message)
        return

    overlay = overlay_patch_heatmap(result.original_image, report)
    left, right = st.columns(2)
    if result.original_image is not None:
        left.image(result.original_image, caption="Original", use_container_width=True)
    if overlay is not None:
        right.image(
            overlay,
            caption=f"Patch risk ({len(report.patches)} patches; top {len(report.top_patches)} outlined)",
            use_container_width=True,
        )

    stats = st.columns(4)
    stats[0].metric("Patches scored", len(report.patches))
    stats[1].metric("Mean patch p(AI)", f"{report.mean_probability:.1%}")
    stats[2].metric("Max patch p(AI)", f"{report.max_probability:.1%}")
    stats[3].metric("Patch agreement", f"{report.agreement:.0%}")

    if report.coverage is not None and float(report.coverage.mean()) < 0.999:
        st.caption(
            f"Patches covered {report.coverage.mean():.0%} of the image; uncovered areas are "
            f"left untinted because nothing was measured there, not because they look authentic."
        )

    st.markdown("**Highest-risk regions**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rank": rank,
                    "x": patch.x,
                    "y": patch.y,
                    "width": patch.width,
                    "height": patch.height,
                    "p(AI)": round(patch.ai_probability, 4),
                }
                for rank, patch in enumerate(report.top_patches, start=1)
            ]
        ).set_index("Rank"),
        use_container_width=True,
    )
    st.caption(
        "A warm patch means the detector responded strongly to that region. It is "
        "a potentially manipulated region worth a human look - not proof that the "
        "region was edited."
    )


def render_explainability(result: PipelineResult) -> None:
    """Show the Grad-CAM overlay, or the reason it is unavailable."""

    st.subheader("Suspicious-region heatmap")
    explanation = result.explanation
    if explanation is None:
        st.info("Explainability was not run for this image.")
        return
    if explanation.available and explanation.overlay is not None:
        left, right = st.columns(2)
        if result.original_image is not None:
            left.image(result.original_image, caption="Original", use_container_width=True)
        right.image(explanation.overlay, caption="Grad-CAM overlay", use_container_width=True)
        st.caption(explanation.message)
    else:
        st.warning(explanation.message)


def render_errors(result: PipelineResult) -> None:
    if not result.errors:
        return
    with st.expander(f"{len(result.errors)} non-fatal issue(s) during analysis"):
        for error in result.errors:
            st.write(f"**{error.get('stage', 'pipeline')}**: {error.get('error')}")


def main() -> None:
    st.set_page_config(page_title="Robust AI Image Detector", page_icon="🔍", layout="wide")
    st.title("🔍 Robust Detection of AI-Generated Images")
    st.caption(
        "Image-level detection under real-world transformations. Results are "
        "confidence estimates from a hackathon-scale model, not proof."
    )

    with st.sidebar:
        st.header("Setup")
        config_path = st.text_input("Config file", DEFAULT_CONFIG)
        checkpoint_input = st.text_input("Model checkpoint", DEFAULT_CHECKPOINT)
        device_choice = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)

    try:
        config = load_config(config_path)
        checkpoint_path = resolve_config_path(config, checkpoint_input)
        pipeline = load_pipeline(
            str(Path(config_path).expanduser().resolve()), str(checkpoint_path), device_choice
        )
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    except ModelSetupError as exc:
        st.error("**Model setup required**")
        st.code(str(exc))
        st.info(
            "This demo is inference-only and needs a trained checkpoint. "
            "See `models/README.md` for the expected format."
        )
        st.stop()

    with st.sidebar:
        st.divider()
        st.header("Model")
        summary = pipeline.bundle.summary()
        st.write(f"**Architecture:** `{summary['architecture']}`")
        st.write(f"**Parameters:** {summary['parameters_millions']:.2f} M")
        st.write(f"**Device:** `{summary['device']}`")
        st.write(f"**Transformations:** {len(pipeline.transform_specs)}")
        st.caption(
            "Under the 2B parameter limit"
            if summary["under_2b_parameter_limit"]
            else "⚠️ Above the 2B parameter limit"
        )

    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False
    )
    if uploaded is None:
        st.info("Upload a JPG, JPEG, PNG or WEBP image to begin.")
        return

    try:
        with st.spinner("Running the pipeline across every transformed version..."):
            result = pipeline.analyse_bytes(uploaded.getvalue(), uploaded.name)
    except ImageValidationError as exc:
        st.error(f"Invalid image: {exc}")
        return
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        return

    image_column, result_column = st.columns([1, 1.4])
    with image_column:
        if result.original_image is not None:
            st.image(result.original_image, caption="Uploaded image", use_container_width=True)
        render_metadata(result)
    with result_column:
        render_headline(result)

    st.divider()
    render_per_transformation(result)
    st.divider()
    render_patches(result)
    st.divider()
    render_charts(result)
    st.divider()
    render_explainability(result)
    render_errors(result)

    if not is_calibrated(pipeline):
        st.info(
            "**Uncalibrated model score.** No calibration parameters are loaded, so the "
            "probability is the model's raw score rather than a statistically calibrated "
            "one. Fit calibration with `scripts/calibrate_threshold.py` on labelled "
            "validation data to make these numbers comparable across checkpoints."
        )

    detailed = result.as_detailed_dict()
    st.download_button(
        "Download detailed JSON",
        data=json.dumps(detailed, indent=2),
        file_name=f"{Path(result.image_path).stem or 'result'}_detection.json",
        mime="application/json",
    )
    with st.expander("Compact report schema"):
        st.json(result.as_report_dict())
    with st.expander("Full JSON result"):
        st.json(detailed)


if __name__ == "__main__":
    main()
