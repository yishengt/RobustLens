"""Minimal Streamlit demo for a trained detector checkpoint."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference.predict import load_detector, predict_pil_image
from src.utils.config import load_config, resolve_config_path


@st.cache_resource(show_spinner="Loading detector checkpoint...")
def cached_detector(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    return config, load_detector(config, checkpoint_path)


def main() -> None:
    st.set_page_config(page_title="AI Image Detector", page_icon="🔍")
    st.title("Robust AI-Generated Image Detector")
    st.caption("Upload an image to estimate the probability that it is AI-generated.")

    with st.sidebar:
        config_path = st.text_input("Config", "configs/config.yaml")
        checkpoint_arg = st.text_input("Checkpoint", "checkpoints/best.pt")

    try:
        config = load_config(config_path)
        checkpoint_path = resolve_config_path(config, checkpoint_arg)
        config, (model, transform, device, _) = cached_detector(
            str(Path(config_path).expanduser().resolve()), str(checkpoint_path)
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Upload a supported image to begin.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
        probability_ai = predict_pil_image(image, model, transform, device)
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"Could not process uploaded image: {exc}")
        return

    probability_real = 1.0 - probability_ai
    threshold = float(config.get("inference", {}).get("threshold", 0.5))
    label = "AI-generated" if probability_ai >= threshold else "Real"
    confidence = max(probability_ai, probability_real)

    st.image(image, caption="Uploaded image", use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("AI-generated probability", f"{probability_ai:.2%}")
    col2.metric("Real-image probability", f"{probability_real:.2%}")
    st.subheader(f"Classification: {label}")
    st.write(f"Confidence score: **{confidence:.2%}**")


if __name__ == "__main__":
    main()
