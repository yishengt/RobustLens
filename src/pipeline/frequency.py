"""Stage 9 (optional): frequency and noise-residual analysis.

Generative models leave characteristic traces in the frequency domain -
upsampling grids, unusually clean high-frequency energy, and low-variance noise
residuals. This module extracts descriptive features for those traces.

It is **disabled by default** and is deliberately conservative: descriptive
features are always available, but :func:`frequency_probability` returns
``None`` unless a trained frequency classifier is configured. The module never
invents a probability, and importing it never breaks the main pipeline - SciPy
is optional and the DCT features are simply omitted when it is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:  # SciPy is optional; only the DCT features need it.
    from scipy.fft import dctn as _dctn

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without SciPy
    _dctn = None
    SCIPY_AVAILABLE = False

DEFAULT_FFT_BINS = 16


@dataclass
class FrequencyFeatures:
    """Descriptive frequency-domain statistics for one image."""

    fft_radial_profile: List[float] = field(default_factory=list)
    fft_high_frequency_ratio: float = 0.0
    dct_energy_ratio: Optional[float] = None
    highpass_energy: float = 0.0
    noise_residual_std: float = 0.0
    available_features: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fft_radial_profile": [round(value, 6) for value in self.fft_radial_profile],
            "fft_high_frequency_ratio": round(self.fft_high_frequency_ratio, 6),
            "dct_energy_ratio": (
                None if self.dct_energy_ratio is None else round(self.dct_energy_ratio, 6)
            ),
            "highpass_energy": round(self.highpass_energy, 6),
            "noise_residual_std": round(self.noise_residual_std, 6),
            "available_features": list(self.available_features),
            "notes": list(self.notes),
        }


def is_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when frequency analysis is switched on in the config."""

    return bool((config or {}).get("frequency", {}).get("enabled", False))


def _grayscale(image: Image.Image) -> np.ndarray:
    """Return the image as a float32 luminance array scaled to ``[0, 1]``."""

    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def fft_spectrum(image: Image.Image) -> np.ndarray:
    """Return the centred log-magnitude FFT spectrum of the luminance channel."""

    gray = _grayscale(image)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(spectrum)).astype(np.float32)


def radial_profile(spectrum: np.ndarray, bins: int = DEFAULT_FFT_BINS) -> np.ndarray:
    """Average spectrum energy in ``bins`` rings from the centre outwards.

    Upsampling artefacts from generative decoders often show up as bumps in the
    outer rings of this profile.
    """

    bins = max(1, int(bins))
    height, width = spectrum.shape
    centre_y, centre_x = (height - 1) / 2.0, (width - 1) / 2.0
    y_grid, x_grid = np.ogrid[:height, :width]
    radius = np.sqrt((y_grid - centre_y) ** 2 + (x_grid - centre_x) ** 2)
    max_radius = float(radius.max()) or 1.0

    indices = np.clip((radius / max_radius * bins).astype(np.int64), 0, bins - 1)
    totals = np.bincount(indices.ravel(), weights=spectrum.ravel(), minlength=bins)
    counts = np.bincount(indices.ravel(), minlength=bins)
    return (totals / np.maximum(counts, 1)).astype(np.float64)


def high_frequency_ratio(profile: np.ndarray) -> float:
    """Share of the radial profile's energy living in its outer half."""

    profile = np.asarray(profile, dtype=np.float64)
    total = float(profile.sum())
    if total <= 0:
        return 0.0
    return float(profile[len(profile) // 2 :].sum() / total)


def dct_features(image: Image.Image) -> Optional[float]:
    """Return the share of DCT energy outside the low-frequency corner.

    Returns ``None`` when SciPy is unavailable.
    """

    if not SCIPY_AVAILABLE:
        return None
    gray = _grayscale(image)
    coefficients = np.abs(_dctn(gray, norm="ortho"))
    total = float(coefficients.sum())
    if total <= 0:
        return 0.0
    height, width = coefficients.shape
    low = coefficients[: max(1, height // 8), : max(1, width // 8)].sum()
    return float((total - low) / total)


def highpass_filter(image: Image.Image) -> np.ndarray:
    """Apply a 3x3 Laplacian high-pass filter to the luminance channel."""

    gray = _grayscale(image)
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel).astype(np.float32)


def noise_residual(image: Image.Image) -> np.ndarray:
    """Return the image minus a 3x3 box-blurred copy of itself.

    Camera sensors leave a rich residual here; many generated images do not.
    """

    gray = _grayscale(image)
    padded = np.pad(gray, 1, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    smoothed = windows.mean(axis=(-2, -1)).astype(np.float32)
    return (gray - smoothed).astype(np.float32)


def extract_features(
    image: Image.Image, config: Optional[Dict[str, Any]] = None
) -> FrequencyFeatures:
    """Compute every available frequency and noise feature for one image."""

    bins = int((config or {}).get("frequency", {}).get("fft_bins", DEFAULT_FFT_BINS))
    features = FrequencyFeatures()
    notes: List[str] = []

    profile = radial_profile(fft_spectrum(image), bins)
    features.fft_radial_profile = [float(value) for value in profile]
    features.fft_high_frequency_ratio = high_frequency_ratio(profile)
    features.available_features.extend(["fft_radial_profile", "fft_high_frequency_ratio"])

    dct_ratio = dct_features(image)
    if dct_ratio is None:
        notes.append("SciPy is not installed; DCT features were skipped.")
    else:
        features.dct_energy_ratio = dct_ratio
        features.available_features.append("dct_energy_ratio")

    features.highpass_energy = float(np.mean(np.abs(highpass_filter(image))))
    features.noise_residual_std = float(np.std(noise_residual(image)))
    features.available_features.extend(["highpass_energy", "noise_residual_std"])

    features.notes = notes
    return features


def frequency_probability(
    image: Image.Image, config: Optional[Dict[str, Any]] = None
) -> tuple[Optional[float], Optional[str]]:
    """Return ``(probability, reason_unavailable)`` from a frequency model.

    The features above are descriptive, not calibrated: turning them into a
    probability requires a trained frequency classifier. Until one is provided
    via ``frequency.checkpoint``, this returns ``(None, reason)`` so fusion can
    fall back cleanly instead of fabricating a score.
    """

    settings = (config or {}).get("frequency", {}) or {}
    if not settings.get("enabled", False):
        return None, "Frequency analysis is disabled in the configuration."

    checkpoint = settings.get("checkpoint")
    if not checkpoint:
        return None, (
            "Frequency analysis is enabled but no frequency.checkpoint is configured, "
            "so no calibrated frequency probability is available."
        )
    if not Path(str(checkpoint)).expanduser().exists():
        return None, f"Frequency model checkpoint not found: {checkpoint}"

    return None, (
        "A frequency checkpoint was found but no frequency classifier is implemented "
        "in this hackathon build; only descriptive features are produced."
    )
