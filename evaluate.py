"""
Synthetic benchmark for the adaptive restoration pipeline.

Takes clean reference images (scikit-image's bundled samples), applies known
degradations, runs adaptive_restoration() on the degraded copy, and reports
PSNR / SSIM of (a) the degraded input and (b) the restored output, both
measured against the clean reference.

Usage:
    python evaluate.py
"""

import os
import sys

import numpy as np
from skimage import data
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "streamlit"))

from restore import adaptive_restoration  # noqa: E402

SEED = 0


def degrade_dark(img, rng):
    """Underexposure: scale intensities to ~35% of the original."""
    return np.clip(img.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)


def degrade_low_contrast(img, rng):
    """Washed out: compress intensities toward mid-grey (std roughly 1/3)."""
    out = img.astype(np.float32) * 0.3 + 0.35 * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_haze(img, rng, t=0.5, airlight=230.0):
    """Atmospheric scattering model: I = J*t + A*(1 - t)."""
    out = img.astype(np.float32) * t + airlight * (1 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_noise(img, rng, sigma=25.0):
    """Additive Gaussian noise, sigma in 8-bit levels."""
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_dark_noise(img, rng):
    return degrade_noise(degrade_dark(img, rng), rng, sigma=15.0)


DEGRADATIONS = {
    "underexposed": degrade_dark,
    "low contrast": degrade_low_contrast,
    "haze": degrade_haze,
    "noise (sigma=25)": degrade_noise,
    "underexposed + noise (sigma=15)": degrade_dark_noise,
}

REFERENCES = {
    "astronaut": data.astronaut,
    "chelsea": data.chelsea,
    "coffee": data.coffee,
    "rocket": data.rocket,
}


def score(reference, candidate):
    psnr = peak_signal_noise_ratio(reference, candidate, data_range=255)
    ssim = structural_similarity(reference, candidate, channel_axis=2, data_range=255)
    return psnr, ssim


def main():
    rng = np.random.default_rng(SEED)
    rows = []  # (degradation, n, psnr_in, ssim_in, psnr_out, ssim_out)

    for deg_name, deg_fn in DEGRADATIONS.items():
        acc = []
        for ref_name, loader in REFERENCES.items():
            clean_rgb = loader()
            degraded_rgb = deg_fn(clean_rgb, rng)

            # adaptive_restoration works on BGR (OpenCV convention); convert
            # in and out so the metrics compare RGB with RGB.
            degraded_bgr = np.ascontiguousarray(degraded_rgb[:, :, ::-1])
            restored_bgr, _ = adaptive_restoration(degraded_bgr)
            restored_rgb = np.ascontiguousarray(restored_bgr[:, :, ::-1])

            p_in, s_in = score(clean_rgb, degraded_rgb)
            p_out, s_out = score(clean_rgb, restored_rgb)
            acc.append((p_in, s_in, p_out, s_out))

        arr = np.array(acc)
        rows.append((deg_name, len(acc), *arr.mean(axis=0)))

    header = f"{'degradation':34s} {'n':>2s} | {'PSNR in':>8s} {'PSNR out':>9s} {'delta':>7s} | {'SSIM in':>8s} {'SSIM out':>9s} {'delta':>7s}"
    print(header)
    print("-" * len(header))
    for name, n, p_in, s_in, p_out, s_out in rows:
        print(
            f"{name:34s} {n:2d} | {p_in:8.2f} {p_out:9.2f} {p_out - p_in:+7.2f} "
            f"| {s_in:8.3f} {s_out:9.3f} {s_out - s_in:+7.3f}"
        )


if __name__ == "__main__":
    main()
