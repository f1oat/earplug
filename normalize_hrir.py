#!/usr/bin/env python3
"""Offline HRIR normalization/compensation for earplug_data.txt.

This script computes a global diffuse-field style magnitude target, designs a
regularized compensation filter, applies it to all HRIRs, and writes a new
earplug-compatible dataset.

Defaults are intentionally conservative:
- 1/3 octave smoothing
- limited boost/cut
- minimum-phase common compensation filter (same for all directions/ears)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ELEV_AZ_COUNTS = np.array([29, 31, 37, 37, 37, 37, 37, 31, 29, 23, 19, 13, 7], dtype=int)
ELEV_OFFSETS = np.array([0, 29, 60, 97, 134, 171, 208, 245, 276, 305, 328, 347, 360], dtype=int)
EPS = 1e-12
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_earplug_data(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse earplug_data.txt style file into [N, 2, 128] float array."""
    headers: list[str] = []
    rows: list[np.ndarray] = []
    pending_header: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("*"):
                pending_header = stripped
                continue

            nums = [float(m.group(0)) for m in FLOAT_RE.finditer(line)]
            if len(nums) < 256:
                continue

            vals = np.asarray(nums[:256], dtype=np.float64)
            left = vals[0::2]
            right = vals[1::2]
            rows.append(np.stack((left, right), axis=0))

            if pending_header is None:
                idx = len(rows) - 1
                pending_header = f"** Impulse Index: {idx}, Sub Index: {idx}, File: generated.wav **"
            headers.append(pending_header)
            pending_header = None

    if not rows:
        raise ValueError(f"No HRIR rows parsed from {path}")

    data = np.stack(rows, axis=0)
    if data.shape[1:] != (2, 128):
        raise ValueError(f"Unexpected HRIR tensor shape {data.shape}")

    return headers, data


def write_earplug_data(path: Path, headers: list[str], data: np.ndarray) -> None:
    """Write [N,2,128] data with earplug-compatible header/data line pairs.

    Trailing whitespace on each data line is intentional to preserve compatibility
    with the legacy parse-to-h.py splitter.
    """
    n = data.shape[0]
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            header = headers[i] if i < len(headers) else f"** Impulse Index: {i}, Sub Index: {i}, File: generated.wav **"
            f.write(header.rstrip() + "\n")

            vals = np.empty(256, dtype=np.float64)
            vals[0::2] = data[i, 0]
            vals[1::2] = data[i, 1]
            line = " ".join(f"{v:.6f}" for v in vals) + " \n"
            f.write(line)
            f.write("\n")


def fractional_octave_smooth(db: np.ndarray, freqs: np.ndarray, frac_oct: float) -> np.ndarray:
    """Smooth magnitude in dB with a fractional-octave moving average."""
    out = np.empty_like(db)
    if frac_oct <= 0:
        out[:] = db
        return out

    ratio = 2.0 ** (0.5 / frac_oct)
    for i, f in enumerate(freqs):
        if f <= 0.0:
            out[i] = db[i]
            continue
        lo = f / ratio
        hi = f * ratio
        mask = (freqs >= lo) & (freqs <= hi)
        if np.any(mask):
            out[i] = float(np.mean(db[mask]))
        else:
            out[i] = db[i]
    return out


def raised_cosine(freqs: np.ndarray, start_hz: float, stop_hz: float, increasing: bool) -> np.ndarray:
    """Raised-cosine ramp from 0..1 (increasing) or 1..0 (decreasing)."""
    if stop_hz <= start_hz:
        return np.ones_like(freqs)
    x = np.clip((freqs - start_hz) / (stop_hz - start_hz), 0.0, 1.0)
    if increasing:
        return 0.5 - 0.5 * np.cos(np.pi * x)
    return 0.5 + 0.5 * np.cos(np.pi * x)


def minimum_phase_from_mag(comp_mag: np.ndarray, nfft: int, out_len: int) -> np.ndarray:
    """Create a minimum-phase FIR from one-sided magnitude response."""
    log_mag = np.log(np.maximum(comp_mag, EPS))
    full_log_mag = np.concatenate([log_mag, log_mag[-2:0:-1]])

    cep = np.fft.ifft(full_log_mag).real
    cep_min = np.zeros_like(cep)
    cep_min[0] = cep[0]
    if nfft % 2 == 0:
        cep_min[1 : nfft // 2] = 2.0 * cep[1 : nfft // 2]
        cep_min[nfft // 2] = cep[nfft // 2]
    else:
        cep_min[1 : (nfft + 1) // 2] = 2.0 * cep[1 : (nfft + 1) // 2]

    min_spec = np.exp(np.fft.fft(cep_min))
    ir = np.fft.ifft(min_spec).real
    return ir[:out_len]


def build_compensation(
    hrir: np.ndarray,
    sample_rate: float,
    nfft: int,
    strength: float,
    smooth_frac_oct: float,
    max_boost_db: float,
    max_cut_db: float,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (freqs, diffuse_field_db, smooth_df_db, comp_db)."""
    h = np.fft.rfft(hrir, n=nfft, axis=-1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)

    mag = np.maximum(np.abs(h), EPS)
    df_db = (20.0 / np.log(10.0)) * np.mean(np.log(mag), axis=(0, 1))

    ref_mask = (freqs >= 700.0) & (freqs <= 1500.0)
    if np.any(ref_mask):
        df_db = df_db - float(np.mean(df_db[ref_mask]))

    smooth_df_db = fractional_octave_smooth(df_db, freqs, smooth_frac_oct)

    comp_db = -strength * smooth_df_db
    comp_db = np.clip(comp_db, -abs(max_cut_db), abs(max_boost_db))

    # Fade compensation in/out to avoid extreme LF/HF behavior.
    nyq = sample_rate * 0.5
    low_start = max(0.0, low_hz * 0.5)
    low_stop = max(low_start + 1.0, low_hz)
    high_start = min(high_hz, nyq)
    high_stop = min(nyq, high_hz * 1.25)

    band = raised_cosine(freqs, low_start, low_stop, increasing=True)
    if high_start < nyq:
        band *= raised_cosine(freqs, high_start, high_stop, increasing=False)

    comp_db *= band
    comp_db[0] = 0.0
    return freqs, df_db, smooth_df_db, comp_db


def apply_common_filter(hrir: np.ndarray, comp_ir: np.ndarray) -> np.ndarray:
    """Convolve each HRIR with common compensation FIR and trim to 128 taps."""
    n, _, taps = hrir.shape
    out = np.zeros_like(hrir)
    for i in range(n):
        for ear in range(2):
            y = np.convolve(hrir[i, ear], comp_ir, mode="full")
            out[i, ear] = y[:taps]
    return out


def compute_bump_vs_azimuth(
    hrir: np.ndarray,
    sample_rate: float,
    nfft: int,
    elev_ring_index: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return azimuth degrees and 2.5k bump (L/R) for an elevation ring."""
    start = int(ELEV_OFFSETS[elev_ring_index])
    count = int(ELEV_AZ_COUNTS[elev_ring_index])
    idx = np.arange(start, start + count)

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    h = np.fft.rfft(hrir[idx], n=nfft, axis=-1)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), EPS))

    k_25 = int(np.argmin(np.abs(freqs - 2500.0)))
    ref_mask = (freqs >= 1500.0) & (freqs <= 4000.0) & (np.abs(freqs - 2500.0) >= 300.0)
    ref = np.mean(mag_db[:, :, ref_mask], axis=2)
    bump = mag_db[:, :, k_25] - ref

    az_deg = np.linspace(0.0, 180.0, count)
    return az_deg, bump[:, 0], bump[:, 1]


def save_plots(
    out_dir: Path,
    plot_prefix: str,
    freqs: np.ndarray,
    df_before_db: np.ndarray,
    df_after_db: np.ndarray,
    smooth_df_db: np.ndarray,
    comp_db: np.ndarray,
    az_deg: np.ndarray,
    bump_l_before: np.ndarray,
    bump_r_before: np.ndarray,
    bump_l_after: np.ndarray,
    bump_r_after: np.ndarray,
) -> dict[str, str]:
    out_paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(freqs[1:], df_before_db[1:], label="Diffuse-field before", linewidth=1.2)
    ax.semilogx(freqs[1:], df_after_db[1:], label="Diffuse-field after", linewidth=1.2)
    ax.semilogx(freqs[1:], smooth_df_db[1:], label="Smoothed before", linestyle="--", linewidth=1.0)
    ax.semilogx(freqs[1:], comp_db[1:], label="Applied compensation", linestyle=":", linewidth=1.4)
    ax.set_title("Diffuse-field response and applied compensation")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level (dB)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    p1 = out_dir / f"{plot_prefix}_diffuse_field.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    out_paths["diffuse_field_plot"] = str(p1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(az_deg, bump_l_before, label="Left before", linewidth=1.2)
    axes[0].plot(az_deg, bump_l_after, label="Left after", linewidth=1.2)
    axes[0].set_ylabel("2.5 kHz bump (dB)")
    axes[0].set_title("Elevation 0 deg: 2.5 kHz bump vs azimuth")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(az_deg, bump_r_before, label="Right before", linewidth=1.2)
    axes[1].plot(az_deg, bump_r_after, label="Right after", linewidth=1.2)
    axes[1].set_xlabel("Azimuth (deg on front hemisphere)")
    axes[1].set_ylabel("2.5 kHz bump (dB)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    p2 = out_dir / f"{plot_prefix}_elev0_2p5k_bump.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    out_paths["elev0_bump_plot"] = str(p2)

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize/compensate earplug HRIR dataset")
    parser.add_argument("--input", default="earplug_data.txt", help="Input earplug data text file")
    parser.add_argument("--output", default="earplug_data_compensated.txt", help="Output compensated data text file")
    parser.add_argument("--sample-rate", type=float, default=44100.0, help="Sampling rate in Hz")
    parser.add_argument("--nfft", type=int, default=2048, help="FFT length for response analysis")
    parser.add_argument("--comp-len", type=int, default=64, help="Length of common compensation FIR")
    parser.add_argument("--strength", type=float, default=0.8, help="Compensation strength [0..1]")
    parser.add_argument("--smooth-frac-oct", type=float, default=3.0, help="Smoothing bandwidth as 1/N octave")
    parser.add_argument("--max-boost-db", type=float, default=6.0, help="Maximum compensation boost (dB)")
    parser.add_argument("--max-cut-db", type=float, default=10.0, help="Maximum compensation cut (dB)")
    parser.add_argument("--low-hz", type=float, default=120.0, help="Compensation fade-in region center")
    parser.add_argument("--high-hz", type=float, default=16000.0, help="Compensation fade-out region start")
    parser.add_argument("--plot-prefix", default="hrir_norm", help="Prefix for output plot/metrics files")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if args.nfft < 256:
        raise ValueError("nfft must be >= 256")
    if args.comp_len < 8 or args.comp_len > 128:
        raise ValueError("comp-len must be between 8 and 128")

    headers, hrir = parse_earplug_data(in_path)
    if hrir.shape[0] != 368:
        print(f"Warning: expected 368 HRIRs, parsed {hrir.shape[0]}")

    freqs, df_before_db, smooth_df_db, comp_db = build_compensation(
        hrir=hrir,
        sample_rate=args.sample_rate,
        nfft=args.nfft,
        strength=float(np.clip(args.strength, 0.0, 1.0)),
        smooth_frac_oct=args.smooth_frac_oct,
        max_boost_db=args.max_boost_db,
        max_cut_db=args.max_cut_db,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
    )
    comp_mag = 10.0 ** (comp_db / 20.0)

    comp_ir = minimum_phase_from_mag(comp_mag, nfft=args.nfft, out_len=args.comp_len)
    hrir_comp = apply_common_filter(hrir, comp_ir)

    # Preserve overall RMS to avoid global level jumps.
    rms_before = float(np.sqrt(np.mean(hrir ** 2)))
    rms_after = float(np.sqrt(np.mean(hrir_comp ** 2)))
    if rms_after > EPS:
        hrir_comp *= rms_before / rms_after

    peak_before = float(np.max(np.abs(hrir)))
    peak_after = float(np.max(np.abs(hrir_comp)))

    write_earplug_data(out_path, headers, hrir_comp)

    h_after = np.fft.rfft(hrir_comp, n=args.nfft, axis=-1)
    df_after_db = (20.0 / np.log(10.0)) * np.mean(np.log(np.maximum(np.abs(h_after), EPS)), axis=(0, 1))
    ref_mask = (freqs >= 700.0) & (freqs <= 1500.0)
    if np.any(ref_mask):
        df_after_db = df_after_db - float(np.mean(df_after_db[ref_mask]))

    az, bump_l_before, bump_r_before = compute_bump_vs_azimuth(hrir, args.sample_rate, args.nfft, elev_ring_index=4)
    _, bump_l_after, bump_r_after = compute_bump_vs_azimuth(hrir_comp, args.sample_rate, args.nfft, elev_ring_index=4)

    metrics = {
        "input": str(in_path),
        "output": str(out_path),
        "num_hrirs": int(hrir.shape[0]),
        "sample_rate_hz": args.sample_rate,
        "nfft": args.nfft,
        "comp_len": args.comp_len,
        "strength": args.strength,
        "smooth_frac_oct": args.smooth_frac_oct,
        "max_boost_db": args.max_boost_db,
        "max_cut_db": args.max_cut_db,
        "low_hz": args.low_hz,
        "high_hz": args.high_hz,
        "peak_before": peak_before,
        "peak_after": peak_after,
        "rms_before": rms_before,
        "rms_after_post_gain": float(np.sqrt(np.mean(hrir_comp ** 2))),
        "bump25k_left_mean_before": float(np.mean(bump_l_before)),
        "bump25k_left_mean_after": float(np.mean(bump_l_after)),
        "bump25k_right_mean_before": float(np.mean(bump_r_before)),
        "bump25k_right_mean_after": float(np.mean(bump_r_after)),
        "bump25k_left_max_before": float(np.max(bump_l_before)),
        "bump25k_left_max_after": float(np.max(bump_l_after)),
        "bump25k_right_max_before": float(np.max(bump_r_before)),
        "bump25k_right_max_after": float(np.max(bump_r_after)),
    }

    out_dir = out_path.parent
    if not args.no_plots:
        plot_paths = save_plots(
            out_dir=out_dir,
            plot_prefix=args.plot_prefix,
            freqs=freqs,
            df_before_db=df_before_db,
            df_after_db=df_after_db,
            smooth_df_db=smooth_df_db,
            comp_db=comp_db,
            az_deg=az,
            bump_l_before=bump_l_before,
            bump_r_before=bump_r_before,
            bump_l_after=bump_l_after,
            bump_r_after=bump_r_after,
        )
        metrics.update(plot_paths)

    metrics_path = out_dir / f"{args.plot_prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote compensated HRIR dataset: {out_path}")
    print(f"Wrote metrics: {metrics_path}")
    if not args.no_plots:
        print("Wrote diagnostic plots:")
        print(f"  - {out_dir / (args.plot_prefix + '_diffuse_field.png')}")
        print(f"  - {out_dir / (args.plot_prefix + '_elev0_2p5k_bump.png')}")
    print(
        "2.5k bump mean (L/R) before -> after: "
        f"{metrics['bump25k_left_mean_before']:.2f}/{metrics['bump25k_right_mean_before']:.2f} dB -> "
        f"{metrics['bump25k_left_mean_after']:.2f}/{metrics['bump25k_right_mean_after']:.2f} dB"
    )


if __name__ == "__main__":
    main()
