#!/usr/bin/env python3
"""Variance-based static-frame attack on GOTCHA videos.

This attack exploits a statistical fingerprint left by mismatched grain sizes:
text regions drawn with a different noise grain than the background exhibit
a different local-variance profile.  Computing each pixel's absolute deviation
from its local mean, then smoothing the result, reveals the hidden glyphs
without any temporal information.

When run against a video, the attack processes every frame independently and
aggregates the results with a pixel-wise maximum.  Phase-sliced reveals hide
some digits per frame, but the aggregate recovers all of them.

Usage:
    python attack_variance.py defended.mp4 --output-dir attack_runs/variance
    python attack_variance.py frame.png --output-dir attack_runs/variance_frame
    python attack_variance.py defended.mp4 --diagnostic --output-dir attack_runs/variance
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageFilter
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: pip install numpy pillow imageio imageio-ffmpeg"
    ) from exc

from attack_bench import (
    image_metrics,
    normalize_image,
    readability_proxy_score,
    save_grayscale_png,
)

DEFAULT_OUTPUT_DIR = "attack_variance"
DEFAULT_KERNEL_SIZE = 3
DEFAULT_BLUR_SIGMA = 8.0
DEFAULT_AGGREGATION = "max"

VIDEO_EXTENSIONS = frozenset((".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the variance attack against a GOTCHA clip or single frame.  "
            "Exploits grain-size mismatch between text and background."
        )
    )
    parser.add_argument("input", help="Input video or image path.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output artifacts (default {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--report-name",
        default="report.json",
        help="JSON report filename written inside --output-dir.",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=DEFAULT_KERNEL_SIZE,
        help="Local-mean kernel size for the deviation step (default 3).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_BLUR_SIGMA,
        help="Gaussian blur strength for smoothing the deviation field (default 8).",
    )
    parser.add_argument(
        "--aggregation",
        choices=("max", "mean"),
        default=DEFAULT_AGGREGATION,
        help="How to combine per-frame signals in video mode (default max).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Keep every Nth frame from video input (default 1).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process, 0 means all (default 0).",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Save intermediate stages for the first frame.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.kernel <= 0:
        raise SystemExit("--kernel must be positive.")
    if args.kernel % 2 == 0:
        raise SystemExit("--kernel must be odd.")
    if args.sigma <= 0:
        raise SystemExit("--sigma must be positive.")
    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be positive.")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be 0 or greater.")


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def extract_signal(
    gray: np.ndarray,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
) -> np.ndarray:
    """Recover hidden glyph field from a single grayscale uint8 frame.

    1. Compute absolute deviation from local mean (BoxBlur).
    2. Smooth the deviation field (GaussianBlur).

    Returns an unnormalized float32 field.  Caller handles final
    normalization so that multi-frame aggregation is meaningful.
    """
    img = Image.fromarray(gray)
    box_radius = max(1, (kernel_size - 1) // 2)
    local_mean = np.array(
        img.filter(ImageFilter.BoxBlur(radius=box_radius)),
        dtype=np.float32,
    )
    deviation = np.abs(gray.astype(np.float32) - local_mean)
    dev_img = Image.fromarray(np.clip(deviation, 0, 255).astype(np.uint8))
    blur_radius = max(1, int(round(blur_sigma)))
    return np.array(
        dev_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
        dtype=np.float32,
    )


def load_image_grayscale(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def load_video_grayscale(
    path: Path,
    frame_step: int,
    max_frames: int,
) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    frames: list[np.ndarray] = []
    try:
        for index, frame in enumerate(reader):
            if index % frame_step != 0:
                continue
            if frame.ndim == 3:
                gray = (
                    0.299 * frame[..., 0].astype(np.float32)
                    + 0.587 * frame[..., 1].astype(np.float32)
                    + 0.114 * frame[..., 2].astype(np.float32)
                )
                frames.append(np.clip(gray, 0, 255).astype(np.uint8))
            else:
                frames.append(frame.astype(np.uint8))
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        reader.close()
    return frames


def aggregate_signals(
    signals: list[np.ndarray],
    mode: str,
) -> np.ndarray:
    stacked = np.stack(signals, axis=0)
    if mode == "max":
        return np.max(stacked, axis=0)
    return np.mean(stacked, axis=0)


def save_diagnostics(
    gray: np.ndarray,
    kernel_size: int,
    blur_sigma: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(arr: np.ndarray, name: str) -> None:
        lo, hi = float(arr.min()), float(arr.max())
        norm = (arr - lo) / max(hi - lo, 1e-9)
        Image.fromarray((norm * 255).astype(np.uint8)).save(out_dir / name)

    img = Image.fromarray(gray)
    save(gray.astype(np.float32), "00_input.png")

    blur_radius = max(1, int(round(blur_sigma)))
    naive = np.array(
        img.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
        dtype=np.float32,
    )
    save(naive, "01_naive_blur_FAILS.png")

    box_radius = max(1, (kernel_size - 1) // 2)
    local_mean = np.array(
        img.filter(ImageFilter.BoxBlur(radius=box_radius)),
        dtype=np.float32,
    )
    save(np.abs(gray.astype(np.float32) - local_mean), "02_local_deviation.png")
    save(extract_signal(gray, kernel_size, blur_sigma), "03_attack_output.png")


def run_attack(args: argparse.Namespace) -> dict:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    if is_video_path(input_path):
        frames = load_video_grayscale(input_path, args.frame_step, args.max_frames)
        if not frames:
            raise SystemExit("No frames loaded from video.")
        signals = [extract_signal(f, args.kernel, args.sigma) for f in frames]
        combined = aggregate_signals(signals, args.aggregation)
        mode = "video"
        frame_count = len(frames)
        if args.diagnostic:
            save_diagnostics(frames[0], args.kernel, args.sigma, output_dir / "diagnostic")
    else:
        gray = load_image_grayscale(input_path)
        combined = extract_signal(gray, args.kernel, args.sigma)
        mode = "frame"
        frame_count = 1
        if args.diagnostic:
            save_diagnostics(gray, args.kernel, args.sigma, output_dir / "diagnostic")

    elapsed = time.perf_counter() - start
    normalized = normalize_image(combined)
    metrics = image_metrics(normalized)
    score = readability_proxy_score(metrics)

    output_path = output_dir / "variance_attack.png"
    save_grayscale_png(normalized, output_path)

    return {
        "algorithm": "variance",
        "input": str(input_path),
        "mode": mode,
        "frame_count": frame_count,
        "parameters": {
            "kernel_size": args.kernel,
            "blur_sigma": args.sigma,
            "aggregation": args.aggregation if mode == "video" else None,
            "frame_step": args.frame_step if mode == "video" else None,
        },
        "seconds": round(elapsed, 4),
        "selection_score": score,
        "metrics": metrics,
        "output_image": str(output_path),
    }


def print_summary(result: dict) -> None:
    metrics = result["metrics"]
    print(
        f"{'mode':<8} {'frames':>7} {'seconds':>8} {'score':>8} {'otsu':>8}  output"
    )
    print(
        f"{result['mode']:<8} "
        f"{result['frame_count']:>7} "
        f"{result['seconds']:>8.4f} "
        f"{result['selection_score']:>8.4f} "
        f"{metrics['otsu_separation']:>8.4f}  "
        f"{result['output_image']}"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")

    result = run_attack(args)

    output_dir = Path(args.output_dir)
    report_path = output_dir / args.report_name
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print_summary(result)
    print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    main()
